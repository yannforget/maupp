"""Create OpenStreetMap database from .osm.pbf files and extract
relevant data.
"""

import subprocess
from tempfile import TemporaryDirectory
import zipfile

import fiona
import geojson
import psycopg2
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import polygonize
from shapely import wkt

from maupp.utils import download_file


# Seas & Oceans shapefile from openstreetmapdata.com
SHAPEFILE_URL = 'http://data.openstreetmapdata.com/water-polygons-split-4326.zip'


def geojson_like(response, description, geom_col='geom'):
    """Convert the response of an SQL query to a list of
    GeoJSON-like queries.

    Parameters
    ----------
    response : list
        Query response (as obtained with `cur.fetchall()`).
    description : list
        Cursor description (as obtained with `cur.description`).
    geom_col : str, optional
        Name of the geometry column.

    Returns
    -------
    features : list of geojson-like dict
        Output features.
    """
    colnames = [col.name for col in description]
    features = []
    for row in response:
        geom = wkt.loads(row[colnames.index(geom_col)])
        properties = {col: row[colnames.index(col)] for col in colnames
                      if col != geom_col}
        feature = geojson.Feature(geometry=mapping(geom),
                                  properties=properties)
        features.append(feature)
    return features


class OSMDatabase:

    def __init__(self, name, user, password, host='localhost'):
        """Access OSM PostGIS database.

        Parameters
        ----------
        name : str
            Database name.
        user : str
            Database user.
        pass : str
            Database password.
        host : str, optional
            Database hostname.
        """
        self.name = name
        self.user = user
        self.password = password
        self.host = host
        if not self.exists:
            self.create()
            self.connection = self.connect()
            self.enable_postgis()
        self.connection = self.connect()

    @property
    def exists(self):
        """Check if the OSM database exists."""
        connection = psycopg2.connect(
            database='postgres',
            user=self.user,
            password=self.password,
            host=self.host
        )
        cur = connection.cursor()
        cur.execute(
            'SELECT datname FROM pg_catalog.pg_database WHERE datname = %s',
            (self.name, )
        )
        response = cur.fetchone()
        connection.commit()
        connection.close()
        if not response:
            return False
        return self.name in response

    def create(self):
        """Create the OSM database."""
        connection = psycopg2.connect(
            database='postgres',
            user=self.user,
            password=self.password,
            host=self.host
        )
        connection.autocommit = True
        cur = connection.cursor()
        cur.execute(
            'CREATE DATABASE {} OWNER {};'.format(self.name, self.user))
        connection.commit()
        connection.close()
        return True

    def connect(self):
        """Connect to database. Return a connection object."""
        return psycopg2.connect(
            database=self.name,
            user=self.user,
            password=self.password,
            host=self.host
        )

    def enable_postgis(self):
        """Enable PostGIS extension on OSM database."""
        cur = self.connection.cursor()
        cur.execute('CREATE EXTENSION postgis;')
        cur.execute('CREATE EXTENSION hstore;')
        self.connection.commit()
        cur.close()
        return True

    @property
    def empty(self):
        """Check if OSM data has already been imported into the db."""
        cur = self.connection.cursor()
        sql = """SELECT EXISTS (
                   SELECT * FROM information_schema.tables
                   WHERE table_name = 'osm_point'
                 );
              """
        cur.execute(sql)
        response = cur.fetchone()
        cur.close()
        return not response[0]

    def data_in_aoi(self, aoi):
        """Check if OSM database contains data in a given AOI."""
        if self.empty:
            return False
        cur = self.connection.cursor()
        sql = """SELECT osm_id FROM osm_polygon WHERE ST_Intersects(
                   way, ST_Transform(ST_GeomFromText(%s, 4326), 3857))
                 LIMIT 1;
              """
        cur.execute(sql, (aoi, ))
        response = len(cur.fetchall()) >= 1
        cur.close()
        return response

    def import_data(self, osm_datafile):
        """Import an .osm.pbf file into the db.

        Parameters
        ----------
        osm_datafile : str
            Path to an .osm.pbf file.
        """
        cmd = ['osm2pgsql', '--database', self.name, '--username', self.user,
               '--host', self.host, '--prefix', 'osm', '--slim']
        if self.empty:
            cmd.append('--create')
        else:
            cmd.append('--append')
        cmd.append(osm_datafile)
        subprocess.run(cmd, env={'PGPASSWORD': self.password})
        return True

    def water(self, aoi):
        """Extract water bodies and oceans from the OSM database.

        Parameters
        ----------
        aoi : shapely polygon
            Area of interest in lat/lon coordinates.

        Returns
        -------
        features : list of geojson-like features
            Output features.
        """
        sql = """SELECT ST_AsText(ST_Transform(way, 4326)) AS geom
                 FROM osm_polygon
                 WHERE "natural"='water'
                   AND ST_Intersects(
                     way, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            water_bodies = geojson_like(cur.fetchall(), cur.description)

        sql = """SELECT ST_AsText(ST_Transform(geom, 4326)) AS geom
                 FROM ocean
                 WHERE ST_Intersects(
                   geom, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            oceans = geojson_like(cur.fetchall(), cur.description)
        return water_bodies + oceans

    def roads(self, aoi):
        """Extract roads from the OSM database.

        Parameters
        ----------
        aoi : shapely polygon
            Area of interest in lat/lon coordinates.

        Returns
        -------
        features : list of geojson-like features
            Output features.
        """
        sql = """SELECT ST_AsText(ST_Transform(way, 4326)) AS geom, highway
                 FROM osm_line
                 WHERE highway IS NOT NULL
                   AND ST_Intersects(
                     way, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            features = geojson_like(cur.fetchall(), cur.description)
        return features

    def buildings(self, aoi):
        """Extract building footprints from the OSM database.

        Parameters
        ----------
        aoi : shapely polygon
            Area of interest in lat/lon coordinates.

        Returns
        -------
        features : list of geojson-like features
            Output features.
        """
        sql = """SELECT ST_AsText(ST_Transform(way, 4326)) AS geom, building
                 FROM osm_polygon
                 WHERE building IS NOT NULL
                   AND ST_Intersects(
                     way, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            features = geojson_like(cur.fetchall(), cur.description)
        return features

    def landuse(self, aoi):
        """Extract land use polygons from the OSM database.

        Parameters
        ----------
        aoi : shapely polygon
            Area of interest in lat/lon coordinates.

        Returns
        -------
        features : list of geojson-like features
            Output features.
        """
        sql = """SELECT ST_AsText(ST_Transform(way, 4326)) AS geom, landuse
                 FROM osm_polygon
                 WHERE landuse IS NOT NULL
                   AND ST_Intersects(
                     way, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            features = geojson_like(cur.fetchall(), cur.description)
        return features

    def leisure(self, aoi):
        """Extract leisure polygons from the OSM database.

        Parameters
        ----------
        aoi : shapely polygon
            Area of interest in lat/lon coordinates.

        Returns
        -------
        features : list of geojson-like features
            Output features.
        """
        sql = """SELECT ST_AsText(ST_Transform(way, 4326)) AS geom, leisure
                 FROM osm_polygon
                 WHERE leisure IS NOT NULL
                   AND ST_Intersects(
                     way, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            features = geojson_like(cur.fetchall(), cur.description)
        return features

    def natural(self, aoi):
        """Extract natural polygons from the OSM database.

        Parameters
        ----------
        aoi : shapely polygon
            Area of interest in lat/lon coordinates.

        Returns
        -------
        features : list of geojson-like features
            Output features.
        """
        sql = """SELECT ST_AsText(ST_Transform(way, 4326)) AS geom, "natural"
                 FROM osm_polygon
                 WHERE "natural" IS NOT NULL
                   AND ST_Intersects(
                     way, ST_Transform(ST_GeomFromText(%s, 4326), 3857)
                 );
              """
        with self.connection.cursor() as cur:
            cur.execute(sql, (aoi.wkt, ))
            features = geojson_like(cur.fetchall(), cur.description)
        return features


def _two_points_line(feature):
    """Convert a Polyline to a Line composed of only two points."""
    features = []
    coords = feature['geometry']['coordinates']
    for i in range(0, len(coords) - 1):
        segment_coords = [coords[i], coords[i+1]]
        geom = geojson.LineString(segment_coords)
        features.append(geojson.Feature(geometry=geom))
    return features


def urban_blocks(features):
    """Generate urban blocks from a set of roads.

    Parameters
    ----------
    features : list of geojson-like dicts
        Roads as an iterable of GeoJSON-like dict.

    Returns
    -------
    blocks : list of geojson-like dict
        Urban blocks as a list of GeoJSON-like dict.
    """
    segments = []
    for feature in features:
        for linestring in _two_points_line(feature):
            segments.append(shape(linestring['geometry']))
    return [
        geojson.Feature(geometry=mapping(block))
        for block in polygonize(segments)
    ]


def geoextract_osmpbf(source, destination, bbox):
    """Make a geographic extract of an .osm.pbf file with osmium.

    Parameters
    ----------
    source : str
        Path to source .osm.pbf file.
    destination : str
        Path to output .osm.pbf file.
    bbox : tuple of float
        Bounding box to cut out (left, bottom, right, top).

    Returns
    -------
    destination : str
        Path to output .osm.pbf file.
    """
    bbox = ','.join([str(coord) for coord in bbox])
    cmd = ['osmium', 'extract', '--bbox',
           bbox, '--output', destination, source]
    subprocess.run(cmd)
    return destination


def _find_shp_path(archive):
    """Find path to .shp file in a .zip archive."""
    with open(archive, 'rb') as src:
        zfile = zipfile.ZipFile(src)
        for member in zfile.infolist():
            if member.filename.endswith('.shp'):
                return member.filename


def geoextract_shp(src_shp, area, vfs=None):
    """Extract features from an input shapefile
    that intersect a given geometry.
    """
    features = []
    with fiona.open(src_shp, vfs=vfs) as shp:
        for feature in shp:
            geom = shape(feature['geometry'])
            if geom.intersects(area):
                features.append(feature)
    return features


def reproject_features(features, src_crs, dst_crs):
    """Reproject GeoJSON-like features."""
    dst_features = []
    for feature in features:
        dst_feature = feature.copy()
        dst_feature['geometry'] = transform_geom(
            src_crs, dst_crs, feature['geometry'])
        dst_features.append(dst_feature)
    return dst_features


def filter_features(features, geom_type='Polygon'):
    """Filter input GeoJSON-like features to a given geometry type."""
    return [f for f in features if f['geometry']['type'] == geom_type]


def create_ocean_table(connection):
    """Create 'ocean' table in the database."""
    cur = connection.cursor()
    sql = """CREATE TABLE ocean (
               id SERIAL PRIMARY KEY,
               geom geometry(POLYGON, 3857)
             );
          """
    cur.execute(sql)
    connection.commit()
    return True


def _ocean_exists(connection):
    """Check if the 'ocean' table already exists in the database."""
    cur = connection.cursor()
    sql = "SELECT tablename FROM pg_tables WHERE tablename = 'ocean';"
    cur.execute(sql)
    response = cur.fetchone()
    if not response:
        return False
    return 'ocean' in response


def import_features(features, connection):
    """Import GeoJSON-like features into the database."""
    geoms = [f['geometry'] for f in features]
    cur = connection.cursor()
    for geom in geoms:
        geom_wkt = shape(geom).wkt
        sql = """INSERT INTO ocean (geom) VALUES (
                   ST_GeomFromText(%s, 3857)
                 );
              """
        cur.execute(sql, (geom_wkt, ))
    connection.commit()
    return True


def import_oceans(connection):
    """Import oceans & seas polygons from openstreetmapdata.com
    into the OSM database.
    """
    if _ocean_exists(connection):
        return True
    create_ocean_table(connection)
    xmin, ymin, xmax, ymax = -21.445313, -36.031332, 52.558594, 36.173357
    africa = Polygon(((xmin, ymax),
                      (xmax, ymax),
                      (xmax, ymin),
                      (xmin, ymin),
                      (xmin, ymax)))
    src_crs = CRS(init='epsg:4326')
    dst_crs = CRS(init='epsg:3857')
    with TemporaryDirectory() as tmp_dir:
        print('downloading shp...')
        archive = download_file(SHAPEFILE_URL, tmp_dir)
        vfs = 'zip://{}'.format(archive)
        shp_path = '/' + _find_shp_path(archive)
        features = geoextract_shp(shp_path, africa, vfs=vfs)
        features = reproject_features(features, src_crs, dst_crs)
        features = filter_features(features, geom_type='Polygon')
        print(features[:10])
        print('importing oceans...')
        import_features(features, connection)
        connection.close()
    return True
