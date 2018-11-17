"""Access data related to case studies."""

import os
import json

from rasterio.crs import CRS
from shapely.geometry import Point

from maupp.utils import reproject_geom, find_utm_epsg


WGS84 = CRS.from_epsg(4326)


class CaseStudy:
    """Abstract class for a case study."""

    def __init__(self, name, lat, lon, datadir, country=None, buffer_size=20000):
        """MAUPP case study."""
        self.name = name
        self.country = country
        self.lat = lat
        self.lon = lon
        self.datadir = datadir
        self.buffer_size = buffer_size

    @property
    def id(self):
        """Identifier."""
        return self.name.lower().replace(' ', '_').replace('-', '_')

    @property
    def center(self):
        """Location as a shapely Point geometry."""
        return Point(self.lon, self.lat)

    @property
    def epsg(self):
        """EPSG code."""
        return self.crs['init'].split(':')[1]

    @property
    def crs(self):
        """CRS as a rasterio.crs.CRS object."""
        epsg_code = find_utm_epsg(self.lat, self.lon)
        return CRS.from_epsg(epsg_code)

    @property
    def aoi(self):
        """Area of interest as a shapely geometry."""
        src_crs = WGS84
        center = reproject_geom(self.center, src_crs, self.crs)
        buffer = center.buffer(self.buffer_size)
        aoi_geom = buffer.exterior.envelope
        return aoi_geom

    @property
    def aoi_latlon(self):
        """Area of interest in lat/lon coordinates."""
        return reproject_geom(self.aoi, src_crs=self.crs, dst_crs=WGS84)

    @property
    def inputdir(self):
        """Input data directory."""
        return os.path.join(self.datadir, 'input', self.id)

    @property
    def outputdir(self):
        """Output data directory."""
        return os.path.join(self.datadir, 'output', self.id)

    @property
    def imagery(self):
        """Imagery selection."""
        fpath = os.path.join(self.inputdir, 'imagery-selection.json')
        if os.path.isfile(fpath):
            with open(fpath) as f:
                products = json.load(f)
            return {int(key): value for key, value in products.items()}
        else:
            return None

    @property
    def reference(self):
        """Available reference data per year."""
        ref_data = {}
        ref_dir = os.path.join(self.inputdir, 'reference')
        if not os.path.isdir(ref_dir):
            return None
        basenames = [f.split('.')[0] for f in os.listdir(ref_dir)]
        years = tuple(set([f.split('_')[-1] for f in basenames]))
        for year in years:
            ref_data[year] = [f.split('_')[0] for f in basenames
                              if f.split('_')[-1] == str(year)]
        return ref_data
