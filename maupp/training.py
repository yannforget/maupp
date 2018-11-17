"""Extract training samples from OSM."""

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import mapping

from maupp.osm import urban_blocks
from maupp.utils import reproject_features, iter_geoms, filter_features


WGS84 = CRS(init='epsg:4326')


def buildings(osm, aoi, crs, transform, width, height, min_coverage=0.2):
    """Extract and rasterize building footprints from OSM.

    Parameters
    ----------
    osm : maupp.osm.OSMDatabase
        An initialized OSMDatabase object.
    aoi : shapely polygon
        Area of interest in lat/lon coordinates.
    crs : CRS
        Target CRS (rasterio.crs.CRS object).
    transform : Affine
        Target affine transform.
    width : int
        Target width.
    height : int
        Target height.
    min_coverage : float, optional
        Min. surface of a pixel that footprints must cover.

    Returns
    -------
    out_img : numpy 2d array
        Rasterized building footprints.
    """
    # Rasterize building footprints into a binary raster with
    # 4x smaller pixel size
    SCALE = 0.25
    tmp_transform, tmp_width, tmp_height = rescale_transform(
        transform, width, height, SCALE)
    features = osm.buildings(aoi)
    features = reproject_features(features, src_crs=WGS84, dst_crs=crs)
    footprints = rasterize(
        shapes=iter_geoms(features),
        out_shape=(tmp_height, tmp_width),
        transform=tmp_transform,
        all_touched=False,
        dtype='uint8')

    # Resample binary raster to a continuous raster with 4x higher
    # pixel size by averaging the values of the binary raster.
    cover, _ = rescale(
        src_array=footprints,
        src_transform=tmp_transform,
        crs=crs,
        scale=1/SCALE)

    # Convert to binary raster
    return cover >= min_coverage


def blocks(osm, aoi, crs, transform, width, height, max_surface=30000):
    """Extract and rasterize urban blocks from the OSM road network.

    Parameters
    ----------
    osm : maupp.osm.OSMDatabase
        An initialized OSMDatabase object.
    aoi : shapely polygon
        Area of interest in lat/lon coordinates.
    crs : CRS
        Target CRS (rasterio.crs.CRS object).
    transform : Affine
        Target affine transform.
    width : int
        Target width.
    height : int
        Target height.
    max_surface : int, optional
        Max. surface of a block in meters.

    Returns
    -------
    out_img : numpy 2d array
        Rasterized urban blocks.
    """
    # Extract only roads of interest
    ROAD_TAGS = ('secondary', 'tertiary', 'residential')
    roads = gpd.GeoDataFrame.from_features(osm.roads(aoi), crs=WGS84)
    roads = roads[roads.highway.isin(ROAD_TAGS)]

    # Compute urban blocks from the road network and filter the
    # output polygons by their surface
    polygons = urban_blocks(roads.__geo_interface__['features'])
    polygons = gpd.GeoDataFrame.from_features(polygons, crs=WGS84)
    polygons = polygons.to_crs(crs)
    polygons = polygons[polygons.area <= max_surface]

    # Rasterize urban blocks
    return rasterize(
        shapes=(mapping(geom) for geom in polygons.geometry),
        out_shape=(height, width),
        transform=transform,
        dtype='uint8').astype(np.bool)


def nonbuilt(osm, aoi, crs, transform, width, height):
    """Extract and rasterize non-built `landuse`, `leisure` and `natural`
    features from OSM.

    Parameters
    ----------
    osm : maupp.osm.OSMDatabase
        An initialized OSMDatabase object.
    aoi : shapely polygon
        Area of interest in lat/lon coordinates.
    crs : CRS
        Target CRS (rasterio.crs.CRS object).
    transform : Affine
        Target affine transform.
    width : int
        Target width.
    height : int
        Target height.

    Returns
    -------
    out_img : numpy 2d array
        Rasterized non-built features.
    """
    NONBUILT_TAGS = ['sand', 'farmland', 'wetland', 'wood', 'park', 'forest',
                     'nature_reserve', 'golf_course', 'greenfield', 'quarry',
                     'pitch', 'scree', 'meadow', 'orchard', 'grass',
                     'grassland', 'garden', 'heath', 'bare_rock', 'beach']
    nonbuilt = osm.landuse(aoi) + osm.leisure(aoi) + osm.natural(aoi)
    nonbuilt = gpd.GeoDataFrame.from_features(nonbuilt, crs=WGS84)

    def tag(row):
        for key in ('landuse', 'leisure', 'natural'):
            if not pd.isna(row[key]):
                return row[key]
        return None

    nonbuilt['tag'] = nonbuilt.apply(tag, axis=1)
    nonbuilt = nonbuilt[nonbuilt.tag.isin(NONBUILT_TAGS)]
    nonbuilt = nonbuilt.to_crs(crs)
    return rasterize(
        shapes=(mapping(geom) for geom in nonbuilt.geometry),
        out_shape=(height, width),
        transform=transform,
        dtype='uint8').astype(np.bool)


def remote(osm, aoi, crs, transform, width, height, min_distance=250):
    """Identify remote areas (i.e. distant from any building or road).
    Roads identified as paths or tracks are ignored.

    Parameters
    ----------
    osm : maupp.osm.OSMDatabase
        An initialized OSMDatabase object.
    aoi : shapely polygon
        Area of interest in lat/lon coordinates.
    crs : CRS
        Target CRS (rasterio.crs.CRS object).
    transform : Affine
        Target affine transform.
    width : int
        Target width.
    height : int
        Target height.
    min_distance : int, optional
        Min. distance from roads or buildings.

    Returns
    -------
    out_img : numpy 2d array
        Binary 2d array.
    """
    roads = osm.roads(aoi)
    roads = filter_features(roads, 'highway', exclude=['path', 'track'])
    roads = reproject_features(roads, WGS84, crs)
    roads = rasterize(
        shapes=iter_geoms(roads),
        out_shape=(height, width),
        transform=transform,
        all_touched=True,
        dtype='uint8').astype(np.bool)

    builtup = osm.buildings(aoi)
    builtup = reproject_features(builtup, WGS84, crs)
    builtup = rasterize(
        shapes=iter_geoms(builtup),
        out_shape=(height, width),
        transform=transform,
        all_touched=True,
        dtype='uint8').astype(np.bool)

    # Calculate distance of each pixel from roads or buildings
    # and multiply by pixel size to output values in meters.
    urban = roads | builtup
    distance = distance_transform_edt(np.logical_not(urban))
    return distance * transform.a >= min_distance


def rescale_transform(src_transform, src_width, src_height, scale):
    """Calculate the transform corresponding to a pixel size multiplied
    by a given scale factor.

    Parameters
    ----------
    src_transform : Affine
        Source affine transform.
    src_width : int
        Source raster width.
    src_height : int
        Source raster height.
    scale : float
        Scale factor (e.g. 0.5 to reduce pixel size by half).

    Returns
    -------
    dst_transform : Affine
        New affine transform.
    dst_width : int
        New raster width.
    dst_height : int
        New raster height.
    """
    dst_transform = Affine(
        src_transform.a * scale,
        src_transform.b,
        src_transform.c,
        src_transform.d,
        src_transform.e * scale,
        src_transform.f)
    dst_width = int(src_width / scale)
    dst_height = int(src_height / scale)
    return dst_transform, dst_width, dst_height


def rescale(src_array, src_transform, crs, scale, resampling_method='average'):
    """Rescale a raster by multiplying its pixel size by the `scale` value.

    Parameters
    ----------
    src_array : numpy 2d array
        Source raster.
    src_transform : rasterio.Affine
        Source affine transform.
    crs : rasterio.crs.CRS
        Source & destination CRS.
    scale : int
        Scale factor (e.g. 0.5 to reduce pixel size by half).
    resampling_method : str, optional
        Possible values are 'nearest', 'bilinear', 'cubic', 'cubic_spline',
        'lanczos', 'average', 'mode', 'gauss', 'max', 'min' and 'med'.

    Returns
    -------
    dst_array : numpy 2d array
        Output raster.
    dst_transform : rasterio.Affine
        Output affine transform.
    """
    resampling_method = getattr(Resampling, resampling_method)
    src_height, src_width = src_array.shape
    dst_transform, dst_width, dst_height = rescale_transform(
        src_transform, src_width, src_height, scale)
    dst_array = np.empty((dst_height, dst_width), 'float32')
    reproject(
        src_array, dst_array,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=resampling_method)
    return dst_array, dst_transform


def training_dataset(buildings, blocks, nonbuilt, remote, water):
    """Build a two-class training dataset from OSM features.
    Legend: 0=NA, 1=Built-up, 2=Non-built-up.

    Parameters
    ----------
    buildings : numpy 2d array
        Binary building footprints raster.
    blocks : numpy 2d array
        Binary urban blocks raster.
    nonbuilt : numpy 2d array
        Binary non-built-up raster.
    remote : numpy 2d array
        Binary remote areas raster.
    water : numpy 2d array
        Binary water mask.

    Returns
    -------
    training_dataset : numpy 2d array
        Output training data raster.
    """
    positive = buildings | blocks
    negative = nonbuilt | remote

    # Ignore pixels with multiple values or in water areas
    confusion = positive & negative
    mask = confusion | water
    positive[mask] = 0
    negative[mask] = 0

    training_samples = np.zeros(shape=positive.shape, dtype='uint8')
    training_samples[positive] = 1
    training_samples[negative] = 2

    return training_samples
