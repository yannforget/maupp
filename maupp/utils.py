from functools import partial
from math import floor
import os
import shutil
import zipfile

import pyproj
from rasterio.warp import transform_geom
import requests
from shapely.ops import transform


def reproject_geom(geom, src_crs, dst_crs):
    """Reproject a shapely geometry.

    Parameters
    ----------
    geom : shapely geometry
        Input shapely geometry.
    src_crs : CRS
        Source CRS.
    dst_crs : CRS
        Target CRS.

    Returns
    -------
    out_geom : shapely geometry
        Reprojected shapely geometry.
    """
    src_proj = pyproj.Proj(**src_crs)
    dst_proj = pyproj.Proj(**dst_crs)
    reproj = partial(pyproj.transform, src_proj, dst_proj)
    return transform(reproj, geom)


def reproject_features(features, src_crs, dst_crs):
    """Reproject a list of GeoJSON-like features.

    Parameters
    ----------
    features : iterable of dict
        An iterable of GeoJSON-like features.
    src_crs : CRS
        Source CRS.
    dst_crs : CRS
        Target CRS.

    Returns
    -------
    out_features : iterable of dict
        Iterable of reprojected GeoJSON-like features.
    """
    out_features = []
    for feature in features:
        out_feature = feature.copy()
        out_geom = transform_geom(src_crs, dst_crs, feature['geometry'])
        out_feature['geometry'] = out_geom
        out_features.append(out_feature)
    return out_features


def filter_features(features, key, include=None, exclude=None):
    """Filter an iterable of GeoJSON-like features based on
    one of their attributes.

    Parameters
    ----------
    features : iterable of dict
        Source GeoJSON-like features.
    key : str
        Property key.
    include : tuple of str, optional
        Property values to include.
    exclude : tuple of str, optional
        Property values to exclude.

    Returns
    -------
    out_features : list of dict
        Filtered GeoJSON-like features.
    """
    if not include and not exclude:
        raise ValueError('No value to include or exclude.')
    out_features = []
    for feature in features:
        if include:
            if feature['properties'][key] in include:
                out_features.append(feature)
        if exclude:
            if not feature['properties'][key] in exclude:
                out_features.append(feature)
    return out_features


def iter_geoms(features):
    """Iterate over geometries of an iterable of GeoJSON-like features."""
    return (feature['geometry'] for feature in features)


def find_utm_epsg(lat, lon):
    """Find the UTM EPSG code based on lat/lon coordinates."""
    utm_zone = (floor((lon + 180) // 6) % 60) + 1
    if lat >= 0:
        pole = 600
    else:
        pole = 700
    return 32000 + pole + utm_zone


def to_crs(epsg):
    """CRS dict from EPSG code."""
    return {'init': 'epsg:{}'.format(epsg)}


def unzip(filepath, remove_archive=True):
    """Decompress a .zip archive."""
    dst_dir = os.path.dirname(os.path.abspath(filepath))
    with open(filepath, 'rb') as src:
        archive = zipfile.ZipFile(src)
        for member in archive.infolist():
            dst_path = os.path.join(dst_dir, member.filename)
            if dst_path.endswith('/'):
                os.makedirs(dst_path)
                continue
            with open(dst_path, 'wb') as dst, archive.open(member) as content:
                shutil.copyfileobj(content, dst)
    if remove_archive:
        os.remove(filepath)


def download_file(url, output_dir):
    """Download file from url."""
    fname = url.split('/')[-1]
    dst_file = os.path.join(output_dir, fname)
    with requests.get(url, stream=True) as r:
        with open(dst_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024**2):
                if chunk:
                    f.write(chunk)
    return dst_file
