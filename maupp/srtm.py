"""Acquisition and preprocessing of the SRTM 30m DEM."""

import os
from tempfile import TemporaryDirectory

import elevation
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.warp import Resampling, reproject


def download(bounds, output_path):
    """Download DEM for a given set of bounds.

    Parameters
    ----------
    bounds : tuple of float
        Bounds in lat/lon coordinates.
    output_path : str
        Path to output file.

    Returns
    -------
    output_path : str
        Path to output file.
    """
    if not os.path.isfile(output_path):
        elevation.clip(bounds, output_path, margin='5%')
    return output_path


def coregister(input_path, output_path, crs, transform, width, height):
    """Co-registration of DEM.

    Parameters
    ----------
    input_path : str
        Path to input DEM geotiff.
    output_path : str
        Path to output DEM geotiff.
    crs : CRS
        Target CRS.
    transform : Affine
        Target affine transform.
    width : int
        Target raster width.
    height : int
        Target raster height.

    Returns
    -------
    output_path : str
        Path to output DEM geotiff.
    """
    with rasterio.open(input_path) as src:
        src_array = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_profile = src.profile

    dst_array = np.zeros(shape=(height, width), dtype='int16')
    reproject(src_array, dst_array, src_transform=src_transform,
              dst_transform=transform, src_crs=crs,
              resampling=Resampling.bilinear)

    dst_profile = src_profile.copy()
    dst_profile.update(transform=transform, crs=crs, width=width,
                       height=height, dtype='int16')
    with rasterio.open(output_path, 'w', **dst_profile) as dst:
        dst.write(dst_array, 1)
    return output_path


def compute_slope(dem_path):
    """Compute slope in percents from a DEM with gdal.

    Parameters
    ----------
    dem_path : str
        Path to input DEM geotiff.

    Returns
    -------
    slope : 2d array
        Slope raster as a 2d numpy array.
    """
    with TemporaryDirectory(prefix='maupp_') as tmp_dir:
        tmp_file = os.path.join(tmp_dir, 'slope.tif')
        gdal.DEMProcessing(tmp_file, dem_path, 'slope', slopeFormat='percent')
        with rasterio.open(tmp_file) as src:
            slope = src.read(1)
    slope[slope < 0] = 0
    slope[slope > 100] = 100
    return slope
