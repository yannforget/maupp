"""Pre-processing of Landsat scenes."""

import argparse
import os
from itertools import chain
import shutil
from tempfile import mkdtemp

import numpy as np
import rasterio
from rasterio import mask, warp
from scipy.ndimage import binary_dilation
from shapely.geometry import mapping
from shapely import wkt
from sklearn.linear_model import LinearRegression

from pylandsat import Scene, preprocessing
from maupp.bqa import fill_mask
from maupp.utils import reproject_geom


def preprocess(product, aoi, crs, output_dir, secondary_product=None):
    """Preprocess Landsat data.

    Parameters
    ----------
    product : str
        Path to primary landsat scene.
    aoi : shapely geometry
        Area of interest (in target CRS).
    crs : rasterio.crs.CRS
        Target CRS.
    output_dir : str
        Output directory.
    secondary_product : str, optional
        Path to secondary landsat scene for mosaicking.
    """
    tmp_dir = mkdtemp(prefix='maupp_')

    primary = Scene(product)
    if primary.blue.crs == crs:
        primary = crop(primary, aoi, os.path.join(
            tmp_dir, primary.product_id + '_crop'))
    else:
        aoi_reproj = reproject_geom(aoi, crs, primary.blue.crs).envelope
        primary = crop(
            primary, aoi_reproj, os.path.join(tmp_dir, primary.product_id + '_crop'))
        primary = reproject(
            primary, crs, os.path.join(tmp_dir, primary.product_id + '_reproj'))
        primary = crop(
            primary, aoi, os.path.join(tmp_dir, primary.product_id + '_crop2'))
    primary = calibrate(primary, os.path.join(tmp_dir, primary.product_id))
    
    # If there is a secondary product, co-register the scenes
    # and perform mosaicking.
    if secondary_product:
        secondary = Scene(secondary_product)
        secondary = coregister(
            primary, secondary, os.path.join(tmp_dir, secondary.product_id + '_reproj'))
        secondary = calibrate(secondary, os.path.join(tmp_dir, secondary.product_id))
        merge(primary, secondary, output_dir)
    # Else, just write data from primary scene to disk.
    else:
        # Spectral bands
        for band in primary:
            fpath = os.path.join(output_dir, band.bname + '.tif')
            with rasterio.open(fpath, 'w', **band.profile) as dst:
                dst.write(band.read(1), 1)
        # Quality band
        fpath = os.path.join(output_dir, 'quality.tif')
        with rasterio.open(fpath, 'w', **primary.quality.profile) as dst:
            dst.write(primary.quality.read(1), 1)
        # Metadata
        src_mtl = primary.file_path('MTL')
        dst_mtl = os.path.join(output_dir, 'primary_MTL.txt')
        shutil.copy(src_mtl, dst_mtl)
    
    return


def calibrate(scene, output_dir):
    """Convert DN values to surface reflectance or brightness temperature.

    Parameters
    ----------
    scene : pylandsat.Scene
        Input Landsat scene.
    output_dir : str
        Output directory.

    Returns
    -------
    out_scene : pylandsat.Scene
        Output Landsat scene.
    """
    os.makedirs(output_dir, exist_ok=True)

    # iterate over spectral bands + the quality band
    for band in chain(scene, (scene.quality, )):

        src_array = band.read(1)

        # convert thermal bands to brightness temperature
        if 'tir' in band.bname:
            gain, bias = band._gain_bias(unit='radiance')
            radiance = preprocessing.to_radiance(src_array, gain, bias)
            k1, k2 = band._k1_k2()
            dst_array = preprocessing.to_brightness_temperature(
                radiance, k1, k2)

        # quality band
        elif band.bname == 'bqa':
            dst_array = src_array

        # convert spectral bands to reflectance
        else:
            gain, bias = band._gain_bias(unit='reflectance')
            dst_array = preprocessing.to_reflectance(src_array, gain, bias)

        # write to disk
        profile = band.profile.copy()
        profile.update(dtype=dst_array.dtype.name, compress='lzw')
        filepath = os.path.join(output_dir, band.fname)
        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(dst_array, 1)

    # copy metadata
    src_mtl = scene.file_path('MTL')
    dst_mtl = os.path.join(output_dir, os.path.basename(src_mtl))
    shutil.copy(src_mtl, dst_mtl)

    return Scene(output_dir)


def reproject(scene, dst_crs, output_dir, **kwargs):
    """Reproject a Landsat scene.

    Parameters
    ----------
    scene : pylandsat.Scene
        Input Landsat scene.
    dst_crs : dict
        Target CRS.
    output_dir : str
        Output directory.
    **kwargs
        Passed to rasterio.warp.reproject().

    Returns
    -------
    out_scene : pylandsat.Scene
        Output Landsat scene.
    """
    os.makedirs(output_dir, exist_ok=True)

    for band in chain(scene, (scene.quality, )):
        
        if band.bname == 'bqa':
            resampling = warp.Resampling.nearest
        else:
            resampling = warp.Resampling.cubic

        src_array = band.read(1)
        profile = band.profile.copy()

        if band.crs == dst_crs:
            dst_array = src_array

        else:
            dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                band.crs, dst_crs, src_array.shape[1], src_array.shape[0], *band.bounds)
            dst_array = np.empty((dst_height, dst_width),
                                 dtype=src_array.dtype)
            warp.reproject(
                source=src_array,
                destination=dst_array,
                src_crs=band.crs,
                src_transform=band.transform,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                resampling=resampling,
                **kwargs)
            profile.update(
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                dtype=dst_array.dtype.name)

        filepath = os.path.join(output_dir, band.fname)
        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(dst_array, 1)
        
    src_mtl = scene.file_path('MTL')
    dst_mtl = os.path.join(output_dir, os.path.basename(src_mtl))
    shutil.copy(src_mtl, dst_mtl)

    return Scene(output_dir)


def coregister(scene_a, scene_b, output_dir, **kwargs):
    """Coregister scene_b in same CRS and extent of scene_a.

    Parameters
    ----------
    scene_a : pylandsat.Scene
        Primary landsat scene (target CRS and extent).
    scene_b : pylandsat.Scene
        Secondary landsat scene (scene to reproject).
    output_dir : str
        Output directory.
    
    Returns
    -------
    out_scene : pylandsat.Scene
        Output landsat scene.
    """
    os.makedirs(output_dir, exist_ok=True)

    for band_a, band_b in zip(
            chain(scene_a, (scene_a.quality, )),
            chain(scene_b, (scene_b.quality, ))):
        
        if band_a.bname == 'bqa':
            resampling = warp.Resampling.nearest
        else:
            resampling = warp.Resampling.cubic
        
        dst_array = np.empty_like(band_a.read(1))
        warp.reproject(
            source=band_b.read(1),
            destination=dst_array,
            src_crs=band_b.crs,
            src_transform=band_b.transform,
            dst_crs=band_a.crs,
            dst_transform=band_a.transform,
            resampling=resampling,
            **kwargs
        )

        filepath = os.path.join(output_dir, band_b.fname)
        with rasterio.open(filepath, 'w', **band_a.profile) as dst:
            dst.write(dst_array, 1)

    src_mtl = scene_b.file_path('MTL')
    dst_mtl = os.path.join(output_dir, os.path.basename(src_mtl))
    shutil.copy(src_mtl, dst_mtl)

    return Scene(output_dir)


def crop(scene, geom, output_dir, **kwargs):
    """Crop a Landsat scene according to a given geometry.

    Parameters
    ----------
    scene : pylandsat.Scene
        Input Landsat scene.
    geom : shapely geometry
        Area of interest (same CRS).
    output_dir : str
        Output directory.
    **kwargs
        Passed to rasterio.mask.mask().

    Returns
    -------
    out_scene : pylandsat.Scene
        Output Landsat scene.
    """
    os.makedirs(output_dir, exist_ok=True)

    for band in chain(scene, (scene.quality, )):

        dst_array, dst_transform = mask.mask(
            band, [mapping(geom)], crop=True, indexes=1, **kwargs)

        height, width = dst_array.shape
        profile = band.profile.copy()
        profile.update(transform=dst_transform, width=width, height=height)
        filepath = os.path.join(output_dir, band.fname)
        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(dst_array, 1)

    src_mtl = scene.file_path('MTL')
    dst_mtl = os.path.join(output_dir, os.path.basename(src_mtl))
    shutil.copy(src_mtl, dst_mtl)

    return Scene(output_dir)


def merge(primary, secondary, output_dir):
    """Merge two Landsat scenes into a single mosaic.

    Parameters
    ----------
    primary : pylandsat.Scene
        Primary input landsat scene.
    secondary : pylandsat.Scene
        Secondary input landsat scene.
    output_dir : str
        Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # get the mask corresponding to valid data in both the primary and
    # the secondary scene in order to perform relative normalization
    # of the secondary scene.
    fill_a = fill_mask(primary.quality.read(1), primary.sensor)
    fill_b = fill_mask(secondary.quality.read(1), secondary.sensor)
    fill_a = binary_dilation(fill_a, iterations=3)
    fill_b = binary_dilation(fill_b, iterations=3)
    fill = ~(fill_a | fill_b)

    for band_a in primary:

        # get slope and intercept to perform relative normalization
        # of secondary scene.
        band_b = getattr(secondary, band_a.bname)
        X = band_b.read(1)[fill].ravel().reshape(-1, 1)
        y = band_a.read(1)[fill].ravel().reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X, y)
        slope, intercept = reg.coef_[0][0], reg.intercept_[0]
        band_b_normalized = slope * band_b.read(1) + intercept

        # fill nodata values in primary scene with values from
        # normalized secondary scene.
        mosaic = band_a.read(1).copy()
        mosaic[fill_a] = band_b_normalized[fill_a]

        filepath = os.path.join(output_dir, band_a.bname + '.tif')
        with rasterio.open(filepath, 'w', **band_a.profile) as dst:
            dst.write(mosaic, 1)

    # mosaic primary and secondary quality bands
    mosaic = primary.quality.read(1)
    mosaic[fill_a] = secondary.quality.read(1)[fill_a]
    filepath = os.path.join(output_dir, 'quality.tif')
    with rasterio.open(filepath, 'w', **primary.quality.profile) as dst:
        dst.write(mosaic, 1)

    # copy metadata of both primary and secondary scenes
    for scene, label in zip((primary, secondary), ('primary', 'secondary')):
        src_mtl = scene.file_path('MTL')
        dst_mtl = os.path.join(output_dir, label + '_MTL.txt')
        shutil.copy(src_mtl, dst_mtl)

    # also keep track of the origin of each pixel
    # 1 = primary scene, 2 = secondary scene
    origin = np.zeros(shape=fill.shape, dtype=np.uint8)
    origin[fill_a] = 2
    origin[origin != 2] = 1
    profile = primary.blue.profile.copy()
    profile.update(dtype=origin.dtype.name)
    filepath = os.path.join(output_dir, 'origin.tif')
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(origin, 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess landsat data.')
    parser.add_argument('--aoi', type=str, required=True,
                        help='area of interest (wkt)')
    parser.add_argument('--epsg', type=int, required=True,
                        help='target epsg code')
    parser.add_argument('--outdir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--aux', type=str,
                        help='auxiliary product')
    parser.add_argument('source', type=str,
                        help='source product')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    crs = rasterio.crs.CRS(init='epsg:{}'.format(args.epsg))
    aoi = wkt.loads(args.aoi)

    preprocess(args.source, aoi, crs, args.outdir, args.aux)
