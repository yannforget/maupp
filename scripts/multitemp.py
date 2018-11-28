"""Multi-temporal post-processing of multi-year built-up data,
validation of the post-processed products and writing of the
final product.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import sys
from datetime import datetime
from math import floor
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_geom
from scipy.ndimage import binary_closing, uniform_filter
from shapely.geometry import Polygon, shape
from skimage.morphology import label
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)

from config import DATA_DIR, DB_NAME, DB_USER, DB_PASS, DB_HOST, YEARS
from maupp import CaseStudy, osm, reference, srtm

WGS84 = CRS(init='EPSG:4326')


def available(case_study):
    """List available results for a given case study."""
    years_available = []
    filenames = []
    for year in YEARS:
        path = os.path.join(case_study.outputdir,
                            str(year), 'probabilities.tif')
        if os.path.isfile(path):
            years_available.append(year)
            filenames.append(path)
    return filenames, years_available


def main_grid(case_study):
    """Get main raster profile (crs, transform...) from 2015 result."""
    probas2015_path = os.path.join(
        case_study.outputdir, '2015', 'probabilities.tif')
    with rasterio.open(probas2015_path) as src:
        src_bounds = src.bounds
        src_crs, src_transform = src.crs, src.transform
        src_profile = src.profile
    left, bottom, right, top = src_bounds
    dst_width = floor((right - left) / 30)
    dst_height = floor((top - bottom) / 30)
    dst_transform = from_origin(left, top, 30, 30)
    dst_profile = src_profile.copy()
    dst_profile.update(
        width=dst_width,
        height=dst_height,
        crs=src_crs,
        transform=dst_transform,
        compress='LZW',
        count=1)
    return dst_profile


def multitemp_stack(filenames, dtype='float32'):
    """Load multi-temporal stack of built-up data as an array
    of shape (height, width, n_years).
    """
    with rasterio.open(filenames[0]) as src:
        width, height = src.width, src.height
    builtup = np.zeros(shape=(height, width, len(filenames)), dtype=dtype)
    for i, fname in enumerate(filenames):
        with rasterio.open(fname) as src:
            builtup[:, :, i] = src.read(1)
    return builtup


def water_mask(case_study, crs, transform, width, height):
    """Water mask from OSM database (water bodies + seas & oceans)."""
    db = osm.OSMDatabase(DB_NAME, DB_USER, DB_PASS, DB_HOST)
    polygons = db.water(case_study.aoi_latlon)
    geoms = [feature['geometry'] for feature in polygons]
    geoms = [transform_geom(WGS84, crs, geom) for geom in geoms]
    if len(geoms) == 0:
        return np.zeros(shape=(height, width), dtype=np.bool)
    water = rasterize(
        shapes=geoms,
        out_shape=(height, width),
        transform=transform,
        dtype=np.uint8).astype(np.bool)
    db.connection.close()
    return water


def slope_mask(case_study, crs, transform, width, height, max_slope=40):
    """High slopes mask from 30m SRTM DEM."""
    dem_path = os.path.join(case_study.inputdir, 'dem', 'elevation.tif')
    with TemporaryDirectory(prefix='maupp_') as tmp_dir:
        dem_reproj_path = os.path.join(tmp_dir, 'elevation.tif')
        dem_reproj_path = srtm.coregister(
            dem_path, dem_reproj_path, crs, transform, width, height)
        slope = srtm.compute_slope(dem_reproj_path)
    return slope >= max_slope


def post_processing(builtup, years):
    """Post-process multitemporal stack based on logical assumptions
    in a context of urban expansion.
    """
    n = len(years)

    # Not built-up in future
    for i in range(n - 1):
        y = years[i]
        y_aux = years[i+1:][:2]
        if len(y_aux) == 2:
            aux_a, aux_b = map(years.index, y_aux)
            nb_in_future = ~builtup[:, :, aux_a] & ~builtup[:, :, aux_b]
            builtup[:, :, i][nb_in_future] = 0

    # Built-up in past
    for i in range(n):
        years_r = list(reversed(years))
        y = years[i]
        y_aux = years_r[n-i:][:2]
        if len(y_aux) == 2:
            aux_a, aux_b = map(years.index, y_aux)
            bu_in_past = builtup[:, :, aux_a] & builtup[:, :, aux_b]
            builtup[:, :, i][bu_in_past] = 1

    return builtup


def validate(case_study, data_dir, years, crs, transform, width, height):
    """Validate final post-processed products."""
    meta = {}
    reference_dir = os.path.join(case_study.inputdir, 'reference')
    for year in years:
        scores = {}
        if reference.available(reference_dir, year, 2):
            validation_samples = reference.reference_dataset(
                reference_dir, year, 2, crs, transform, width, height)
            y_true = validation_samples[validation_samples > 0].ravel()
            y_true = y_true == 1
            probas_path = os.path.join(
                data_dir, 'probabilities_{}.tif'.format(year))
            with rasterio.open(probas_path) as src:
                probas = src.read(1)
            y_pred = probas[validation_samples > 0].ravel()
            scores['roc_auc'] = roc_auc_score(y_true, y_pred)
            with rasterio.open(probas_path.replace('probabilities', 'builtup')) as src:
                builtup = src.read(1)
            y_pred = builtup[validation_samples > 0].ravel() == 1
            scores['f1_score'] = f1_score(y_true, y_pred)
            scores['precision'] = precision_score(y_true, y_pred)
            scores['recall'] = recall_score(y_true, y_pred)
        with open(os.path.join(case_study.outputdir, str(year), 'metrics.json')) as f:
            metrics = json.load(f)
            scores['cv_mean'] = metrics['cv_mean']
            scores['cv_std'] = metrics['cv_std']
        meta[year] = scores
    return meta


def urban_patches(built_up, transform):
    """Identify urban patches."""
    built_up = binary_closing(built_up, iterations=int(
        100 / transform.a), border_value=1)
    patch_geoms = []
    for geom, value in shapes(built_up.astype('uint8'), connectivity=4, transform=transform):
        if value == 1:
            geom = shape(geom)
            geom = Polygon(geom.exterior)
            patch_geoms.append(geom.__geo_interface__)
    patches = rasterize(
        shapes=patch_geoms,
        out_shape=built_up.shape,
        transform=transform,
        all_touched=True,
        dtype='uint16')
    return label(patches, neighbors=8)


def characterize(builtup_t1, builtup_t2, transform):
    """Characterize type of built-up expansion.
    1=Existing, 2=Inclusion, 3=Expansion, 4=Leapfrog.
    """
    patches_t1 = urban_patches(builtup_t1, transform)
    patches_t2 = urban_patches(builtup_t2, transform)

    expansion_labels = []
    for i in np.unique(patches_t2)[1:]:
        if np.count_nonzero((patches_t2 == i) & (patches_t1 > 0)) > 0:
            expansion_labels.append(i)

    bu_expansion = np.zeros(shape=builtup_t1.shape, dtype='uint8')
    bu_expansion[builtup_t1 == 1] = 1
    bu_expansion[(builtup_t1 == 0) & (builtup_t2 == 1) & (patches_t1 > 0)] = 2
    bu_expansion[(bu_expansion == 0) & (builtup_t2 == 1) &
                 np.isin(patches_t2, expansion_labels)] = 3
    bu_expansion[(bu_expansion == 0) & (builtup_t2 == 1)] = 4
    return bu_expansion


def _periods(years):
    """Get a list of periods for a given set of years, i.e.
    couples of start & stop years.
    """
    periods = []
    n = len(years)
    for i in range(n - 1):
        periods.append((years[i], years[i+1]))
    if len(years) > 2:
        periods.append((years[0], years[-1]))
    if len(years) > 3:
        periods.append((years[1], years[-1]))
    return periods


def _metadata(case_study):
    """Collect metadata in a dictionnary."""
    return {
        'creation_date': datetime.strftime(datetime.now(), '%c'),
        'imagery': case_study.imagery,
        'latitude': case_study.lat,
        'longitude': case_study.lon,
        'area_of_interest': case_study.aoi_latlon.wkt,
        'crs': str(case_study.crs),
        'country': case_study.country
    }


def multitemp_analysis(case_study):
    """Multi-temporal post processing of multi-year built-up data."""
    output_dir = os.path.join(DATA_DIR, 'final', case_study.id)
    os.makedirs(output_dir, exist_ok=True)
    filenames, years = available(case_study)
    profile = main_grid(case_study)
    crs, transform = profile['crs'], profile['transform']
    width, height = profile['width'], profile['height']
    profile.update(dtype='float32', count=1, compress='LZW')

    water = water_mask(case_study, crs, transform, width, height)
    slope = slope_mask(case_study, crs, transform, width, height, max_slope=40)

    with TemporaryDirectory(prefix='maupp_') as tmp_dir:

        dst_filenames = []
        for fname, year in zip(filenames, years):

            output_path = os.path.join(
                tmp_dir, 'probabilities_{}.tif'.format(year))
            dst_filenames.append(output_path)
            dst_array = np.zeros(shape=(height, width), dtype='float32')
            with rasterio.open(fname) as src:
                reproject(
                    src.read(1), dst_array,
                    src_transform=src.transform,
                    dst_transform=transform,
                    src_crs=src.crs,
                    dst_crs=crs,
                    resampling=Resampling.average)

            dst_array = uniform_filter(dst_array, size=3)
            dst_array[water] = 0
            dst_array[slope] = 0

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(dst_array, 1)

        probas = multitemp_stack(dst_filenames)
        builtup = probas >= 0.8
        builtup = post_processing(builtup, years)
        builtup = builtup.astype('uint8')

        for i, year in enumerate(years):
            output_path = os.path.join(tmp_dir, 'builtup_{}.tif'.format(year))
            profile.update(dtype='uint8')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(builtup[:, :, i], 1)

        meta = validate(case_study, tmp_dir, years, crs,
                        transform, width, height)
        with open(os.path.join(tmp_dir, 'validation.json'), 'w') as f:
            json.dump(meta, f, indent=True)

        filenames = [os.path.join(
            tmp_dir, 'builtup_{}.tif'.format(y)) for y in years]
        builtup = multitemp_stack(filenames, dtype='bool')
        for y1, y2 in _periods(years):
            t1, t2 = map(years.index, (y1, y2))
            expansion = characterize(
                builtup[:, :, t1], builtup[:, :, t2], transform)
            profile.update(dtype='uint8')
            output_path = os.path.join(
                tmp_dir, 'expansion_{}_{}.tif'.format(y1, y2))
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(expansion, 1)

        metadata_path = os.path.join(tmp_dir, 'metadata.json')
        metadata = _metadata(case_study)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=True)

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(tmp_dir, output_dir)

        print(case_study.name + ' : done.')


if __name__ == '__main__':

    case_studies = pd.read_csv(
        os.path.join(DATA_DIR, 'input', 'case-studies.csv'), index_col=0)
    DROPPED = ['Rufisque', 'Shaki', 'Owo', 'Dire Dawa']
    case_studies.drop(labels=DROPPED, inplace=True, errors='ignore')

    cities = []
    if len(sys.argv) > 1:
        name = sys.argv[1]
        lat, lon, country = case_studies.loc[name][['latitude', 'longitude', 'country']]
        case_study = CaseStudy(name, lat, lon, DATA_DIR, country)
        cities.append(case_study)
    else:
        for name, row in case_studies.iterrows():
            case_study = CaseStudy(
                name, row.latitude, row.longitude, DATA_DIR, row.country)
            cities.append(case_study)
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(multitemp_analysis, cities)
