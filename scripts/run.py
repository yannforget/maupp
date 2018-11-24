"""Select, acquire and classify data for a given case study
and a given set of years.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import itertools
import os
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, transform_geom
from scipy.ndimage import uniform_filter
from sklearn.externals import joblib

from maupp import (CaseStudy, acquisition, features, osm, selection, srtm,
                   training, classification, reference, preprocessing)

from config import *

WGS84 = CRS(init='epsg:4326')


class MissingDataError(Exception):
    pass


def data_selection(case_study):
    """Selection of Optical & SAR imagery for a given case study.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.

    Returns
    -------
    imagery_selection : dict
        Sentinel-1, ERS, Envisat & Landsat products for each year.
    """
    if case_study.imagery:
        return case_study.imagery
    imagery_selection = {}
    for year in YEARS:
        products = selection.selection(
            case_study, year, DHUS_USERNAME, DHUS_PASSWORD, margin=365)
        imagery_selection[str(year)] = products
    output_f = os.path.join(case_study.inputdir, 'imagery-selection.json')
    with open(output_f, 'w') as f:
        json.dump(imagery_selection, f, indent=True)
    return imagery_selection


def data_acquisition(case_study):
    """Download satellite imagery for a given case study.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    """
    output_dir = os.path.join(case_study.inputdir, 'raw')
    os.makedirs(output_dir, exist_ok=True)
    products = []
    for year, source in itertools.product(YEARS, ('landsat', 'sar')):
        products += case_study.imagery[year][source]
    download = partial(
        acquisition.download,
        output_dir=output_dir,
        esa_sso_username=ESA_SSO_USERNAME,
        esa_sso_password=ESA_SSO_PASSWORD,
        dhus_username=DHUS_USERNAME,
        dhus_password=DHUS_PASSWORD)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(download, products)
    return output_dir


def update_osm_database(case_study):
    """Import OSM data into the PostGIS db for a given case study.
    Depends on `<DATA_DIR>/input/africa-latest.osm.pbf` and the
    osmium software.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    """
    db = osm.OSMDatabase(DB_NAME, DB_USER, DB_PASS, DB_HOST)
    main_datafile = os.path.join(
        DATA_DIR, 'input', 'africa-latest.osm.pbf')
    dst_datafile = os.path.join(
        case_study.inputdir, 'osm', case_study.id + '.osm.pbf')
    os.makedirs(os.path.dirname(dst_datafile), exist_ok=True)
    if not db.data_in_aoi(case_study.aoi_latlon.wkt):
        if os.path.isfile(dst_datafile):
            os.remove(dst_datafile)
        osm.geoextract_osmpbf(main_datafile, dst_datafile,
                              case_study.aoi_latlon.bounds)
        db.import_data(dst_datafile)
    db.connection.close()
    return True


def ocean_table():
    """Check that oceans and seas have been added to the OSM db."""
    db = osm.OSMDatabase(DB_NAME, DB_USER, DB_PASS, DB_HOST)
    osm.import_oceans(db.connection)
    return True


def imagery_preprocessing(case_study, year):
    """Pre-process SAR & Landsat imagery a given case study.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    year : int
        Year of interest.
    """
    input_dir = os.path.join(case_study.inputdir, 'raw')
    for source in ('sar', 'landsat'):
        products = case_study.imagery[year][source]
        if not products:
            continue
        out_dir = os.path.join(case_study.inputdir, source, str(year))
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= 1:
            continue
        os.makedirs(out_dir, exist_ok=True)
        main_product = acquisition.find_product(products[0], input_dir)
        if len(products) == 2:
            aux_product = acquisition.find_product(products[1], input_dir)
        else:
            aux_product = None
        if source == 'sar':
            preprocessing.sar.preprocess(
                product=main_product,
                georegion=case_study.aoi_latlon.wkt,
                epsg=case_study.crs.to_epsg(),
                out_dir=out_dir,
                auxiliary_product=aux_product)
        if source == 'landsat':
            preprocessing.landsat.preprocess(
                product=main_product,
                aoi=case_study.aoi,
                crs=case_study.crs,
                output_dir=out_dir,
                secondary_product=aux_product)
    return True


def water_mask(case_study, crs, transform, width, height):
    """Extract water mask from OSM database (water bodies + oceans)."""
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


def feature_extraction(case_study, year):
    """Compute co-registered set of SAR & Landsat features."""
    imagery = case_study.imagery[year]
    sar_is_available = bool(imagery['sar'])
    landsat_is_available = bool(imagery['landsat'])
    sar, landsat, water = None, None, None

    if sar_is_available:

        sar = features.SAR(
            os.path.join(case_study.inputdir, 'sar', str(year)))
        water = water_mask(
            case_study, sar.crs, sar.transform, sar.width, sar.height)

        if 'vv_simple_11x11_1' not in sar.available:
            sar.textures(
                water=water,
                radii=GLCM_RADII,
                offsets=GLCM_OFFSETS,
                kinds=GLCM_KINDS)
        if 'textures_pca' not in sar.available:
            sar.dim_reduction(n_components=GLCM_N_COMPONENTS)

    if landsat_is_available:

        landsat = features.Landsat(os.path.join(case_study.inputdir,
                                                'landsat', str(year)))

        if not isinstance(water, np.ndarray):
            water = water_mask(
                case_study, landsat.crs, landsat.transform,
                landsat.width, landsat.height)

        if sar_is_available:
            if landsat.transform != sar.transform:
                landsat.coregister(sar.path('vv'))
        if 'ndsv' not in landsat.available:
            landsat.compute_ndsv()

    return sar, landsat, water


def training_features(case_study, year, transform, width, height):
    """Extract and rasterize training data from the OSM database."""
    aoi = case_study.aoi_latlon
    crs = case_study.crs

    db = osm.OSMDatabase(DB_NAME, DB_USER, DB_PASS, DB_HOST)
    buildings = training.buildings(
        db, aoi, crs, transform, width, height, min_coverage=0.25)
    blocks = training.blocks(
        db, aoi, crs, transform, width, height, max_surface=30000)
    nonbuilt = training.nonbuilt(
        db, aoi, crs, transform, width, height)
    remote = training.remote(
        db, aoi, crs, transform, width, height, min_distance=250)

    return buildings, blocks, nonbuilt, remote


def training_labels(buildings, blocks, nonbuilt, remote, water):
    """Labelize training samples extracted from OSM."""
    positive = buildings | blocks
    negative = nonbuilt | remote
    mask = (positive & negative) | water
    training_samples = np.zeros(shape=positive.shape, dtype='uint8')
    training_samples[positive] = 1
    training_samples[negative] = 2
    training_samples[mask] = 0
    return training_samples


def compute_slope(case_study, sar, landsat, year):
    """Compute slope in percent. Download DEM if necessary.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    sar : maupp.features.SAR
        SAR features.
    landsat : maupp.features.SAR
        Landsat features.
    year : int
        Year of interest.

    Returns
    -------
    slope : 2d array
        Slope in percents as a 2d numpy array.
    """
    dem_dir = os.path.join(case_study.inputdir, 'dem')
    os.makedirs(dem_dir, exist_ok=True)

    dem_path = os.path.join(dem_dir, 'elevation.tif')
    dem_path = srtm.download(case_study.aoi_latlon.bounds, dem_path)

    slope_path = os.path.join(dem_dir, str(year), 'slope.tif')
    os.makedirs(os.path.dirname(slope_path), exist_ok=True)
    if os.path.isfile(slope_path):
        with rasterio.open(slope_path) as src:
            return src.read(1)

    if sar:
        crs, transform = sar.crs, sar.transform
        width, height = sar.width, sar.height
    else:
        crs, transform = landsat.crs, landsat.transform
        width, height = landsat.width, landsat.height

    dem_reproj = dem_path.replace('elevation', 'elevation_reproj')
    dem_reproj = srtm.coregister(dem_path, dem_reproj, crs, transform,
                                 width, height)
    slope = srtm.compute_slope(dem_reproj)
    return slope


def _profile(filename):
    """Get CRS, transform, width and height of a raster."""
    with rasterio.open(filename) as src:
        return (src.crs, src.transform, src.width, src.height)


def _fnames(labels, sar, landsat):
    """Get a list of filenames from a list of feature labels."""
    fnames, labels_ = [], []
    for label in labels:
        if sar:
            if label in sar.available:
                fnames.append(sar.path(label))
                labels_.append(label)
        if landsat:
            if label in landsat.available:
                fnames.append(landsat.path(label))
                labels_.append(label)
    return fnames, labels_


def filter_training(case_study, training_samples, sar, landsat, year):
    """Use satellite data from 2015 to fit an intermediary model that
    can be used to estimate the probability of being built for each pixel
    at the current year. Then, filter the provided training samples
    based on the previously computed probabilities.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    training_samples : 2d array
        Input training labels of shape (height, width).
    sar : maupp.features.SAR
        SAR features. Ignored if None.
    landsat : maupp.features.Landsat
        Landsat features. Ignored if None.
    year : int
        Year of interest.

    Returns
    -------
    filtered : 2d array
        Output filtered training labels.
    """
    labels = ['vv_simple_11x11_1', 'vv_advanced_11x11_1',
              'blue', 'green', 'red', 'nir', 'swir', 'tirs']
    fnames, labels = _fnames(labels, sar, landsat)

    sar2015, landsat2015 = None, None
    if sar:
        sar2015_dir = sar.dir.replace(str(year), '2015')
        sar2015 = features.SAR(sar2015_dir)
    if landsat:
        landsat2015_dir = landsat.dir.replace(str(year), '2015')
        landsat2015 = features.Landsat(landsat2015_dir)

    fnames2015, _ = _fnames(labels, sar2015, landsat2015)
    _, transform2015, width2015, height2015 = _profile(fnames2015[0])
    buildings, blocks, nonbuilt, remote = training_features(
        case_study, 2015, transform2015, width2015, height2015)
    water = water_mask(
        case_study, case_study.crs, transform2015, width2015, height2015)

    samples2015 = training_labels(buildings, blocks, nonbuilt, remote, water)
    samples2015 = classification.limit_samples(
        samples2015, max_samples=50000, random_seed=RANDOM_SEED)
    rf = classification.train(
        filenames=fnames2015,
        training_labels=samples2015,
        n_jobs=-1,
        n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES)

    X = classification.transform(fnames)
    crs, transform, width, height = _profile(fnames[0])
    proba = classification.predict(rf, X, width, height)
    proba = uniform_filter(proba, 3)

    # Write raster to disk
    if sar:
        profile = sar.profile
    else:
        profile = landsat.profile
    _write_raster(
        proba,
        os.path.join(case_study.outputdir, str(year), 'proba_interm.tif'),
        profile)

    filtered = training_samples.copy()
    threshold = np.percentile(proba[filtered == 2], 2015 - year)
    filtered[proba <= threshold] = 0

    # Samples classified as non-built-up in 2015
    proba2015_path = os.path.join(case_study.outputdir, 'probabilities.tif')
    if os.path.isfile(proba2015_path):
        with rasterio.open(proba2015_path) as src:
            proba2015 = np.empty(shape=(height, width), dtype='float32')
            reproject(src.read(1), proba2015, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.bilinear)
        filtered[proba2015 <= 0.75] = 0

    return filtered


def _source(label):
    """Guess the source of a feature according to its label."""
    if label in features.BAND_LABELS or label == 'ndsv':
        return 'landsat'
    else:
        return 'sar'


def _label(filename):
    """Get label of a feature given its file path."""
    return os.path.basename(filename).replace('.tif', '')


def _format_importances(filenames, array):
    """Format feature importances as returned by sklearn as a
    pandas dataframe.
    """
    labels = []
    for fname in filenames:
        label = _label(fname)
        if label in ('ndsv', 'textures_pca'):
            with rasterio.open(fname) as src:
                ndim = src.count
            for i in range(1, ndim+1):
                labels.append(label + '_' + str(i))
        else:
            labels.append(label)

    importances = pd.DataFrame(data=array, index=labels)
    importances['source'] = list(map(_source, importances.index))
    return importances


def classify(case_study, filenames, training_samples, year):
    """Classification of built-up and non-built-up areas with Random
    Forest.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    filenames : list of str
        Filenames of raster datasets.
    training_labels : 2d array
        Training labels (1=BuiltUp, 2=NonBuiltUp).
    year : int
        Year of interest.

    Returns
    -------
    proba : 2d array
        RF probabilistic output (probability of being built-up).
    importances : array
        RF feature importances as an array of shape (n_features).
    """
    model_path = os.path.join(case_study.outputdir, str(year), 'rf.joblib')
    fitted = os.path.isfile(model_path)
    if fitted:
        rf = joblib.load(model_path)
    else:
        rf = classification.train(
            filenames=filenames,
            training_labels=training_samples,
            n_estimators=RF_N_ESTIMATORS,
            max_features=RF_MAX_FEATURES,
            n_jobs=-1)
        joblib.dump(rf, model_path, compress=True)

    crs, transform, width, height = _profile(filenames[0])
    X = classification.transform(filenames)
    proba = classification.predict(rf, X, width, height)
    feature_labels = [os.path.basename(f).replace('.tif', '')
                      for f in filenames]
    importances = _format_importances(filenames, rf.feature_importances_)
    return proba, importances


def validate(case_study, year, margin, crs, transform, width, height, proba):
    """Compute assessment metrics according to an independant
    reference dataset.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    year : int
        Year of interest.
    margin : int
        Temporal margin in years for the validation dataset.
    crs : CRS
        Target CRS.
    transform : Affine
        Target affine transform.
    width : int
        Target raster width.
    height : int
        Target raster height.
    proba : 2d array
        RF probabilistic output.

    Returns
    -------
    metrics : dict
        Assessment metrics.
    """
    ref_dir = os.path.join(case_study.inputdir, 'reference')
    validation_samples = reference.reference_dataset(
        ref_dir, year, margin, crs, transform, width, height)
    return classification.validate(proba, validation_samples, threshold=0.5)


def cross_validation(filenames, training_samples, k=10):
    """Assess model performance with spatial K-Fold cross validation.

    Parameters
    ----------
    filenames : list of str
        Input data as a list of file paths.
    training_samples : 2d array
        Training labels (1=BuiltUp, 2=NonBuiltUp).
    k : int, optional
        Number of folds.

    Returns
    -------
    mean : float
        Mean F1-score.
    std : float
        Standard deviation of the scores.
    """
    scores = classification.cross_validation(
        filenames, training_samples, k=k, max_samples=MAX_TRAINING_SAMPLES)
    return scores.mean(), scores.std()


def _write_raster(raster, filename, profile):
    """Write raster to disk."""
    dst_profile = profile.copy()
    dst_array = raster.copy()
    if raster.dtype == 'bool':
        dst_array = dst_array.astype(np.uint8)
    dst_profile.update(count=1, dtype=dst_array.dtype.name, compress='LZW')
    with rasterio.open(filename, 'w', **dst_profile) as dst:
        dst.write(dst_array, 1)
    return filename


def _write_dict(dictionnary, filename):
    """Write dictionnary to disk as a JSON file."""
    with open(filename, 'w') as f:
        json.dump(dictionnary, f, indent=True)
    return filename


def run(case_study, year):
    """Run full analysis for a given case study and year."""
    out_dir = os.path.join(case_study.outputdir, str(year))
    expected_path = os.path.join(out_dir, 'classes.tif')
    if os.path.isfile(expected_path):
        return True
    os.makedirs(out_dir, exist_ok=True)

    print('Data selection...')
    data_selection(case_study)

    print('Data acquisition...')
    data_acquisition(case_study)

    print('Updating OSM database...')
    ocean_table()
    update_osm_database(case_study)

    print('Preprocessing satellite imagery...')
    imagery_preprocessing(case_study, year)

    print('Features extraction...')
    sar, landsat, water = feature_extraction(case_study, year)
    filenames = []
    if sar:
        filenames.append(sar.path('textures_pca'))
    if landsat:
        filenames += [landsat.path(label) for label in features.BAND_LABELS
                      if label in landsat.available]
    if len(filenames) == 0:
        raise MissingDataError('Satellite imagery not available.')

    print('Training data collection...')
    crs, transform, width, height = _profile(filenames[0])

    trn_labels_f = os.path.join(out_dir, 'training_labels.tif')
    if os.path.isfile(trn_labels_f):
        with rasterio.open(trn_labels_f) as src:
            training_samples = src.read(1)
    else:
        buildings, blocks, nonbuilt, remote = training_features(
            case_study, year, transform, width, height)
        training_samples = training_labels(
            buildings, blocks, nonbuilt, remote, water)

    if year < 2015:
        training_samples = filter_training(
            case_study, training_samples, sar, landsat, year)

    training_samples = classification.limit_samples(
        training_samples, MAX_TRAINING_SAMPLES, RANDOM_SEED)

    print('Classification...')
    proba, importances = classify(
        case_study, filenames, training_samples, year)

    slope = compute_slope(case_study, sar, landsat, year)
    classes = classification.post_processing(
        proba, slope, water, PROBA_THRESHOLD, MAX_SLOPE, FILTER_SIZE)

    print('Validation...')
    metrics = {}
    reference_dir = os.path.join(case_study.inputdir, 'reference')
    if reference.available(reference_dir, year, margin=2):
        metrics = validate(
            case_study, year, 2, crs, transform, width, height, proba)
        metrics['fpr'] = list(metrics['fpr'])
        metrics['tpr'] = list(metrics['tpr'])

    training_samples_ = classification.limit_samples(
        training_samples, 50000, RANDOM_SEED)
    cv_mean, cv_std = cross_validation(filenames, training_samples_, k=10)
    metrics['cv_mean'] = cv_mean
    metrics['cv_std'] = cv_std

    if sar:
        profile = sar.profile
    else:
        profile = landsat.profile

    print('Writing results to disk...')

    for data, fname in zip(
            (water, proba, classes, training_samples),
            ('water.tif', 'probabilities.tif', 'classes.tif', 'training_samples.tif')):
        _write_raster(data, os.path.join(out_dir, fname), profile)

    _write_dict(metrics, os.path.join(out_dir, 'metrics.json'))
    importances.to_csv(os.path.join(out_dir, 'importances.csv'))

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('case_study', type=str, help='case study name')
    args = parser.parse_args()

    # Case study metadata
    case_studies = pd.read_csv(
        os.path.join(DATA_DIR, 'input', 'case-studies.csv'),
        index_col=0)
    name = args.case_study
    case_study = CaseStudy(
        name=args.case_study,
        lat=case_studies.loc[name].latitude,
        lon=case_studies.loc[name].longitude,
        datadir=DATA_DIR,
        country=case_studies.loc[name].country)

    os.makedirs(case_study.inputdir, exist_ok=True)
    os.makedirs(case_study.outputdir, exist_ok=True)

    for year in reversed(sorted(YEARS)):

        if os.path.isfile(os.path.join(case_study.outputdir, str(year), 'probabilities.tif')):
            print(str(year) + ' : done.')
            continue

        try:
            run(case_study, year)
            print(str(year) + ' : done.')
        except MissingDataError:
            print(str(year) + ' : No satellite data available. Skipping...')

