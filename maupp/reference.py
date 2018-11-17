"""Find and rasterize reference samples."""

import os
import json

from rasterio.features import rasterize
from rasterio.crs import CRS

from maupp.utils import reproject_features


WGS84 = CRS(init='epsg:4326')
LEGEND = {'builtup': 1, 'baresoil': 2,
          'lowveg': 3, 'highveg': 4, 'nonbuilt': 5}


def _parse(filename):
    """Parse reference data filename to extract land cover and year."""
    basename = filename.split('.')[0]
    land_cover, year = basename.split('_')
    year = int(year)
    return land_cover, year


def available(reference_dir, year, margin):
    """List available reference data for a given case study and
    a given year.

    Parameters
    ----------
    reference_dir : str
        Path to the directory where reference .geojson files are stored.
    year : int
        Year of interest.
    margin : int
        Max. temporal margin (in years) between the year of interest and
        the reference data.

    Returns
    -------
    available_files : list of str
        Available data as a list of filenames.
    """
    available_files = []
    if not os.path.isdir(reference_dir):
        return []

    for m in range(0, margin + 1):
        for filename in os.listdir(reference_dir):
            _, file_year = _parse(filename)
            if file_year in (year - m, year + m):
                available_files.append(filename)
        # Return early if data found with lower margin
        if available_files:
            return available_files

    return available_files


def reference_dataset(reference_dir, year, margin, crs, transform,
                      width, height):
    """Get a reference dataset raster with the following legend:
    0=NA, 1=Built-Up, 2=Bare Soil, 3=Low Vegetation, 4=High Vegetation,
    and 5=Non-Built-Up.

    Parameters
    ----------
    reference_dir : str
        Path to the directory where reference .geojson files are stored.
    year : int
        Year of interest.
    margin : int
        Max. temporal margin (in years) between the year of interest and
        the reference data.
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
    reference_data : numpy 2d array
        Reference data raster as a 2d numpy array.
    """
    available_files = available(reference_dir, year, margin)
    if not available_files:
        return None

    shapes = []
    for filename in available_files:
        with open(os.path.join(reference_dir, filename)) as fp:
            collection = json.load(fp)
        features = reproject_features(collection['features'], WGS84, crs)
        land_cover, _ = _parse(filename)
        shapes += [(ftr['geometry'], LEGEND[land_cover]) for ftr in features]

    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        dtype='uint8')
