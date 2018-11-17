"""Compute simple, advanced and higher-order Haralick textures with Orfeo Toolbox.
https://www.orfeo-toolbox.org/CookBook/Applications/app_HaralickTextureExtraction.html
"""

import subprocess

import numpy as np


TEXTURES = {
    'simple': [
        'energy',
        'entropy',
        'correlation',
        'inverse_difference_moment',
        'inertia',
        'cluster_shade',
        'cluster_prominence',
        'haralick_correlation'
    ],
    'advanced': [
        'mean',
        'variance',
        'dissimilarity',
        'sum_average',
        'sum_variance',
        'sum_entropy',
        'difference_of_entropies',
        'difference_of_variances',
        'ic1',
        'ic2'
    ],
    'higher': [
        'short_run_emphasis',
        'long_run_emphasis',
        'grey_level_nonuniformity',
        'run_length_nonuniformity',
        'run_percentage',
        'low_grey_level_run_emphasis',
        'high_grey_level_run_emphasis',
        'short_run_low_grey_level_emphasis',
        'short_run_high_grey_level_emphasis',
        'long_run_low_grey_level_emphasis',
        'long_run_high_grey_level_emphasis'
    ]
}


def histogram_cutting(src_array, percent=2, mask=None):
    """Perform histogram cutting on input 2d array.

    Parameters
    ----------
    src_array : numpy 2d array
        Input raster.
    percent : int, optional
        Cut percentage.
    mask : numpy 2d array, optional
        Mask applied to src_array.

    Returns
    -------
    dst_array : numpy 2d array
        Output raster.
    """
    dst_array = src_array.copy()
    if isinstance(mask, np.ndarray):
        values = src_array[mask].ravel()
    else:
        values = src_array.ravel()
    vmin = np.percentile(values, percent)
    vmax = np.percentile(values, 100-percent)
    dst_array[dst_array < vmin] = vmin
    dst_array[dst_array > vmax] = vmax
    return dst_array


def rescale_to_uint8(src_array):
    """Rescale data to UINT8 range.

    Parameters
    ----------
    src_array : numpy 2d array
        Input raster.

    Returns
    -------
    dst_array : numpy 2d array
        Output UINT8 raster.
    """
    vmin = src_array.min()
    vmax = src_array.max()
    dst_array = np.interp(src_array, (vmin, vmax), (0, 255))
    return dst_array.astype(np.uint8)


def compute_textures(src_raster, dst_raster, kind='simple',
                     radius=2, offset=1, nb_bins=64):
    """Compute simple, advanced or higher order Haralick textures with
    OrfeoToolBox.

    Parameters
    ----------
    src_raster : str
        Path to input raster (UINT8).
    dst_raster : str
        Path to output raster.
    kind : str, optional
        'simple', 'advanced', or 'higher' (default='simple').
    radius : int, optional
        X and Y radius.
    offset : int, optional
        X and Y offsets.
    nb_bins : int, optional
        Histogram number of bins.

    Returns
    -------
    dst_raster : str
        Path to output raster.
    """
    subprocess.run([
        'otbcli_HaralickTextureExtraction', '-in', src_raster,
        '-parameters.xrad', str(radius), '-parameters.yrad', str(radius),
        '-parameters.xoff', str(offset), '-parameters.yoff', str(offset),
        '-parameters.min', str(0), '-parameters.max', str(255),
        '-parameters.nbbin', str(nb_bins), '-texture', kind,
        '-out', dst_raster, 'double'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return dst_raster
