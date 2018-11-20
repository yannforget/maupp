"""Pre-processing of SAR products."""

import argparse
import os
import shutil
import subprocess
from tempfile import mkdtemp

import rasterio
from pkg_resources import resource_filename

from maupp.acquisition import guess_platform


ORBIT_FILES = {
    'Sentinel-1': 'Sentinel Precise (Auto Download)',
    'Envisat': 'DORIS Precise VOR (ENVISAT) (Auto Download)',
    'Envisat_ERS': 'DELFT Precise (ENVISAT, ERS1&amp;2) (Auto Download)',
    'ERS': 'PRARE Precise (ERS1&amp;2) (Auto Download)'
}


class GPTError(Exception):
    pass


def _product_id(filename):
    """Guess product identifier from ERS, Envisat or Sentinel-1 filename."""
    basename = os.path.basename(filename)
    platform = guess_platform(basename)
    parts = basename.split('_')
    if platform in ('ERS', 'Envisat'):
        product_id = '_'.join(parts[:8])
    elif platform == ('Sentinel-1'):
        product_id = '_'.join(parts[:9])
    else:
        raise ValueError('Product ID unrecognized.')
    return product_id


def preprocess(product, georegion, epsg, out_dir, auxiliary_product=None):
    """Calibrate, correct and mosaic SAR products for a given area of interest.

    Parameters
    ----------
    product : str
        Path to input product.
    georegion : str
        Area of interest in WKT format (lat/lon).
    epsg : int
        Target EPSG code.
    out_dir : str
        Output directory.
    auxiliary_product : str, optional
        Auxiliary product for mosaicking.
    """
    tmp_dir = mkdtemp(prefix='maupp_')
    product_id = _product_id(product)
    cal = calibrate(product, georegion, epsg, tmp_dir)
    if auxiliary_product:
        cal_aux = calibrate(auxiliary_product, georegion, epsg, tmp_dir)
        cal = mosaic([cal, cal_aux], tmp_dir, 'BILINEAR_INTERPOLATION')
    platform = guess_platform(product_id)
    geotiffs = to_geotiff(cal, out_dir, platform)
    shutil.rmtree(tmp_dir)
    return geotiffs


def run_graph(src_product, graph, debug=False, **kwargs):
    """Perform preprocessing in SNAP on a source product with
    the GPT command-line tool.

    Parameters
    ----------
    src_product : str
        Path to source product. Can also be a list of products.
    graph : str
        Path to .xml SNAP graph.
    debug : bool, optional
        Show/hide output log from Snap.
    kwargs : str, optional
        Additional named parameters passed to the graph.
        Spaces not supported.
    """
    cmd = ['gpt', graph]
    if kwargs:
        for name, value in kwargs.items():
            cmd.append('-P{}={}'.format(name, value))

    if isinstance(src_product, list) or isinstance(src_product, tuple):
        for product in src_product:
            cmd.append(product)
    else:
        cmd.append(src_product)

    if debug:
        p = subprocess.run(cmd)
    else:
        p = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Raise an error if GPT has failed
    if p.returncode == 1:
        raise GPTError
    return True

def calibrate(src_path, georegion, epsg, out_dir):
    """Preprocess ERS, Envisat or Sentinel-1 product with SNAP.
    Preprocessing steps: 1) Update orbit files, 2) Radiometric calibration,
    3) Thermal Noise Removal, 4) Speckle Reduction, 5) Terrain Flattening,
    6) Terrain Correction, 7) AOI subset.

    Parameters
    ----------
    src_path : str
        Path to source product.
    georegion : str
        Area of interest in WKT.
    epsg : int
        Target EPSG for ortho-rectification.
    out_dir : str
        Output directory.

    Returns
    -------
    dst_product : str
        Path to output product.
    """
    # Use a different SNAP xml graph for each platform
    product_id = _product_id(src_path)
    platform = guess_platform(product_id)
    if platform in ('ERS', 'Envisat'):
        graph = resource_filename(__name__, 'snap_graphs/ers_envisat.xml')
    elif platform == 'Sentinel-1':
        graph = resource_filename(__name__, 'snap_graphs/sentinel1.xml')
    else:
        raise ValueError('No snap graph available for this platform.')

    dst_product = os.path.join(out_dir, product_id + '.dim')
    os.makedirs(out_dir, exist_ok=True)

    orbit_type = ORBIT_FILES[platform]

    try:
        run_graph(
            src_product=src_path,
            graph=graph,
            orbitType=orbit_type,
            dstcrs='EPSG:{}'.format(epsg),
            georegion=georegion,
            output=dst_product)

    except GPTError:
        # If GPT failed, try to use DELFT Precise orbit files for
        # ERS and Envisat products.
        if platform in ('ERS', 'Envisat'):
            run_graph(
                src_product=src_path,
                graph=graph,
                orbitType=ORBIT_FILES['Envisat_ERS'],
                dstcrs='EPSG:{}'.format(epsg),
                georegion=georegion,
                output=dst_product)
        # For Sentinel-1 products, try without Thermal Noise Removal.
        elif platform == 'Sentinel-1':
            graph = resource_filename(
                __name__, 'snap_graphs/sentinel1_no_tnr.xml')
            run_graph(
                src_product=src_path,
                graph=graph,
                dstcrs='EPSG:{}'.format(epsg),
                georegion=georegion,
                output=dst_product)
        else:
            raise

    return dst_product


def mosaic(src_products, dst_dir, resampling_method):
    """Mosaic two SAR products with same CRS in SNAP.

    Parameters
    ----------
    src_products : list of str
        List of paths.
    dst_dir : str
        Output directory.
    resampling_method : str
        Resampling method : `NEAREST_NEIGHBOUR`, `BILINEAR_INTERPOLATION`,
        or `CUBIC_CONVOLUTION`.

    Returns
    -------
    dst_product : str
        Path to output product.
    """
    dim_datadir = src_products[0].replace('.dim', '.data')
    sample = [f for f in os.listdir(dim_datadir) if f.endswith('.img')][0]
    with rasterio.open(os.path.join(dim_datadir, sample)) as src:
        height, width = src.height, src.width
        pixel_size = src.transform.a

    run_graph(
        src_product=src_products,
        graph=resource_filename(__name__, 'snap_graphs/mosaic.xml'),
        resamplingMethod=resampling_method,
        sceneHeight=height,
        sceneWidth=width,
        pixelSize=pixel_size,
        output=os.path.join(dst_dir, 'mosaic.dim')
    )

    return os.path.join(dst_dir, 'mosaic.dim')


def to_geotiff(src_product, out_dir, platform):
    """Extract bands of a BEAM-DIMAP product and write them as
    individual GeoTIFFs.

    Parameters
    ----------
    src_product : str
        Path to source BEAM-DIMAP product.
    out_dir : str
        Output directory.
    platform : str
        ERS, Envisat or Sentinel-1.

    Returns
    -------
    dst_products : tuple of str
        Tuple of path to output files.
    """
    if platform.lower() in ['ers', 'envisat']:
        output = os.path.join(out_dir, 'vv.tif')
        graph = resource_filename(__name__, 'snap_graphs/vv.xml')
        run_graph(
            src_product=src_product,
            graph=graph,
            outputVV=output
        )
        return (output, )
    elif platform.lower() == 'sentinel-1':
        output_vv = os.path.join(out_dir, 'vv.tif')
        output_vh = os.path.join(out_dir, 'vh.tif')
        graph = resource_filename(__name__, 'snap_graphs/vvvh.xml')
        run_graph(
            src_product=src_product,
            graph=graph,
            outputVV=output_vv,
            outputVH=output_vh
        )
        return (output_vv, output_vh)
    else:
        raise ValueError('Platform unrecognized.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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

    preprocess(args.source, args.aoi, args.epsg, args.outdir, args.aux)
