"""Downloading Landsat and SAR imagery."""

import fnmatch
import os

import requests
from requests.auth import HTTPBasicAuth
from pylandsat import Product
from sentinelsat.sentinel import SentinelAPI
from asarapi.download import log_in, log_out, request_download

from maupp.utils import unzip


def guess_platform(product_id):
    """Guess platform of a product according to its identifier."""
    if len(product_id) == 40 and product_id.startswith('L'):
        return 'Landsat'
    if product_id.startswith('ASA'):
        return 'Envisat'
    if product_id.startswith('SAR'):
        return 'ERS'
    if product_id.startswith('S1'):
        return 'Sentinel-1'
    raise ValueError('Unrecognized product ID.')


def _is_online(uuid, dhus_username, dhus_password, request_reupload=True):
    """Check if a given Sentinel product is online or achived. If not,
    request a re-upload.

    Parameters
    ----------
    uuid : str
        Product UUID.
    request_reupload : bool, optional
        Request re-upload if the product is offline.

    Returns
    -------
    online : bool
        True if online, False if offline.
    """
    url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/Online/$value"
    auth = HTTPBasicAuth(dhus_username, dhus_password)
    r = requests.get(url.format(uuid=uuid), auth=auth)
    online = r.text == 'true'
    if not online and request_reupload:
        url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/$value"
        requests.get(url.format(uuid=uuid), auth=auth)
    return online


def download(product_id, output_dir, esa_sso_username, esa_sso_password,
             dhus_username, dhus_password):
    """Download product given its identifier.

    Parameters
    ----------
    product_id : str
        Landsat, ERS, Envisat or Sentinel-1 product ID.
    output_dir : str
        Output directory.

    Returns
    -------
    product_dir : str
        Product directory.
    """
    platform = guess_platform(product_id)
    # Avoid if product is already downloaded
    product_path = find_product(product_id, output_dir)
    if product_path:
        return product_path

    if platform == 'Landsat':
        product = Product(product_id)
        product.download(output_dir, progressbar=False)

    elif platform in ('ERS', 'Envisat'):
        session = log_in(esa_sso_username, esa_sso_password)
        try:
            request_download(session, product_id,
                             output_dir, progressbar=False)
        except FileExistsError:
            pass
        log_out(session)

    else:
        api = SentinelAPI(dhus_username, dhus_password,
                          show_progressbars=False)
        meta = api.query(filename=product_id + '*')
        uuid = list(meta)[0]
        if _is_online(uuid, dhus_username, dhus_password):
            api.download(uuid, output_dir)
            unzip(os.path.join(output_dir, product_id + '.zip'))
        else:
            raise requests.exceptions.HTTPError(
                '503: Product offline. Re-upload requested.')
    
    return find_product(product_id, output_dir)


def find_product(product_id, directory):
    """Find a satellite product and returns its absolute path.

    Parameters
    ----------
    product_id : str
        ERS, Envisat, Sentinel-1 or Landsat product identifier.
    directory : str
        Directory to search.

    Returns
    -------
    path : str
        Path to file or directory of the product.

    Raises
    ------
    FileNotFoundError
        If product not found.
    """
    pattern = '*' + product_id + '*'
    matches = fnmatch.filter(os.listdir(directory), pattern)
    if not matches:
        return None
    return os.path.join(directory, matches[0])
