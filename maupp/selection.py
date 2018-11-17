"""Optical and SAR data selection."""

from datetime import datetime, timedelta
import itertools
import shutil
import tempfile

import asarapi.catalog
import numpy as np
import pandas as pd
import pylandsat
import pylandsat.download
import rasterio
import rasterio.mask
from sentinelsat import SentinelAPI
from shapely.geometry import mapping
from shapely import wkt

from maupp.bqa import _sensor, cloud_mask, fill_mask
from maupp.utils import reproject_geom


def _middle_date(date_a, date_b):
    """Returns middle date between date_a and date_b."""
    if date_a > date_b:
        return date_a - ((date_a - date_b) / 2)
    elif date_b > date_a:
        return date_a + ((date_b - date_a) / 2)
    else:
        return date_a


def coverage(geom_a, geom_b):
    """Calculate coverage of geom_a by geom_b in percents."""
    covered = geom_a.intersection(geom_b).area
    cover = covered / geom_a.area
    return cover * 100


def local_cloud_cover(aoi, crs, product_id):
    """Recalculate cloud cover of a Landsat scene based on an area of
    interest instead of the whole scene. The quality band of the scene
    is downloaded from Google Cloud in a temporary directory in order
    to recalculate the cloud cover.

    Parameters
    ----------
    aoi : shapely geometry
        Area of interest.
    crs : dict
        CRS of the area of interest.
    product_id : str
        Scene product identifier.

    Returns
    -------
    cloud_cover : float
        Cloud cover in percents.
    """
    sensor = _sensor(product_id)
    tmp_dir = tempfile.mkdtemp(prefix='maupp')
    product = pylandsat.Product(product_id)
    bqa_url = product.baseurl + product_id + '_BQA.TIF'
    fpath = pylandsat.download.download_file(bqa_url, tmp_dir)
    with rasterio.open(fpath) as src:
        if src.crs != crs:
            aoi = reproject_geom(aoi, crs, src.crs)
        bqa, _ = rasterio.mask.mask(src, [mapping(aoi)], crop=True)
        bqa = bqa[0, :, :]
    shutil.rmtree(tmp_dir)
    clouds = cloud_mask(bqa, sensor)
    fill = fill_mask(bqa, sensor)
    n_pixels = np.count_nonzero(fill == 0)
    n_clouds = np.count_nonzero(clouds)
    return n_clouds / n_pixels * 100


def find_wrs(aoi, min_cover=99.):
    """Find Landsat WRS paths and rows for a given area of interest.

    Parameters
    ----------
    aoi : shapely geometry
        Area of interest in lat/lon coordinates.
    min_cover : float, optional
        Minimum AOI coverage in percents (default = 99%).

    Returns
    -------
    wrs : tuple of tuple of int
        Tuple of path/row coordinates. The path/row couple that
        covers the largest part of the AOI is returned first.
    """
    catalog = pylandsat.Catalog()
    wrs = catalog.wrs(aoi)
    wrs = pd.DataFrame(wrs)
    wrs.cover = wrs.cover * 100
    wrs.set_index(['path', 'row'], inplace=True)
    wrs.sort_values(by='cover', ascending=False)
    if wrs.cover.max() >= min_cover:
        return (wrs.iloc[0].name,)
    else:
        cover = 0
        combinations = list(itertools.combinations(wrs.index, 2))
        for wrs_a, wrs_b in combinations:
            footprint_a = wkt.loads(wrs.loc[wrs_a].geom)
            footprint_b = wkt.loads(wrs.loc[wrs_b].geom)
            footprint = footprint_a.union(footprint_b)
            cover = coverage(aoi, footprint)
            if cover >= min_cover:
                break
        if cover < min_cover:
            raise ValueError(
                'More than two scenes are needed to cover the AOI.')
        if wrs.loc[wrs_a].cover > wrs.loc[wrs_b].cover:
            return wrs_a, wrs_b
        else:
            return wrs_b, wrs_a


def best_combination_landsat(candidates_a, candidates_b):
    """Find the best combination of scenes to cover an AOI based on
    cloud cover and temporal distance between the scenes.

    Parameters
    ----------
    candidates_a : pandas dataframe
        First set of search results.
    candidates_b : pandas dataframe
        Second set of search results.

    Returns
    -------
    combination : tuple of str
        The two product IDs corresponding to the best combination.
    """
    couples = list(itertools.product(candidates_a.index, candidates_b.index))
    combinations = pd.DataFrame(index=pd.MultiIndex.from_tuples(couples))
    for pid_a, pid_b in couples:
        combinations.at[(pid_a, pid_b), 'cloud_cover'] = (
            candidates_a.loc[pid_a].cloud_cover / 2 +
            candidates_b.loc[pid_b].cloud_cover / 2)
        combinations.at[(pid_a, pid_b), 'distance'] = abs(
            candidates_a.loc[pid_a].sensing_time -
            candidates_b.loc[pid_b].sensing_time).days
    combinations['score'] = combinations.cloud_cover ** 2 * \
        combinations.distance
    combinations = combinations.sort_values(by='score', ascending=True)
    return combinations.iloc[0].name


def select_landsat(case_study, year, margin=365, increase_margin=True,
                   max_cc=25., local_cc=False):
    """Select the best scene(s) that covers the area of interest of a given
    case study for a given year. If two scenes are needed, two product IDs
    are returned. More than two will raise an error.

    Parameters
    ----------
    case_study : maupp.CaseStudy
        Case study of interest.
    year : int
        Year of interest.
    margin : int, optional
        Temporal margin in days.
    increase_margin : bool, optional
        Add 365 days to margin if no results.
    max_cc : float, optional
        Max. cloud cover in percents.
    local_cc : float, optional
        Update cloud cover based on the AOI instead of the full scene.

    Returns
    -------
    scenes : tuple of str
        Tuple of length 1 or 2 with the Product IDs of the selected scene(s).
    """
    aoi = case_study.aoi_latlon
    aoi_utm, aoi_utm_crs = case_study.aoi, case_study.crs
    catalog = pylandsat.Catalog()
    wrs = find_wrs(aoi)
    begin = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    margins = [0]
    if margin:
        margins.append(margin)
    if increase_margin:
        margins.append(margin + 365)
    margins = [timedelta(m) for m in margins]

    # Only one scene is needed to cover the whole area of interest.
    # Keep the first non-empty search results.
    if len(wrs) == 1:

        path, row = wrs[0]
        for margin in margins:
            candidates = catalog.search(
                begin=begin - margin,
                end=end + margin,
                path=int(path),
                row=int(row),
                tiers=['T1'],
                maxcloud=max_cc)
            candidates = pd.DataFrame(candidates)
            if not candidates.empty:
                break

        if candidates.empty:
            return tuple()

        candidates.set_index('product_id', inplace=True)
        if local_cc:
            candidates.cloud_cover = pd.Series(candidates.index).apply(
                lambda x: local_cloud_cover(aoi_utm, aoi_utm_crs, x))

        candidates = candidates.sort_values(by='cloud_cover')
        return (candidates.iloc[0].name, )

    # Two scenes are needed
    elif len(wrs) == 2:

        candidates = []
        for path, row in wrs:
            for margin in margins:
                candidates_ = catalog.search(
                    begin=begin-margin,
                    end=end+margin,
                    path=int(path),
                    row=int(row),
                    tiers=['T1'],
                    maxcloud=max_cc)
                candidates_ = pd.DataFrame(candidates_)
                # If search result not empty, append and stop increasing
                # the temporal margin.
                if not candidates_.empty:
                    candidates_ = candidates_.set_index('product_id')
                    candidates.append(candidates_)
                    break

        # No results
        if len(candidates) < 2:
            return tuple()
        for c in candidates:
            if c.empty:
                return tuple()

        # Calculate local cloud couver
        if local_cc:
            for i, cand in enumerate(candidates):
                candidates[i].cloud_cover = pd.Series(cand.index).apply(
                    lambda x: local_cloud_cover(aoi_utm, aoi_utm_crs, x))

        return best_combination_landsat(candidates[0], candidates[1])

    else:
        raise ValueError(
            'Scene selection not supported for more than 2 path/rows.')


def search_sar(aoi, begin, end, dhus_username=None, dhus_password=None,
               contains=False):
    """Search for SAR scenes from various missions (Sentinel-1, ERS-1,
    ERS-2, Envisat) depending on the date.

    Parameters
    ----------
    aoi : shapely geometry
        Area of interest (lat/lon).
    begin : datetime
        Beginning of the search period.
    end : datetime
        End of the search period.
    dhus_username : str, optional
        Scihub copernicus username.
    dhus_password : str, optional
        Scihub copernicus password.
    contains : bool, optional
        Product footprint must contains AOI geometry.

    Returns
    -------
    products : dataframe
        Search results as a pandas dataframe with product identifiers
        as index.
    """
    # Sentinel-1
    if end.year >= 2015 and dhus_username and dhus_password:
        relation = 'Intersects'
        if contains:
            relation = 'Contains'
        api = SentinelAPI(dhus_username, dhus_password)
        response = api.query(aoi, (begin, end), area_relation=relation,
                             producttype='GRD', polarisationmode='VV VH')
        products = pd.DataFrame(response).T
        if not products.empty:
            products.set_index('identifier', drop=True, inplace=True)
    # ERS, Envisat
    else:
        products = asarapi.catalog.query(
            aoi, begin, end, product='precision', contains=contains)

    # harmonize date column label
    if 'date' not in products.columns and 'beginposition' in products.columns:
        products['date'] = products.beginposition

    return products


def get_combinations_sar(products, aoi):
    """Get a dataframe with all possible combinations of products and calculate
    their coverage of the AOI and the temporal distance between the products.

    Parameters
    ----------
    products : dataframe
        Search results with product identifiers as index.
    aoi : shapely geometry
        Area of interest (lat/lon).

    Returns
    -------
    combinations : dataframe
        Double-indexed output dataframe. Only combinations that contain
        the AOI are returned (with a 1% margin).
    """
    couples = list(itertools.combinations(products.index, 2))
    combinations = pd.DataFrame(index=pd.MultiIndex.from_tuples(couples))
    for id_a, id_b in couples:
        footprint_a = wkt.loads(products.loc[id_a].footprint)
        footprint_b = wkt.loads(products.loc[id_b].footprint)
        footprint = footprint_a.union(footprint_b)
        combinations.at[(id_a, id_b), 'date_a'] = products.loc[id_a].date
        combinations.at[(id_a, id_b), 'date_b'] = products.loc[id_b].date
        combinations.at[(id_a, id_b), 'cover'] = coverage(aoi, footprint)
    combinations = combinations[combinations.cover >= 99.]
    combinations['dist'] = combinations.date_b - combinations.date_a
    combinations.dist = combinations.dist.apply(lambda x: abs(x.days))
    combinations = combinations.sort_values(by='dist', ascending=True)
    return combinations


def best_combination_sar(products, aoi, preferred_date=None):
    """Find the best combination of SAR products to cover a given
    area of interest.

    Parameters
    ----------
    products : dataframe
        Search results with product identifiers as index.
    aoi : shapely geometry
        Area of interest (lat/lon).
    preferred_date : datetime
        Prefered date.

    Returns
    -------
    product_ids : tuple of str
        Tuple containing the product IDs of the combined products.
    """
    combinations = get_combinations_sar(products, aoi)
    if combinations.empty:
        return tuple()

    # calculate distance of each product to the preferred date
    if preferred_date:
        combinations['dist_to_pref'] = (
            abs((combinations.date_a - preferred_date)) +
            abs((combinations.date_b - preferred_date)))
        combinations.dist_to_pref = combinations.dist_to_pref.apply(
            lambda x: x.days)
    else:
        combinations['dist_to_pref'] = 0

    # arbitrary scoring function
    combinations['score'] = (
        combinations.dist + combinations.dist_to_pref +
        (100 - combinations.cover) * 100)
    combinations.sort_values(by='score', ascending=True)
    pid_a, pid_b = combinations.iloc[0].name

    # returns product id with highest initial coverage first
    cover_a = coverage(aoi, wkt.loads(products.loc[pid_a].footprint))
    cover_b = coverage(aoi, wkt.loads(products.loc[pid_b].footprint))
    if cover_a >= cover_b:
        return (pid_a, pid_b)
    else:
        return (pid_b, pid_a)


def select_sar(aoi, year, dhus_username, dhus_password, preferred_date=None,
               margin=0, increase_margin=False):
    """Select the best SAR product(s) that cover a given area of interest
    at a given year. Can return two products if the AOI need two scenes
    to be covered.

    Parameters
    ----------
    aoi : shapely geometry
        Area of interest (lat/lon).
    year : int
        Year of interest.
    dhus_username : str
        Scihub copernicus username.
    dhus_password : str
        Scihub copernicus password.
    preferred_date : datetime, optional
        Prefered date, e.g. from the associated optical scene.
    margin : int, optional
        Temporal margin from year of interest in days.
    increase_margin : bool, optional
        Add 365 days to margin if no results.

    Returns
    -------
    product_ids : tuple of str
        Tuple of length 1 or 2 containing the product ID of the
        selected product(s).
    """
    begin = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    margins = [0]
    if margin:
        margins.append(margin)
    if increase_margin:
        margins.append(margin + 365)
    margins = [timedelta(m) for m in margins]

    for contains, margin in itertools.product([True, False], margins):
        products = search_sar(
            aoi=aoi.wkt,
            begin=begin - margin,
            end=end + margin,
            dhus_username=dhus_username,
            dhus_password=dhus_password,
            contains=contains)
        if not products.empty:
            break

    # no result found
    if products.empty:
        return tuple()
    if len(products) == 1 and not contains:
        return tuple()

    if contains:
        products['dist_pref'] = abs(products.date - preferred_date)
        products.sort_values(by='dist_pref', ascending=True)
        return (products.iloc[0].name, )
    else:
        return best_combination_sar(products, aoi, preferred_date)


def selection(case_study, year, dhus_username, dhus_password, margin=365):
    """Perform data selection for a given case study and year.

    Parameters
    ----------
    case_study : maupp.casestudy.CaseStudy
        Case study of interest.
    year : int
        Year of interest.
    margin : int, optional
        Temporal margin in days.

    Returns
    -------
    products : dict
        Dictionnary with 'landsat' and 'sar' kets containing the
        product IDs of the selected products.
    """
    landsat_products = select_landsat(
        case_study, year, margin, increase_margin=True, local_cc=True)

    # get the acquisition date of landsat products
    # if two products, get average between both dates
    catalog = pylandsat.Catalog()
    if len(landsat_products) == 1:
        date = catalog.metadata(landsat_products[0])['sensing_time']
        date = datetime.strptime(date, '%Y-%m-%d')
    elif len(landsat_products) == 2:
        date_a = catalog.metadata(landsat_products[0])['sensing_time']
        date_b = catalog.metadata(landsat_products[1])['sensing_time']
        date_a = datetime.strptime(date_a, '%Y-%m-%d')
        date_b = datetime.strptime(date_b, '%Y-%m-%d')
        date = _middle_date(date_a, date_b)
    else:
        date = datetime(year, 7, 1)

    sar_products = select_sar(
        aoi=case_study.aoi_latlon,
        year=year,
        dhus_username=dhus_username,
        dhus_password=dhus_password,
        preferred_date=date,
        margin=margin,
        increase_margin=True
    )

    return {'landsat': landsat_products, 'sar': sar_products}
