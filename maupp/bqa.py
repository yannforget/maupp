"""Extract binary masks from Landsat Quality Band."""

import numpy as np


# Bit locations
BITS = {
    'OLI_TIRS': {
        'fill': [0],
        'terrain_occlusion': [1],
        'radiometric_saturation': [3, 2],
        'cloud': [4],
        'cloud_confidence': [6, 5],
        'cloud_shadow_confidence': [8, 7],
        'snow_ice_confidence': [10, 9],
        'cirrus_confidence': [12, 11]
    },
    'TM': {
        'fill': [0],
        'dropped_pixel': [1],
        'radiometric_saturation': [3, 2],
        'cloud': [4],
        'cloud_confidence': [6, 5],
        'cloud_shadow_confidence': [8, 7],
        'snow_ice_confidence': [10, 9],
    },
    'MSS': {
        'fill': [0],
        'dropped_pixel': [1],
        'radiometric_saturation': [3, 2],
        'cloud': [4]
    }
}


BITS['OLI'] = BITS['OLI_TIRS']
BITS['ETM'] = BITS['TM']


def _sensor(product_id):
    """Find sensor ID from product ID."""
    sid = product_id[:2]
    if sid == 'LC':
        return 'OLI_TIRS'
    elif sid == 'LO':
        return 'OLI'
    elif sid == 'LE':
        return 'ETM'
    elif sid == 'LT':
        return 'TM'
    elif sid == 'LM':
        return 'MSS'


def _capture_bits(bqa, b1, b2):
    width_int = int((b1 - b2 + 1) * '1', 2)
    return ((bqa >> b2) & width_int).astype('uint8')


def _get_mask(bqa, condition, sensor):
    bits = BITS[sensor][condition]
    if len(bits) == 2:
        b1, b2 = bits
    else:
        b1 = bits[0]
        b2 = b1
    return _capture_bits(bqa, b1, b2)


def cloud_mask(bqa, sensor):
    """Get binary cloud mask."""
    clouds = _get_mask(bqa, condition='cloud', sensor=sensor)
    if 'cloud_shadow_confidence' in BITS[sensor]:
        shadows = _get_mask(
            bqa, condition='cloud_shadow_confidence', sensor=sensor)
        clouds = (clouds == 1) | (shadows > 1)
    return clouds.astype('uint8')


def fill_mask(bqa, sensor):
    """Get binary fill mask."""
    return _get_mask(bqa, condition='fill', sensor=sensor).astype(np.bool)
