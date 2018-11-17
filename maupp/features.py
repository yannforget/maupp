"""Access or compute features from preprocessed satellite data."""


from itertools import product
import os
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio import warp
from sklearn.decomposition import IncrementalPCA

from maupp.bqa import cloud_mask
from maupp.glcm import (compute_textures, rescale_to_uint8, histogram_cutting,
                        TEXTURES)


BAND_LABELS = ['blue', 'green', 'red', 'nir', 'swir', 'swir2', 'tirs', 'tirs2']


def calc_ndi(bi, bj):
    """Compute normalized difference index for a given
    couple of bands.
    """
    # Transform input bands so that bi + bj != 0
    if bi.min() < 1:
        bi = bi + np.abs(bi.min()) + 1
    if bj.min() < 1:
        bj = bj + np.abs(bj.min()) + 1
    # Calculate NDI
    ndi = (bi - bj) / (bi + bj)
    return ndi


def _to_numeric(value):
    """Try to convert a string to an integer or a float.
    If not possible, returns the initial string.
    """
    try:
        value = int(value)
    except:
        try:
            value = float(value)
        except:
            pass
    return value


def parse_mtl(fpath):
    """Parse MTL metadata as a dict."""
    meta = {}
    f = open(fpath)
    for line in f.readlines():
        if line.startswith('GROUP = L1_METADATA_FILE'):
            pass  # ignore main group
        elif line.strip().startswith('END'):
            pass  # ignore end statements
        elif line.strip().startswith('GROUP'):
            group = line.split('=')[-1].strip()
            meta[group] = {}
        else:
            param, value = [s.strip() for s in line.split('=')]
            value = value.replace('"', '')
            value = _to_numeric(value)
            meta[group][param] = value
    return meta


class Features:

    def __init__(self, directory):
        """Access and compute features."""
        self.dir = directory
        self.profile = self.find_profile()

    def __getattr__(self, name):
        """Returns image data or metadata if possible."""
        fpath = os.path.join(self.dir, name + '.tif')
        if os.path.isfile(fpath):
            with rasterio.open(fpath) as src:
                if src.count == 1:
                    return src.read(1)
                else:
                    return src.read()
        elif name in self.profile:
            return self.profile[name]
        else:
            raise AttributeError()

    def __iter__(self):
        """Iterate over feature labels and filenames."""
        for label in self.available:
            yield (label, self.path(label))

    def path(self, label):
        """Get path to a given feature according to its label."""
        expected_path = os.path.join(self.dir, label + '.tif')
        if os.path.isfile(expected_path):
            return expected_path
        else:
            raise FileNotFoundError('Feature does not exist.')

    @property
    def available(self):
        """List available features."""
        files = [f.split('.') for f in os.listdir(self.dir)]
        labels = [label for label, ext in files if ext.lower() == 'tif']
        if not labels:
            raise FileNotFoundError('No feature available.')
        return labels

    def find_profile(self):
        """Get the profile of the first feature."""
        for _, path in self:
            with rasterio.open(path) as src:
                return src.profile
            break


class Landsat(Features):

    def metadata(self, which='primary'):
        """Access MTL metadata of `primary` or `secondary` scene that has been
        used to build the mosaic.
        """
        fpath = os.path.join(self.dir, which + '_MTL.txt')
        return parse_mtl(fpath)

    @property
    def ndsv_labels(self):
        """Get NDSV labels corresponding to all the possible band combinations
        as a list of tuples.
        """
        bands = [b for b in BAND_LABELS if b in self.available]
        combinations = []
        for i, bi in enumerate(bands):
            for j, bj in enumerate(bands[i+1:]):
                combinations.append((bi, bj))
        return combinations

    def compute_ndsv(self):
        """Compute normalized difference spectral vector."""
        fpath = os.path.join(self.dir, 'ndsv.tif')
        combinations = self.ndsv_labels
        profile = self.profile.copy()
        profile.update(count=len(combinations), dtype='float32')
        with rasterio.open(fpath, 'w', **profile) as dst:
            for v, (bi_label, bj_label) in enumerate(combinations):
                bi = getattr(self, bi_label).astype('float32')
                bj = getattr(self, bj_label).astype('float32')
                dst.write(calc_ndi(bi, bj), v+1)
        return fpath

    @property
    def quality(self):
        """Quality band."""
        with rasterio.open(os.path.join(self.dir, 'quality.tif')) as src:
            return src.read(1)

    @property
    def clouds(self):
        """Cloud mask."""
        sensor = self.metadata()['PRODUCT_METADATA']['SENSOR_ID']
        return cloud_mask(self.quality, sensor)

    def coregister(self, primary):
        """Co-register all Landsat features with a given primary raster.
        Multibands rasters are ignored.
        """
        with rasterio.open(primary) as src:
            dst_crs = src.crs
            dst_width = src.width
            dst_height = src.height
            dst_transform = src.transform

        for label, path in self:
            if 'quality' in label:
                resampling = warp.Resampling.nearest
            else:
                resampling = warp.Resampling.cubic
            with rasterio.open(path) as src:
                if src.count > 1:
                    continue
                dst_profile = src.profile.copy()
                dst_array = np.empty((dst_height, dst_width), src.dtypes[0])
                warp.reproject(
                    source=src.read(1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling)
            dst_profile.update(
                crs=dst_crs,
                width=dst_width,
                height=dst_height,
                transform=dst_transform)
            with rasterio.open(path, 'w', **dst_profile) as dst:
                dst.write(dst_array, 1)

        return True


class SAR(Features):

    def textures(self, water=None, radii=[5], offsets=[1],
                 kinds=['simple', 'advanced', 'higher']):
        """Compute GLCM textures for a given list of radii and offsets.

        Parameters
        ----------
        water : numpy 2d array, optional
            Water mask.
        radii : list of int, optional
            X and Y radius values.
        offsets : list of int, optional
            X and Y offset values.
        kinds : list of str, optional
            List of texture sets to compute.
        """
        polarisations = ['vv']
        if 'vh' in self.available:
            polarisations.append('vh')

        for polar in polarisations:

            src_array = getattr(self, polar)
            src_array = histogram_cutting(src_array, percent=2, mask=water)
            src_array = rescale_to_uint8(src_array)

            src_profile = self.profile.copy()
            src_profile.update(dtype=src_array.dtype.name)

            with TemporaryDirectory() as tmp_dir:
                src_raster = os.path.join(tmp_dir, 'src_raster.tif')
                with rasterio.open(src_raster, 'w', **src_profile) as dst:
                    dst.write(src_array, 1)

                for radius, offset, kind in product(radii, offsets, kinds):
                    fname = '{polar}_{kind}_{size}x{size}_{offset}.tif'.format(
                        polar=polar, kind=kind, size=radius*2+1, offset=offset)
                    dst_raster = os.path.join(self.dir, fname)
                    compute_textures(
                        src_raster, dst_raster, kind, radius, offset)

        return True

    def dim_reduction(self, n_components=6):
        """Perform dimensionality reduction on GLCM textures using
        incremental PCA.
        """
        n_features = 0
        for label, kind in product(self.available, TEXTURES.keys()):
            if kind in label:
                n_features += len(TEXTURES[kind])
        n_samples = self.width * self.height

        with TemporaryDirectory() as tmp_dir:

            fp = np.memmap(
                filename=os.path.join(tmp_dir, 'textures.dat'),
                dtype='float32',
                mode='w+',
                shape=(n_samples, n_features)
            )
            for label in self.available:
                i, j = 0, 0
                if 'simple' in label:
                    n = 8
                elif 'advanced' in label:
                    n = 10
                elif 'higher' in label:
                    n = 11
                else:
                    continue
                j += n
                data = getattr(self, label)
                fp[:, i:j] = data.reshape(n, n_samples).swapaxes(0, 1)
                i += n

            pca = IncrementalPCA(
                n_components=n_components, copy=False, batch_size=100000)
            X = pca.fit_transform(fp)

        img = X.swapaxes(0, 1)
        img = img.reshape(n_components, self.height, self.width)
        profile = self.profile.copy()
        profile.update(count=n_components, dtype=img.dtype.name)
        filename = os.path.join(self.dir, 'textures_pca.tif')
        with rasterio.open(filename, 'w', **profile) as dst:
            for i in range(n_components):
                dst.write(img[i, :, :], i+1)
        return filename
