from codecs import open
from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESC = f.read()


setup(
    name='maupp',
    version='0.1',
    description='MAUPP WP1 Module',
    long_description=LONG_DESC,
    long_description_content_type='text/markdown',
    url='https://github.com/yannforget/maupp',
    author='Yann Forget',
    author_email='yannforget@mailbox.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    keywords=['earth observation', 'gis', 'remote sensing', 'landsat'],
    packages=find_packages(exclude=['docs', 'tests', 'notebooks', 'data']),
    install_requires=[
        'fiona',
        'geojson',
        'geopandas',
        'matplotlib',
        'numpy',
        'pandas',
        'psycopg2',
        'rasterio',
        'scikit-learn',
        'scipy',
        'shapely',
        'tqdm',
        'asarapi',
        'pylandsat',
        'sentinelsat',
        'elevation'
    ],
    include_package_data=True
)
