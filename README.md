[![DOI](https://zenodo.org/badge/157991272.svg)](https://zenodo.org/badge/latestdoi/157991272)

This is the code supporting the creation of the MAUPP dataset. Results can be explored and downloaded in the [MAUPP Website](https://maupp.ulb.ac.be/page/wp1results/).

Reference
=========

Yann Forget, Michal Shimoni, Marius Gilbert, and Catherine Linard. *Mapping 20 Years of Urban Expansion in 45 Urban Areas of sub-Saharan Africa*. 2020.

Usage
=====

``` bash
# Install package in a python environment
git clone https://github.com/yannforget/maupp
cd maupp
pip install -e .

# Setup a POSTGIS database where OSM data is
# going to be stored.

# Fill scripts/config.sample.py accordingly and
# rename it config.py.

# Run the processing chain
python scripts/run.py <name_of_case_study>
```
