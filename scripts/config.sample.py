"""Configuration parameters."""

# Main data directory. Must exists.
DATA_DIR = '/home/maupp/code/maupp/data'
REMOVE_INTERMEDIARY_FILES = False

# Years to process
YEARS = [1995, 2000, 2005, 2010, 2015]

# Temporal margin in days for imagery selection
# Will be doubled if no product is available.
MARGIN = 365

# Postgres database where OSM data will be stored
DB_NAME = 'maupp'
DB_USER = ''
DB_PASS = ''
DB_HOST = 'localhost'
DB_PORT = 5432

# ESA SSO Credentials
ESA_SSO_USERNAME = ''
ESA_SSO_PASSWORD = ''

# Scihub Copernicus Credentials
DHUS_USERNAME = ''
DHUS_PASSWORD = ''

# Radii & Offsets for which GLCM textures will be computed
GLCM_RADII = [5]
GLCM_OFFSETS = [1]

# Texture set to compute ('simple', 'advanced', or 'higher')
GLCM_KINDS = ['simple', 'advanced']

# Number of components for dimensionality reduction of GLCM textures
GLCM_N_COMPONENTS = 6

# Max. number of training samples per class
MAX_TRAINING_SAMPLES = 50000

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_FEATURES = 'sqrt'

# Post processing
PROBA_THRESHOLD = 0.9
MAX_SLOPE = 40
FILTER_SIZE = 5

RANDOM_SEED = 111
