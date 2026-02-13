"""
Configuration module for MT evaluation pipeline.
Stores all configurable parameters in one place.
"""
from pathlib import Path

# Model configurations
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"

# COMET prediction parameters
COMET_BATCH_SIZE = 8
COMET_USE_GPU = True  # Set to False to force CPU usage

# File paths - adjust relative to the MT Evaluation folder
DATA_DIR = Path("../Dataset/Literary")
RESULTS_DIR = Path("../Results")

# Language-specific file mappings
LANGUAGE_FILES = {
    'Hindi': DATA_DIR / 'Hindi.csv',
    'Marathi': DATA_DIR / 'Marathi.csv',
    'Odia': DATA_DIR / 'Odia.csv',
    'Tamil': DATA_DIR / 'Tamil.csv'
}

# Required columns in input CSV
REQUIRED_COLUMNS = ['Source', 'Reference']

# Output configuration
OUTPUT_DECIMAL_PLACES = {
    'BLEU': 2,
    'METEOR': 4,
    'COMET': 4,
    'COMET_QE': 4
}

# NLTK data requirements
NLTK_PACKAGES = ['wordnet', 'punkt', 'omw-1.4']

# Suppress warnings
SUPPRESS_WARNINGS = True
