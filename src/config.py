import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Data files
RAW_DATA_FILE = DATA_RAW / "swiggy.csv"
CLEAN_DATA_FILE = DATA_PROCESSED / "swiggy_clean.csv"
FEATURES_DATA_FILE = DATA_PROCESSED / "swiggy_features.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Target definitions - REMOVE THESE DUPLICATES OR KEEP ONLY ONE SET
# RATING_THRESHOLD = 4.2  # ← COMMENT OUT OR REMOVE THIS LINE
# COST_THRESHOLD_PERCENTILE = 0.8  # ← COMMENT OUT OR REMOVE THIS LINE  
# DELIVERY_TIME_THRESHOLD = 30  # ← COMMENT OUT OR REMOVE THIS LINE

# Actual dataset structure based on your output
ACTUAL_COLUMNS = {
    'state': 'State',
    'city': 'City', 
    'restaurant_name': 'Restaurant Name',
    'location': 'Location',
    'category': 'Category',
    'dish_name': 'Dish Name',
    'price': 'Price (INR)',
    'rating': 'Rating',
    'rating_count': 'Rating Count'
}

# Updated target definitions for dish-level data
RATING_THRESHOLD = 4.2
PRICE_THRESHOLD_PERCENTILE = 0.8  # For premium dishes
POPULARITY_THRESHOLD = 100  # Minimum rating count for popular dishes

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)