import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Union, List, Dict, Any
import logging
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_plotting():
    """Set up consistent plotting style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def save_model(model, filename: str):
    """Save trained model"""
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filename: str):
    """Load trained model"""
    filepath = MODELS_DIR / filename
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model

def save_dataframe(df: pd.DataFrame, filename: str):
    """Save DataFrame to processed data directory"""
    filepath = DATA_PROCESSED / filename
    df.to_csv(filepath, index=False)
    logger.info(f"DataFrame saved to {filepath}")

def load_dataframe(filename: str) -> pd.DataFrame:
    """Load DataFrame from processed data directory"""
    filepath = DATA_PROCESSED / filename
    df = pd.read_csv(filepath)
    logger.info(f"DataFrame loaded from {filepath}")
    return df

def memory_usage(df: pd.DataFrame) -> str:
    """Get memory usage of DataFrame in MB"""
    return f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"

def print_data_info(df: pd.DataFrame, name: str = "Dataset"):
    """Print comprehensive data information"""
    print(f"=== {name} Information ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {memory_usage(df)}")
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False))