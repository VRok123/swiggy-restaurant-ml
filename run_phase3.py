# run_phase3.py
import sys
sys.path.append('src')

import pandas as pd
from data_cleaning import run_data_cleaning_pipeline
from advanced_eda import run_advanced_eda
from utils import logger, load_dataframe
from config import *

def main():
    print("🚀 Starting Phase 3: Enhanced Data Cleaning & EDA")
    print("=" * 60)
    
    try:
        # Load raw data
        logger.info("Loading raw data...")
        df_raw = pd.read_csv(RAW_DATA_FILE)
        print(f"📊 Raw data shape: {df_raw.shape}")
        
        # Run data cleaning pipeline
        print("\n🧹 Running Data Cleaning Pipeline...")
        df_clean, restaurant_df = run_data_cleaning_pipeline(df_raw)
        
        # Run advanced EDA
        print("\n📈 Running Advanced EDA...")
        run_advanced_eda(df_clean, restaurant_df)
        
        print("\n✅ Phase 3 completed successfully!")
        print(f"📁 Generated Files:")
        print(f"   - Cleaned dishes: data/processed/swiggy_dishes_clean.csv")
        print(f"   - Restaurant aggregates: data/processed/swiggy_restaurants_clean.csv")
        print(f"   - Enhanced visualizations: figures/")
        
        # Show dataset info
        print(f"\n📊 Final Dataset Info:")
        print(f"   - Clean dishes: {df_clean.shape}")
        print(f"   - Restaurant aggregates: {restaurant_df.shape}")
        print(f"   - Average dishes per restaurant: {len(df_clean)/len(restaurant_df):.1f}")
        
    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        raise

if __name__ == "__main__":
    main()