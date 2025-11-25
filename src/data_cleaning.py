import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
from utils import logger, save_dataframe
from config import *

class DataCleaner:
    def __init__(self):
        self.column_map = ACTUAL_COLUMNS
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning pipeline"""
        logger.info("Starting data cleaning pipeline...")
        
        df_clean = df.copy()
        
        # 1. Standardize column names
        df_clean = self.standardize_column_names(df_clean)
        
        # 2. Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # 3. Clean text fields
        df_clean = self.clean_text_fields(df_clean)
        
        # 4. Handle numeric outliers
        df_clean = self.handle_numeric_outliers(df_clean)
        
        # 5. Create aggregated restaurant-level data
        restaurant_df = self.create_restaurant_aggregates(df_clean)
        
        # 6. Validate cleaning results
        self.validate_cleaning(df_clean, restaurant_df)
        
        return df_clean, restaurant_df
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores"""
        column_mapping = {
            'State': 'state',
            'City': 'city',
            'Restaurant Name': 'restaurant_name', 
            'Location': 'location',
            'Category': 'category',
            'Dish Name': 'dish_name',
            'Price (INR)': 'price',
            'Rating': 'rating',
            'Rating Count': 'rating_count'
        }
        df = df.rename(columns=column_mapping)
        logger.info("✅ Column names standardized")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        duplicates_removed = initial_count - final_count
        
        logger.info(f"✅ Duplicates removed: {duplicates_removed} records")
        return df
    
    def clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields"""
        text_columns = ['state', 'city', 'restaurant_name', 'location', 'category', 'dish_name']
        
        for col in text_columns:
            if col in df.columns:
                # Convert to string, handle NaN, strip whitespace, normalize case
                df[col] = (df[col]
                          .astype(str)
                          .str.strip()
                          .str.lower()
                          .replace('nan', np.nan)
                          .replace('none', np.nan))
                
                # Remove extra spaces
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        logger.info("✅ Text fields cleaned and standardized")
        return df
    
    def handle_numeric_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns using IQR method"""
        
        # Price outlier handling
        if 'price' in df.columns:
            Q1 = df['price'].quantile(0.25)
            Q3 = df['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            df['price'] = np.where(df['price'] < lower_bound, lower_bound, df['price'])
            df['price'] = np.where(df['price'] > upper_bound, upper_bound, df['price'])
            
            logger.info(f"✅ Price outliers handled: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Rating count outlier handling
        if 'rating_count' in df.columns:
            # Use log transformation for highly skewed data
            df['rating_count_log'] = np.log1p(df['rating_count'])
            logger.info("✅ Rating count log transformation applied")
        
        return df
    
    def create_restaurant_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create restaurant-level aggregated dataset"""
        
        # Make sure category column is properly handled
        df_agg = df.copy()
        df_agg['category'] = df_agg['category'].fillna('unknown').astype(str)
        
        restaurant_agg = df_agg.groupby(['restaurant_name', 'city', 'location']).agg({
            'price': ['mean', 'count', 'std'],
            'rating': ['mean', 'median', 'std'],
            'rating_count': ['sum', 'mean', 'max'],
            'category': lambda x: ', '.join(x.unique().astype(str)),  # Fixed this line
            'dish_name': 'count'
        }).round(2)
        
        # Flatten column names
        restaurant_agg.columns = [
            'avg_price', 'dish_count', 'price_std',
            'avg_rating', 'median_rating', 'rating_std', 
            'total_rating_count', 'avg_rating_count', 'max_rating_count',
            'categories', 'total_dishes'
        ]
        
        restaurant_agg = restaurant_agg.reset_index()
        
        # Create restaurant-level features
        restaurant_agg['price_per_dish'] = restaurant_agg['avg_price']
        restaurant_agg['popularity_score'] = (
            restaurant_agg['avg_rating'] * np.log1p(restaurant_agg['total_rating_count'])
        )
        
        # Create classification targets
        restaurant_agg['is_high_rated'] = (restaurant_agg['avg_rating'] >= RATING_THRESHOLD).astype(int)
        restaurant_agg['is_popular'] = (restaurant_agg['total_rating_count'] >= POPULARITY_THRESHOLD).astype(int)
        
        price_threshold = restaurant_agg['avg_price'].quantile(PRICE_THRESHOLD_PERCENTILE)
        restaurant_agg['is_premium'] = (restaurant_agg['avg_price'] >= price_threshold).astype(int)
        
        logger.info(f"✅ Restaurant aggregates created: {restaurant_agg.shape}")
        return restaurant_agg
    
    def validate_cleaning(self, df_clean: pd.DataFrame, restaurant_df: pd.DataFrame):
        """Validate cleaning results"""
        logger.info("\n🔍 Cleaning Validation Results:")
        logger.info(f"   Original data shape: {len(df_clean)} rows")
        logger.info(f"   Restaurant aggregates: {len(restaurant_df)} restaurants")
        logger.info(f"   Average dishes per restaurant: {len(df_clean)/len(restaurant_df):.1f}")
        
        # Check data quality
        missing_after = df_clean.isnull().sum().sum()
        duplicates_after = df_clean.duplicated().sum()
        
        logger.info(f"   Missing values after cleaning: {missing_after}")
        logger.info(f"   Duplicates after cleaning: {duplicates_after}")

def run_data_cleaning_pipeline(df: pd.DataFrame):
    """Run the complete data cleaning pipeline"""
    cleaner = DataCleaner()
    df_clean, restaurant_df = cleaner.clean_dataset(df)
    
    # Save cleaned data
    save_dataframe(df_clean, "swiggy_dishes_clean.csv")
    save_dataframe(restaurant_df, "swiggy_restaurants_clean.csv")
    
    return df_clean, restaurant_df