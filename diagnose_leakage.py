# diagnose_leakage.py
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger, setup_plotting
from config import *

def analyze_feature_target_relationship():
    """Analyze relationships between features and targets to detect leakage"""
    logger.info("Analyzing feature-target relationships for data leakage...")
    
    # Load data
    features_df = pd.read_csv(DATA_PROCESSED / "model_features.csv")
    restaurants_df = pd.read_csv(DATA_PROCESSED / "swiggy_restaurants_enhanced.csv")
    
    X = features_df
    y_high = restaurants_df['is_high_rated']
    y_popular = restaurants_df['is_popular']
    y_premium = restaurants_df['is_premium']
    
    feature_names = features_df.columns.tolist()
    
    print("🔍 Feature-Target Relationship Analysis")
    print("=" * 60)
    
    # Check for features that might contain target information
    suspicious_features = []
    
    for feature in feature_names:
        feature_series = X[feature]
        
        # Check correlation with targets
        corr_high = np.corrcoef(feature_series, y_high)[0, 1] if feature_series.dtype != 'object' else 0
        corr_popular = np.corrcoef(feature_series, y_popular)[0, 1] if feature_series.dtype != 'object' else 0
        corr_premium = np.corrcoef(feature_series, y_premium)[0, 1] if feature_series.dtype != 'object' else 0
        
        max_corr = max(abs(corr_high), abs(corr_popular), abs(corr_premium))
        
        if max_corr > 0.8:  # High correlation threshold
            suspicious_features.append((feature, max_corr))
            print(f"⚠️  HIGH CORRELATION: {feature} -> {max_corr:.3f}")
    
    # Analyze mutual information
    print(f"\n📊 Mutual Information with Targets:")
    for target_name, y_target in [('High Rated', y_high), ('Popular', y_popular), ('Premium', y_premium)]:
        mi_scores = mutual_info_classif(X, y_target, random_state=RANDOM_STATE)
        mi_df = pd.DataFrame({'feature': feature_names, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        
        print(f"\n🎯 {target_name}:")
        for _, row in mi_df.head(5).iterrows():
            print(f"   {row['feature']}: {row['mi_score']:.4f}")
            
            if row['mi_score'] > 0.5:  # High mutual information
                suspicious_features.append((row['feature'], row['mi_score']))
    
    # Check for features that might be computed from targets
    print(f"\n🔎 Suspicious Features (Potential Leakage):")
    suspicious_features = list(set(suspicious_features))
    for feature, score in sorted(suspicious_features, key=lambda x: x[1], reverse=True):
        print(f"   {feature}: {score:.3f}")
    
    return suspicious_features, X, y_high, y_popular, y_premium, feature_names

def create_safe_features(restaurants_df):
    """Create features without potential data leakage"""
    logger.info("Creating safe features without data leakage...")
    
    df = restaurants_df.copy()
    
    # Remove features that might contain target information
    features_to_remove = [
        'popularity_score',  # Likely computed from rating and rating_count
        'value_score',       # Likely computed from rating and price
        'premium_score',     # Likely computed from price and rating
        'rating_consistency', # Derived from rating_std
        'price_consistency',  # Derived from price_std
    ]
    
    # Base features (safe)
    safe_features = [
        'avg_price', 'dish_count', 'total_rating_count',
        'rating_std', 'price_std', 'category_diversity'
    ]
    
    # Create new safe features
    df['price_to_dish_ratio'] = df['avg_price'] / (df['dish_count'] + 1)
    df['rating_count_per_dish'] = df['total_rating_count'] / (df['dish_count'] + 1)
    df['has_high_variance'] = (df['rating_std'] > df['rating_std'].median()).astype(int)
    
    # One-hot encoding for city (top 10 only to avoid high dimensionality)
    top_cities = df['city'].value_counts().head(10).index
    df['city_safe'] = df['city'].apply(lambda x: x if x in top_cities else 'other')
    city_dummies = pd.get_dummies(df['city_safe'], prefix='city')
    
    # Remove duplicate columns by taking first occurrence
    city_dummies = city_dummies.loc[:, ~city_dummies.columns.duplicated()]
    
    df = pd.concat([df, city_dummies], axis=1)
    
    # Select final safe features
    safe_feature_columns = safe_features + [
        'price_to_dish_ratio', 'rating_count_per_dish', 'has_high_variance'
    ] + city_dummies.columns.tolist()
    
    # Remove any features that might still be problematic
    safe_feature_columns = [f for f in safe_feature_columns if f not in features_to_remove]
    
    # Ensure no duplicate column names
    safe_feature_columns = list(dict.fromkeys(safe_feature_columns))
    
    # Create the safe feature matrix directly from the dataframe
    X_safe = df[safe_feature_columns].copy()
    
    # Handle missing values
    X_safe = X_safe.fillna(X_safe.median())
    
    logger.info(f"Created safe feature matrix: {X_safe.shape}")
    print(f"🔒 Safe features ({len(safe_feature_columns)}): {safe_feature_columns}")
    
    return X_safe, safe_feature_columns

if __name__ == "__main__":
    setup_plotting()
    suspicious_features, X, y_high, y_popular, y_premium, feature_names = analyze_feature_target_relationship()
    
    if suspicious_features:
        print(f"\n🚨 Found {len(suspicious_features)} potentially problematic features")
        print("Creating safe features without data leakage...")
        
        restaurants_df = pd.read_csv(DATA_PROCESSED / "swiggy_restaurants_enhanced.csv")
        X_safe, safe_features = create_safe_features(restaurants_df)
        
        # Save safe features
        X_safe_df = X_safe.copy()  # X_safe is already a DataFrame with proper columns
        X_safe_df.to_csv(DATA_PROCESSED / "model_features_safe.csv", index=False)
        logger.info(f"Safe features saved to: {DATA_PROCESSED / 'model_features_safe.csv'}")
        X_safe_df.to_csv(DATA_PROCESSED / "model_features_safe.csv", index=False)
        logger.info(f"Safe features saved to: {DATA_PROCESSED / 'model_features_safe.csv'}")
    else:
        print("✅ No obvious data leakage detected")