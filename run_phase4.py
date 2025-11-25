# run_phase4.py
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger, setup_plotting, save_dataframe
from config import *

def load_cleaned_data():
    """Load the cleaned datasets"""
    dishes_df = pd.read_csv(DATA_PROCESSED / "swiggy_dishes_clean.csv")
    restaurants_df = pd.read_csv(DATA_PROCESSED / "swiggy_restaurants_clean.csv")
    return dishes_df, restaurants_df

def create_advanced_features(restaurants_df):
    """Create advanced features for modeling"""
    logger.info("Creating advanced features...")
    
    df = restaurants_df.copy()
    
    # 1. Price segments
    df['price_segment'] = pd.cut(df['avg_price'], 
                                bins=[0, 200, 400, 600, float('inf')],
                                labels=['budget', 'mid_range', 'premium', 'luxury'])
    
    # 2. Restaurant size categories
    df['size_category'] = pd.cut(df['dish_count'],
                                bins=[0, 50, 100, 200, float('inf')],
                                labels=['small', 'medium', 'large', 'very_large'])
    
    # 3. Rating consistency score (lower std = more consistent)
    df['rating_consistency'] = 1 / (1 + df['rating_std'].fillna(0))
    
    # 4. Price consistency score
    df['price_consistency'] = 1 / (1 + df['price_std'].fillna(0))
    
    # 5. Value score (rating per unit price)
    df['value_score'] = df['avg_rating'] / (df['avg_price'] + 1)
    
    # 6. Popularity density (ratings per dish)
    df['popularity_density'] = df['total_rating_count'] / (df['dish_count'] + 1)
    
    # 7. Category diversity (number of unique categories)
    df['category_diversity'] = df['categories'].apply(lambda x: len(str(x).split(',')))
    
    # 8. Premium score (combination of price and rating)
    df['premium_score'] = (df['avg_price'] / 100) * df['avg_rating']
    
    logger.info(f"Created {len([col for col in df.columns if col not in restaurants_df.columns])} new features")
    return df

def encode_categorical_features(df):
    """Encode categorical features for modeling"""
    logger.info("Encoding categorical features...")
    
    df_encoded = df.copy()
    
    # Label encode categorical variables
    categorical_columns = ['price_segment', 'size_category']
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
    
    # One-hot encode city (top cities only)
    top_cities = df_encoded['city'].value_counts().head(20).index
    df_encoded['city_top'] = df_encoded['city'].apply(
        lambda x: x if x in top_cities else 'other'
    )
    
    city_dummies = pd.get_dummies(df_encoded['city_top'], prefix='city')
    df_encoded = pd.concat([df_encoded, city_dummies], axis=1)
    
    # Drop the original categorical columns that we don't want in modeling
    columns_to_drop = ['city_top']  # We only need the dummy columns, not this intermediate
    df_encoded = df_encoded.drop(columns=[col for col in columns_to_drop if col in df_encoded.columns])
    
    return df_encoded

def prepare_modeling_data(df):
    """Prepare final dataset for modeling"""
    logger.info("Preparing modeling data...")
    
    # Select features for modeling - ONLY NUMERIC/ENCODED FEATURES
    numeric_features = [
        'avg_price', 'dish_count', 'avg_rating', 'total_rating_count',
        'median_rating', 'rating_std', 'price_std', 'popularity_score',
        'rating_consistency', 'price_consistency', 'value_score',
        'popularity_density', 'category_diversity', 'premium_score'
    ]
    
    # Only include one-hot encoded city columns and encoded segment columns
    categorical_encoded_features = [col for col in df.columns if col.startswith('city_')]
    segment_encoded_features = [col for col in df.columns if '_encoded' in col]
    
    all_features = numeric_features + categorical_encoded_features + segment_encoded_features
    
    # Only include features that exist in the dataframe and are numeric/encoded
    available_features = [f for f in all_features if f in df.columns and df[f].dtype in [np.number, np.int64, np.float64]]
    
    # Create feature matrix - only numeric and properly encoded features
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Target variables
    y_high_rated = df['is_high_rated']
    y_popular = df['is_popular'] 
    y_premium = df['is_premium']
    
    logger.info(f"Final feature matrix: {X.shape}")
    logger.info(f"Available features: {len(available_features)}")
    
    # Print feature summary
    print(f"\n📊 Feature Summary:")
    print(f"   Total features: {len(available_features)}")
    print(f"   Data types: {X.dtypes.value_counts().to_dict()}")
    print(f"   Sample features: {available_features[:8]}")
    
    return X, y_high_rated, y_popular, y_premium, available_features

def create_train_test_splits(X, y_high_rated, y_popular, y_premium):
    """Create train-test splits for all targets"""
    logger.info("Creating train-test splits...")
    
    # Split for high-rated classification
    X_high_train, X_high_test, y_high_train, y_high_test = train_test_split(
        X, y_high_rated, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_high_rated
    )
    
    # Split for popular classification
    X_pop_train, X_pop_test, y_pop_train, y_pop_test = train_test_split(
        X, y_popular, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_popular
    )
    
    # Split for premium classification  
    X_prem_train, X_prem_test, y_prem_train, y_prem_test = train_test_split(
        X, y_premium, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_premium
    )
    
    splits = {
        'high_rated': (X_high_train, X_high_test, y_high_train, y_high_test),
        'popular': (X_pop_train, X_pop_test, y_pop_train, y_pop_test),
        'premium': (X_prem_train, X_prem_test, y_prem_train, y_prem_test)
    }
    
    logger.info("Train-test splits created successfully")
    return splits

def analyze_feature_importance(X, features):
    """Quick analysis of feature correlations"""
    logger.info("Analyzing feature correlations...")
    
    # Make sure X only contains numeric data
    X_numeric = X.select_dtypes(include=[np.number])
    
    if X_numeric.shape[1] < 2:
        logger.warning("Not enough numeric features for correlation analysis")
        return
    
    # Calculate correlation matrix
    correlation_matrix = X_numeric.corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                annot=True, fmt='.2f', annot_kws={'size': 8})
    plt.title('Feature Correlation Matrix (Numeric Features Only)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Top correlated feature pairs
    correlation_pairs = correlation_matrix.unstack().sort_values(key=abs, ascending=False)
    unique_pairs = correlation_pairs[~correlation_pairs.index.duplicated(keep='first')]
    top_correlations = unique_pairs[unique_pairs != 1.0].head(10)
    
    print("\n🔗 Top Feature Correlations:")
    for pair, corr in top_correlations.items():
        print(f"   {pair[0]} - {pair[1]}: {corr:.3f}")
    
    # Feature variance analysis
    print(f"\n📈 Feature Variance Analysis:")
    feature_variance = X_numeric.var().sort_values(ascending=False)
    print(f"   Highest variance: {feature_variance.index[0]} ({feature_variance.iloc[0]:.2f})")
    print(f"   Lowest variance: {feature_variance.index[-1]} ({feature_variance.iloc[-1]:.2f})")

def main():
    print("🚀 Starting Phase 4: Feature Engineering & Modeling Preparation")
    print("=" * 60)
    
    try:
        setup_plotting()
        
        # Load cleaned data
        logger.info("Loading cleaned data...")
        dishes_df, restaurants_df = load_cleaned_data()
        print(f"📊 Restaurants data: {restaurants_df.shape}")
        
        # Create advanced features
        print("\n🛠 Creating Advanced Features...")
        restaurants_enhanced = create_advanced_features(restaurants_df)
        
        # Encode categorical features
        restaurants_encoded = encode_categorical_features(restaurants_enhanced)
        
        # Prepare modeling data
        print("\n📋 Preparing Modeling Data...")
        X, y_high, y_pop, y_prem, features = prepare_modeling_data(restaurants_encoded)
        
        # Create train-test splits
        splits = create_train_test_splits(X, y_high, y_pop, y_prem)
        
        # Analyze features
        print("\n🔍 Analyzing Features...")
        analyze_feature_importance(X, features)
        
        # Save processed data
        save_dataframe(restaurants_encoded, "swiggy_restaurants_enhanced.csv")
        save_dataframe(pd.DataFrame(X, columns=features), "model_features.csv")
        
        print("\n✅ Phase 4 completed successfully!")
        print(f"📁 Generated Files:")
        print(f"   - Enhanced restaurants: data/processed/swiggy_restaurants_enhanced.csv")
        print(f"   - Model features: data/processed/model_features.csv")
        print(f"   - Feature correlation matrix: figures/feature_correlation_matrix.png")
        
        print(f"\n📊 Modeling Ready:")
        print(f"   - Feature matrix: {X.shape}")
        print(f"   - High-rated target: {y_high.sum()} positive, {len(y_high)-y_high.sum()} negative")
        print(f"   - Popular target: {y_pop.sum()} positive, {len(y_pop)-y_pop.sum()} negative")
        print(f"   - Premium target: {y_prem.sum()} positive, {len(y_prem)-y_prem.sum()} negative")
        print(f"   - Available features: {len(features)}")
        
    except Exception as e:
        logger.error(f"❌ Phase 4 failed: {e}")
        raise

if __name__ == "__main__":
    main()