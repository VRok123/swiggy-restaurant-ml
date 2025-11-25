# analyze_exact_features.py
import pandas as pd
import numpy as np
import joblib

def analyze_exact_feature_structure():
    """Analyze the EXACT 30 features the models were trained with"""
    print("🔍 ANALYZING EXACT MODEL FEATURE STRUCTURE")
    print("=" * 60)
    
    # Load the training data that matches the models
    df = pd.read_csv('data/processed/model_features_safe.csv')
    print(f"Training data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # The models expect 30 features, but we have 31 columns
    # Let's find which column to exclude
    print(f"\n📊 Column analysis:")
    for i, col in enumerate(df.columns):
        print(f"  {i:2d}. {col}")
    
    # Check if there's a target column or duplicate we should exclude
    target_like_cols = [col for col in df.columns if 'target' in col.lower() or 'label' in col.lower()]
    if target_like_cols:
        print(f"\n🎯 Possible target columns: {target_like_cols}")
    
    # Load a model to see what it expects
    print(f"\n🔧 Model expectations:")
    model = joblib.load('models/deployment_high_rated_model.pkl')
    print(f"  Expected features: {model.n_features_in_}")
    
    # The key insight: We have 31 columns but models expect 30
    # Let's find which column to drop by checking correlation/duplication
    print(f"\n🔍 Finding the column to exclude:")
    
    # Check for highly correlated or duplicate columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    # Find pairs with very high correlation (>0.95)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j], 
                    correlation_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("  Highly correlated column pairs:")
        for col1, col2, corr in high_corr_pairs:
            print(f"    {col1} <-> {col2}: {corr:.3f}")
    
    # Check for constant columns
    constant_cols = []
    for col in numeric_df.columns:
        if numeric_df[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"  Constant columns: {constant_cols}")
    
    return df.columns.tolist()

def create_correct_30_feature_mapping():
    """Create the EXACT 30-feature mapping the models expect"""
    print(f"\n🎯 CREATING CORRECT 30-FEATURE MAPPING")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/model_features_safe.csv')
    columns = df.columns.tolist()
    
    # Based on analysis, we need to exclude ONE column to get 30 features
    # Let's exclude the last city duplicate or any non-feature column
    
    # The training data has duplicate city columns (city_bengaluru, city_bengaluru.1, etc.)
    # We need to use only one of each city pair
    
    feature_mapping = {}
    used_cities = set()
    feature_index = 0
    
    # First, add all non-city features
    non_city_features = [
        'avg_price', 'dish_count', 'total_rating_count', 'rating_std',
        'price_std', 'category_diversity', 'price_to_dish_ratio', 
        'rating_count_per_dish', 'has_high_variance'
    ]
    
    for feature in non_city_features:
        if feature in columns:
            feature_mapping[feature] = feature_index
            feature_index += 1
            print(f"  {feature}: index {feature_mapping[feature]}")
    
    # Now add city features - but only one of each city (exclude duplicates)
    city_columns = [col for col in columns if col.startswith('city_')]
    
    # Group cities by base name (without .1 suffix)
    city_groups = {}
    for city_col in city_columns:
        base_name = city_col.replace('.1', '')  # Remove .1 suffix
        if base_name not in city_groups:
            city_groups[base_name] = []
        city_groups[base_name].append(city_col)
    
    print(f"\n🏙️ City feature mapping:")
    for base_city, city_variants in city_groups.items():
        # Use the first variant (without .1)
        primary_city = city_variants[0]
        feature_mapping[primary_city] = feature_index
        print(f"  {primary_city}: index {feature_index}")
        feature_index += 1
        
        # Map city names to these indices
        if 'bengaluru' in primary_city:
            feature_mapping['city_bangalore'] = feature_mapping[primary_city]
        elif 'new delhi' in primary_city:
            feature_mapping['city_delhi'] = feature_mapping[primary_city]
    
    print(f"\n✅ Total features mapped: {feature_index}/30")
    return feature_mapping

def test_correct_mapping():
    """Test with the correct 30-feature mapping"""
    print(f"\n🎯 TESTING WITH CORRECT 30-FEATURE MAPPING")
    print("=" * 60)
    
    feature_mapping = create_correct_30_feature_mapping()
    
    class CorrectFeatureMapper:
        def prepare_features(self, features_dict):
            """Prepare exactly 30 features as models expect"""
            feature_vector = [0.0] * 30
            
            # Map basic features
            basic_mapping = {
                'avg_price': 0,
                'dish_count': 1,
                'total_rating_count': 2, 
                'rating_std': 3,
                'price_std': 4,
                'category_diversity': 5,
                'price_to_dish_ratio': 6,
                'rating_count_per_dish': 7,
                'has_high_variance': 8
            }
            
            for feature, index in basic_mapping.items():
                if feature in features_dict:
                    feature_vector[index] = float(features_dict[feature])
            
            # Map city - using the correct city column names from training
            city = features_dict.get('city', 'other')
            if city == 'mumbai':
                feature_vector[9] = 1.0   # city_mumbai
            elif city == 'delhi':
                feature_vector[10] = 1.0  # city_new delhi  
            elif city == 'bangalore':
                feature_vector[11] = 1.0  # city_bengaluru
            elif city == 'chennai':
                feature_vector[12] = 1.0  # city_chennai
            elif city == 'kolkata':
                feature_vector[13] = 1.0  # city_kolkata
            elif city == 'hyderabad':
                feature_vector[14] = 1.0  # city_hyderabad
            elif city == 'pune':
                feature_vector[15] = 1.0  # city_pune (if exists, else other)
            elif city == 'ahmedabad':
                feature_vector[16] = 1.0  # city_ahmedabad
            else:
                feature_vector[17] = 1.0  # city_other
            
            return np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    
    mapper = CorrectFeatureMapper()
    
    # Test cases based on ACTUAL training patterns for YES examples
    test_cases = [
        {
            "name": "ACTUAL High-Rated Pattern",
            "features": {
                "avg_price": 220, "dish_count": 40, "total_rating_count": 2000,
                "avg_rating": 4.5, "median_rating": 4.6, "rating_std": 0.3,
                "price_std": 80, "category_diversity": 8, "price_to_dish_ratio": 5.5,
                "rating_count_per_dish": 50.0, "has_high_variance": 0,
                "price_volatility": 0.36, "city": "mumbai"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔥 {test_case['name']}")
        features = test_case['features']
        
        # Prepare features
        feature_array = mapper.prepare_features(features)
        print(f"Prepared features shape: {feature_array.shape}")
        print(f"Non-zero values: {np.count_nonzero(feature_array)}")
        
        # Test each model
        models = {
            'high_rated': 'models/deployment_high_rated_model.pkl',
            'popular': 'models/deployment_popular_model.pkl', 
            'premium': 'models/deployment_premium_model.pkl'
        }
        
        for model_name, model_path in models.items():
            try:
                model = joblib.load(model_path)
                prediction = model.predict_proba(feature_array)
                prob_yes = prediction[0][1]
                
                result = "✅ YES" if prob_yes > 0.5 else "❌ NO"
                print(f"   {model_name}: {result} (prob: {prob_yes:.1%})")
                
                if prob_yes > 0.5:
                    print(f"   🎉 SUCCESS! Guaranteed YES prediction!")
                    print(f"   Use these features:")
                    for key in ['avg_rating', 'dish_count', 'avg_price', 'total_rating_count']:
                        print(f"     {key}: {features[key]}")
                    
            except Exception as e:
                print(f"   {model_name}: ❌ {e}")

if __name__ == "__main__":
    analyze_exact_feature_structure()
    test_correct_mapping()