# test_with_training_patterns.py
import pandas as pd
import requests
import os

def find_training_patterns():
    """Find what actual YES examples look like in training data"""
    print("🔍 FINDING ACTUAL YES PATTERNS FROM TRAINING DATA")
    print("=" * 60)
    
    # Look for files that might contain the actual targets
    files_to_check = [
        'data/processed/model_features.csv',
        'data/processed/model_features_safe.csv',
        'data/processed/swiggy_restaurants_enhanced.csv'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"\n📊 Analyzing: {file}")
                
                # Look for any binary target columns
                binary_cols = []
                for col in df.columns:
                    if df[col].dtype in [int, bool] and df[col].nunique() == 2:
                        binary_cols.append(col)
                
                if binary_cols:
                    print(f"Found binary columns: {binary_cols}")
                    
                    for target_col in binary_cols:
                        yes_examples = df[df[target_col] == 1]
                        no_examples = df[df[target_col] == 0]
                        
                        print(f"\n🎯 Analysis for '{target_col}':")
                        print(f"   YES examples: {len(yes_examples)}")
                        print(f"   NO examples: {len(no_examples)}")
                        
                        if len(yes_examples) > 0:
                            print(f"   AVERAGE FEATURES FOR YES EXAMPLES:")
                            numeric_cols = yes_examples.select_dtypes(include=['number']).columns
                            for col in numeric_cols[:6]:  # Show first 6 numeric columns
                                if col != target_col:
                                    yes_avg = yes_examples[col].mean()
                                    no_avg = no_examples[col].mean()
                                    print(f"     {col}: YES={yes_avg:.2f}, NO={no_avg:.2f}")
            
            except Exception as e:
                print(f"Error analyzing {file}: {e}")

def test_with_real_patterns():
    """Test with patterns found in actual YES examples"""
    print("\n🎯 TESTING WITH REAL TRAINING PATTERNS")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Based on common patterns in restaurant data
    test_cases = [
        {
            "name": "Based on Common High-Rated Pattern",
            "features": {
                "avg_price": 350, "dish_count": 45, "total_rating_count": 2500,
                "avg_rating": 4.6, "median_rating": 4.7, "rating_std": 0.15,
                "price_std": 80, "category_diversity": 9, "price_to_dish_ratio": 7.78,
                "rating_count_per_dish": 55.56, "has_high_variance": 0,
                "price_volatility": 0.23, "city": "mumbai"
            }
        }
    ]
    
    endpoints = [
        ("/predict/high-rated", "High-Rated"),
        ("/predict/popular", "Popular"), 
        ("/predict/premium", "Premium")
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        for endpoint, description in endpoints:
            try:
                response = requests.post(base_url + endpoint, json=test_case['features'], timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    prediction = "✅ YES" if data['prediction'] == 1 else "❌ NO"
                    print(f"   {description}: {prediction} (prob: {data['probability']:.1%})")
                else:
                    print(f"   {description}: ❌ HTTP {response.status_code}")
            except Exception as e:
                print(f"   {description}: ❌ {e}")

if __name__ == "__main__":
    find_training_patterns()
    test_with_real_patterns()