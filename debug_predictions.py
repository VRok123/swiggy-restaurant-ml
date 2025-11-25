# debug_predictions.py
import requests
import json
import numpy as np

def debug_predictions():
    """Debug why models are predicting NO"""
    print("🔍 DEBUGGING MODEL PREDICTIONS")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test with different feature combinations
    test_cases = [
        {
            "name": "High-Rated Restaurant",
            "features": {
                "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
                "avg_rating": 4.5, "median_rating": 4.6, "rating_std": 0.1,  # High rating, low variance
                "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
                "rating_count_per_dish": 14.12, "has_high_variance": 0, 
                "price_volatility": 0.33, "city": "mumbai"
            }
        },
        {
            "name": "Popular Restaurant", 
            "features": {
                "avg_price": 350.0, "dish_count": 120, "total_rating_count": 5000,  # High rating count
                "avg_rating": 4.3, "median_rating": 4.4, "rating_std": 0.2,
                "price_std": 100.0, "category_diversity": 12, "price_to_dish_ratio": 2.92,
                "rating_count_per_dish": 41.67, "has_high_variance": 0,
                "price_volatility": 0.29, "city": "delhi"
            }
        },
        {
            "name": "Premium Restaurant",
            "features": {
                "avg_price": 1200.0, "dish_count": 45, "total_rating_count": 800,  # High price
                "avg_rating": 4.4, "median_rating": 4.5, "rating_std": 0.15,
                "price_std": 300.0, "category_diversity": 6, "price_to_dish_ratio": 26.67,
                "rating_count_per_dish": 17.78, "has_high_variance": 0,
                "price_volatility": 0.25, "city": "bangalore"
            }
        },
        {
            "name": "Current UI Settings",
            "features": {
                "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
                "avg_rating": 4.1, "median_rating": 4.2, "rating_std": 0.3,
                "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
                "rating_count_per_dish": 14.12, "has_high_variance": 0,
                "price_volatility": 0.33, "city": "mumbai"
            }
        }
    ]
    
    endpoints = [
        ("/predict/high-rated", "High-Rated (≥4.2)"),
        ("/predict/popular", "Popular (High Rating Count)"), 
        ("/predict/premium", "Premium (High Price)")
    ]
    
    for test_case in test_cases:
        print(f"\n🎯 TEST CASE: {test_case['name']}")
        print("Features:", {k: v for k, v in test_case['features'].items() if k in ['avg_rating', 'total_rating_count', 'avg_price']})
        
        for endpoint, description in endpoints:
            try:
                response = requests.post(base_url + endpoint, json=test_case['features'], timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    prediction = "YES" if data['prediction'] == 1 else "NO"
                    print(f"   {description}: {prediction} ({data['probability']:.1%}) - {data['confidence']} confidence")
                else:
                    print(f"   {description}: ❌ HTTP {response.status_code}")
            except Exception as e:
                print(f"   {description}: ❌ {e}")

def check_model_thresholds():
    """Check what the actual model thresholds should be"""
    print("\n🔧 CHECKING MODEL THRESHOLDS")
    print("=" * 60)
    
    # Based on your original project, these were the thresholds:
    thresholds = {
        "high_rated": "Rating ≥ 4.2",
        "popular": "Rating count ≥ 100", 
        "premium": "Price in top 20% (usually > 400-500 INR)"
    }
    
    for model, threshold in thresholds.items():
        print(f"   {model}: {threshold}")
    
    print("\n💡 If models are predicting NO, try:")
    print("   • Increase avg_rating to 4.3+ for High-Rated")
    print("   • Increase total_rating_count to 2000+ for Popular") 
    print("   • Increase avg_price to 600+ for Premium")

if __name__ == "__main__":
    debug_predictions()
    check_model_thresholds()