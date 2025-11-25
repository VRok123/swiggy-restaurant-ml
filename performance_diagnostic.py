# performance_diagnostic.py
import time
import requests
import numpy as np

def diagnose_performance():
    """Diagnose where the performance bottleneck is"""
    print("🔍 PERFORMANCE DIAGNOSTIC")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_features = {
        "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
        "avg_rating": 4.1, "median_rating": 4.2, "rating_std": 0.3,
        "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
        "rating_count_per_dish": 14.12, "has_high_variance": 0, 
        "price_volatility": 0.33, "city": "mumbai"
    }
    
    # Test each endpoint with detailed timing
    endpoints = ["/predict/high-rated", "/predict/popular", "/predict/premium"]
    
    for endpoint in endpoints:
        print(f"\n📊 Testing {endpoint}:")
        
        total_start = time.time()
        
        try:
            response = requests.post(base_url + endpoint, json=test_features, timeout=30)
            total_time = time.time() - total_start
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Total Response: {total_time:.3f}s")
                print(f"   📊 Prediction: {data.get('prediction')}")
                print(f"   🎯 Probability: {data.get('probability')}")
                print(f"   ⚡ API Response Time: {data.get('response_time', 'N/A')}")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")

def test_direct_model_performance():
    """Test model performance directly without API overhead"""
    print("\n🔧 DIRECT MODEL PERFORMANCE TEST")
    print("=" * 60)
    
    try:
        from run_phase8_optimized import OptimizedDeploymentManager
        import joblib
        
        # Load manager directly
        manager = OptimizedDeploymentManager()
        manager._create_optimized_feature_mapping()
        
        # Load models directly to measure load time
        print("📦 Loading models directly...")
        load_start = time.time()
        
        models = {}
        models['high_rated'] = joblib.load('models/deployment_high_rated_model.pkl')
        models['popular'] = joblib.load('models/deployment_popular_model.pkl')
        models['premium'] = joblib.load('models/deployment_premium_model.pkl')
        
        load_time = time.time() - load_start
        print(f"   Model load time: {load_time:.3f}s")
        
        # Test feature preparation
        test_features = {
            "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
            "avg_rating": 4.1, "median_rating": 4.2, "rating_std": 0.3,
            "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
            "rating_count_per_dish": 14.12, "has_high_variance": 0, 
            "price_volatility": 0.33, "city": "mumbai"
        }
        
        print("⚡ Testing feature preparation...")
        feature_start = time.time()
        feature_array = manager.prepare_features_optimized(test_features)
        feature_time = time.time() - feature_start
        print(f"   Feature prep time: {feature_time:.3f}s")
        
        # Test prediction times for each model
        for model_name, model in models.items():
            print(f"🎯 Testing {model_name} prediction...")
            
            # First prediction (cold start)
            first_start = time.time()
            proba1 = model.predict_proba(feature_array)
            first_time = time.time() - first_start
            
            # Second prediction (warm)
            second_start = time.time()
            proba2 = model.predict_proba(feature_array)
            second_time = time.time() - second_start
            
            print(f"   First prediction: {first_time:.3f}s")
            print(f"   Second prediction: {second_time:.3f}s")
            print(f"   Probability: {proba1[0, 1]:.4f}")
            
    except Exception as e:
        print(f"   ❌ Direct test failed: {e}")

if __name__ == "__main__":
    diagnose_performance()
    test_direct_model_performance()