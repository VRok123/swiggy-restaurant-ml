# verify_production.py
import requests
import time
import json
from datetime import datetime

def run_performance_tests():
    """Run comprehensive performance tests"""
    print("🚀 RUNNING PRODUCTION PERFORMANCE TESTS")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_cases = [
        {
            "name": "Standard Restaurant",
            "data": {
                "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
                "avg_rating": 4.1, "median_rating": 4.2, "rating_std": 0.3,
                "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
                "rating_count_per_dish": 14.12, "has_high_variance": 0, 
                "price_volatility": 0.33, "city": "mumbai"
            }
        },
        {
            "name": "Premium Restaurant", 
            "data": {
                "avg_price": 800.0, "dish_count": 45, "total_rating_count": 500,
                "avg_rating": 4.5, "median_rating": 4.6, "rating_std": 0.2,
                "price_std": 200.0, "category_diversity": 6, "price_to_dish_ratio": 17.78,
                "rating_count_per_dish": 11.11, "has_high_variance": 0,
                "price_volatility": 0.25, "city": "delhi"
            }
        }
    ]
    
    endpoints = [
        "/predict/high-rated",
        "/predict/popular", 
        "/predict/premium"
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        
        for endpoint in endpoints:
            url = base_url + endpoint
            
            # Measure response time
            start_time = time.time()
            try:
                response = requests.post(url, json=test_case['data'], timeout=10)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    status = "✅ SUCCESS"
                else:
                    result_data = {}
                    status = "❌ FAILED"
                    
                results.append({
                    "test_case": test_case['name'],
                    "endpoint": endpoint,
                    "status": status,
                    "response_time": round(response_time, 3),
                    "http_status": response.status_code,
                    "prediction": result_data.get('prediction', 'N/A'),
                    "probability": result_data.get('probability', 'N/A')
                })
                
                print(f"   {status} {endpoint}: {response_time:.3f}s")
                
            except Exception as e:
                response_time = time.time() - start_time
                results.append({
                    "test_case": test_case['name'],
                    "endpoint": endpoint, 
                    "status": "❌ ERROR",
                    "response_time": round(response_time, 3),
                    "error": str(e)
                })
                print(f"   ❌ ERROR {endpoint}: {e}")
    
    # Calculate statistics
    successful_tests = [r for r in results if r['status'] == '✅ SUCCESS']
    if successful_tests:
        avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
        max_response_time = max(r['response_time'] for r in successful_tests)
        min_response_time = min(r['response_time'] for r in successful_tests)
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Tests Completed: {len(successful_tests)}/{len(results)}")
        print(f"   Average Response Time: {avg_response_time:.3f}s")
        print(f"   Best Response Time: {min_response_time:.3f}s")
        print(f"   Worst Response Time: {max_response_time:.3f}s")
        
        # Performance grading
        if avg_response_time < 0.5:
            grade = "🟢 EXCELLENT"
        elif avg_response_time < 1.0:
            grade = "🟡 GOOD" 
        elif avg_response_time < 2.0:
            grade = "🟠 ACCEPTABLE"
        else:
            grade = "🔴 POOR"
            
        print(f"   Performance Grade: {grade}")
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "performance_tests": results,
        "summary": {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(results) if results else 0,
            "avg_response_time": avg_response_time if successful_tests else 0,
            "performance_grade": grade if successful_tests else "UNKNOWN"
        }
    }
    
    with open("reports/performance_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Performance report saved: reports/performance_validation_report.json")
    
    return report

if __name__ == "__main__":
    run_performance_tests()