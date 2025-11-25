# deploy_to_production.py
import sys
sys.path.append('src')

import requests
import time
import json
from datetime import datetime

class ProductionValidator:
    """Validate production readiness considering actual performance constraints"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
    
    def validate_production_readiness(self):
        """Comprehensive production validation"""
        print("🚀 SWIGGY ML - PRODUCTION VALIDATION")
        print("=" * 70)
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'recommendations': [],
            'production_ready': False
        }
        
        # 1. API Availability Check
        print("\n🔍 1. API AVAILABILITY CHECK")
        api_available = self.check_api_availability()
        validation_results['checks']['api_availability'] = api_available
        print(f"   {'✅' if api_available else '❌'} API is {'available' if api_available else 'unavailable'}")
        
        if not api_available:
            validation_results['recommendations'].append("Start the API server on port 8000")
            self.print_final_summary(validation_results)
            return validation_results
        
        # 2. Model Health Check
        print("\n🔍 2. MODEL HEALTH CHECK")
        model_health = self.check_model_health()
        validation_results['checks']['model_health'] = model_health
        print(f"   {'✅' if model_health else '❌'} Models are {'healthy' if model_health else 'unhealthy'}")
        
        # 3. Performance Validation (with realistic expectations)
        print("\n🔍 3. PERFORMANCE VALIDATION")
        performance = self.validate_performance()
        validation_results['checks']['performance'] = performance
        validation_results['performance_metrics'] = performance
        
        # 4. Feature Processing Check
        print("\n🔍 4. FEATURE PROCESSING CHECK")
        feature_processing = self.check_feature_processing()
        validation_results['checks']['feature_processing'] = feature_processing
        print(f"   {'✅' if feature_processing['success'] else '❌'} Feature processing: {feature_processing['message']}")
        
        # 5. End-to-End Prediction Test
        print("\n🔍 5. END-TO-END PREDICTION TEST")
        e2e_test = self.run_end_to_end_test()
        validation_results['checks']['end_to_end'] = e2e_test
        print(f"   {'✅' if e2e_test['success'] else '❌'} End-to-end test: {e2e_test['message']}")
        
        # Determine production readiness
        all_checks_passed = all([
            api_available,
            model_health,
            performance['acceptable'],
            feature_processing['success'],
            e2e_test['success']
        ])
        
        validation_results['production_ready'] = all_checks_passed
        
        self.print_final_summary(validation_results)
        return validation_results
    
    def check_api_availability(self):
        """Check if API is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def check_model_health(self):
        """Check if all models are loaded and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('models_healthy', False)
            return False
        except:
            return False
    
    def validate_performance(self):
        """Validate performance with realistic expectations for ML systems"""
        print("   ⚡ Testing response times...")
        
        test_features = {
            "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
            "avg_rating": 4.1, "median_rating": 4.2, "rating_std": 0.3,
            "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
            "rating_count_per_dish": 14.12, "has_high_variance": 0, 
            "price_volatility": 0.33, "city": "mumbai"
        }
        
        endpoints = ["/predict/high-rated", "/predict/popular", "/predict/premium"]
        response_times = []
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}{endpoint}", json=test_features, timeout=30)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                status = "✅" if response.status_code == 200 else "❌"
                print(f"     {status} {endpoint}: {response_time:.3f}s")
                
            except Exception as e:
                print(f"     ❌ {endpoint}: Error - {e}")
                response_times.append(10.0)  # High penalty for errors
        
        if not response_times:
            return {
                'acceptable': False,
                'average_response_time': 0,
                'message': 'No successful responses',
                'grade': 'F'
            }
        
        avg_response_time = sum(response_times) / len(response_times)
        
        # Realistic performance grading for ML systems
        if avg_response_time < 1.0:
            grade = 'A'
            acceptable = True
            message = 'Excellent performance'
        elif avg_response_time < 2.0:
            grade = 'B' 
            acceptable = True
            message = 'Good performance for ML system'
        elif avg_response_time < 5.0:
            grade = 'C'
            acceptable = True
            message = 'Acceptable for batch processing'
        else:
            grade = 'D'
            acceptable = False
            message = 'Performance needs improvement'
        
        return {
            'acceptable': acceptable,
            'average_response_time': avg_response_time,
            'message': message,
            'grade': grade,
            'response_times': response_times
        }
    
    def check_feature_processing(self):
        """Check feature processing pipeline"""
        try:
            test_features = {
                "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
                "avg_rating": 4.1, "median_rating": 4.2, "rating_std": 0.3,
                "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
                "rating_count_per_dish": 14.12, "has_high_variance": 0, 
                "price_volatility": 0.33, "city": "mumbai"
            }
            
            response = requests.post(f"{self.base_url}/predict/high-rated", json=test_features, timeout=30)
            
            return {
                'success': response.status_code == 200,
                'message': f"HTTP {response.status_code}",
                'prediction_received': response.status_code == 200
            }
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'prediction_received': False
            }
    
    def run_end_to_end_test(self):
        """Run comprehensive end-to-end test"""
        try:
            # Test multiple scenarios
            test_cases = [
                {"scenario": "Standard Restaurant", "city": "mumbai", "avg_rating": 4.1},
                {"scenario": "Premium Restaurant", "city": "delhi", "avg_rating": 4.5},
                {"scenario": "Budget Restaurant", "city": "bangalore", "avg_rating": 3.8}
            ]
            
            successful_tests = 0
            total_tests = 0
            
            for test_case in test_cases:
                features = {
                    "avg_price": 450.0, "dish_count": 85, "total_rating_count": 1200,
                    "avg_rating": test_case["avg_rating"], "median_rating": 4.2, "rating_std": 0.3,
                    "price_std": 150.0, "category_diversity": 8, "price_to_dish_ratio": 5.29,
                    "rating_count_per_dish": 14.12, "has_high_variance": 0, 
                    "price_volatility": 0.33, "city": test_case["city"]
                }
                
                for endpoint in ["/predict/high-rated", "/predict/popular", "/predict/premium"]:
                    total_tests += 1
                    try:
                        response = requests.post(f"{self.base_url}{endpoint}", json=features, timeout=30)
                        if response.status_code == 200:
                            successful_tests += 1
                    except:
                        pass
            
            success_rate = successful_tests / total_tests if total_tests > 0 else 0
            
            return {
                'success': success_rate >= 0.8,
                'message': f"{successful_tests}/{total_tests} tests passed ({success_rate:.1%})",
                'success_rate': success_rate
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'success_rate': 0
            }
    
    def print_final_summary(self, results):
        """Print final validation summary"""
        print("\n" + "=" * 70)
        print("📊 PRODUCTION VALIDATION SUMMARY")
        print("=" * 70)
        
        checks = results['checks']
        
        print(f"🔍 CHECKS PASSED: {sum(1 for check in checks.values() if (isinstance(check, bool) and check) or (isinstance(check, dict) and check.get('success', False) or check.get('acceptable', False)))}/{len(checks)}")
        
        if 'performance' in checks:
            perf = checks['performance']
            print(f"⚡ PERFORMANCE: {perf['grade']} - {perf['message']}")
            print(f"   Average Response Time: {perf['average_response_time']:.3f}s")
        
        print(f"🎯 PRODUCTION READY: {'✅ YES' if results['production_ready'] else '❌ NO'}")
        
        if results['production_ready']:
            print("\n🎉 CONGRATULATIONS! Your Swiggy ML System is PRODUCTION READY!")
            print("\n🌐 AVAILABLE ENDPOINTS:")
            print("   • http://localhost:8000/docs - API Documentation")
            print("   • http://localhost:8000/health - Health Check")
            print("   • http://localhost:8501 - Streamlit Dashboard")
            print("\n🚀 NEXT STEPS:")
            print("   1. Deploy to cloud platform (AWS/Azure/GCP)")
            print("   2. Set up monitoring and alerting")
            print("   3. Implement API authentication")
            print("   4. Plan model retraining schedule")
        else:
            print("\n📋 RECOMMENDATIONS:")
            for rec in results['recommendations']:
                print(f"   • {rec}")
            
            # Add specific recommendations based on failed checks
            if 'performance' in checks and not checks['performance']['acceptable']:
                print("   • Consider model optimization or hardware upgrade")
            if 'feature_processing' in checks and not checks['feature_processing']['success']:
                print("   • Fix feature processing pipeline")
        
        # Save validation report
        reports_dir = "reports"
        import os
        os.makedirs(reports_dir, exist_ok=True)
        
        report_file = f"{reports_dir}/production_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Validation report saved: {report_file}")

def main():
    """Main production validation function"""
    validator = ProductionValidator()
    results = validator.validate_production_readiness()
    
    return results

if __name__ == "__main__":
    main()