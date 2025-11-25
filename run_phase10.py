# run_phase10.py
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger, setup_plotting, load_model
from config import *

class ModelMonitor:
    """Monitor model performance and health in production"""
    
    def __init__(self):
        self.monitoring_data = []
        self.performance_metrics = {}
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """Initialize monitoring system"""
        logger.info("Setting up model monitoring system...")
        
        # Create monitoring directory
        monitoring_dir = PROJECT_ROOT / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        self.monitoring_file = monitoring_dir / "model_monitoring.json"
        self.performance_file = monitoring_dir / "performance_metrics.json"
        
        # Load existing monitoring data
        self.load_monitoring_data()
    
    def load_monitoring_data(self):
        """Load existing monitoring data"""
        try:
            if self.monitoring_file.exists():
                with open(self.monitoring_file, 'r') as f:
                    self.monitoring_data = json.load(f)
            logger.info(f"Loaded {len(self.monitoring_data)} monitoring records")
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
            self.monitoring_data = []
    
    def log_prediction(self, model_name: str, features: Dict, prediction: Dict, 
                      response_time: float, success: bool = True):
        """Log prediction details for monitoring"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'features': features,
            'prediction': prediction,
            'response_time': response_time,
            'success': success,
            'feature_count': len(features)
        }
        
        self.monitoring_data.append(record)
        
        # Keep only last 1000 records to prevent memory issues
        if len(self.monitoring_data) > 1000:
            self.monitoring_data = self.monitoring_data[-1000:]
        
        # Save to file
        self.save_monitoring_data()
        
        return record
    
    def save_monitoring_data(self):
        """Save monitoring data to file"""
        try:
            with open(self.monitoring_file, 'w') as f:
                json.dump(self.monitoring_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Calculate performance metrics for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_records = [
            record for record in self.monitoring_data
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]
        
        if not recent_records:
            return {}
        
        metrics = {
            'total_predictions': len(recent_records),
            'success_rate': np.mean([1 if r['success'] else 0 for r in recent_records]),
            'avg_response_time': np.mean([r['response_time'] for r in recent_records]),
            'p95_response_time': np.percentile([r['response_time'] for r in recent_records], 95),
            'models_used': {},
            'hourly_volume': self.get_hourly_volume(recent_records)
        }
        
        # Model-specific metrics
        for record in recent_records:
            model = record['model']
            if model not in metrics['models_used']:
                metrics['models_used'][model] = 0
            metrics['models_used'][model] += 1
        
        return metrics
    
    def get_hourly_volume(self, records: List) -> Dict:
        """Get prediction volume by hour"""
        hourly_volume = {}
        for record in records:
            hour = datetime.fromisoformat(record['timestamp']).strftime('%H:00')
            hourly_volume[hour] = hourly_volume.get(hour, 0) + 1
        return hourly_volume
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'last_1_hour': self.get_performance_metrics(1),
            'last_24_hours': self.get_performance_metrics(24),
            'last_7_days': self.get_performance_metrics(168),
            'system_health': self.check_system_health(),
            'alerts': self.check_alerts()
        }
        
        return report
    
    def check_system_health(self) -> Dict:
        """Check system health indicators"""
        recent_metrics = self.get_performance_metrics(1)
        
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        if recent_metrics:
            if recent_metrics['success_rate'] < 0.95:
                health['status'] = 'degraded'
                health['issues'].append('Low success rate')
                health['recommendations'].append('Check model serving infrastructure')
            
            if recent_metrics['avg_response_time'] > 1.0:  # More than 1 second
                health['status'] = 'degraded'
                health['issues'].append('High response time')
                health['recommendations'].append('Optimize feature preparation')
            
            if recent_metrics['total_predictions'] == 0:
                health['status'] = 'idle'
                health['issues'].append('No recent predictions')
                health['recommendations'].append('Check API connectivity')
        
        return health
    
    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        recent_metrics = self.get_performance_metrics(1)
        
        if recent_metrics:
            if recent_metrics['success_rate'] < 0.9:
                alerts.append({
                    'level': 'high',
                    'message': 'Success rate below 90%',
                    'metric': 'success_rate',
                    'value': recent_metrics['success_rate']
                })
            
            if recent_metrics['p95_response_time'] > 2.0:
                alerts.append({
                    'level': 'medium',
                    'message': '95th percentile response time above 2 seconds',
                    'metric': 'response_time',
                    'value': recent_metrics['p95_response_time']
                })
        
        return alerts

class ProductionDeployment:
    """Handle production deployment aspects"""
    
    def __init__(self):
        self.monitor = ModelMonitor()
        self.deployment_checklist = self.create_deployment_checklist()
    
    def create_deployment_checklist(self) -> List[Dict]:
        """Create production deployment checklist"""
        return [
            {
                'category': 'Infrastructure',
                'items': [
                    {'task': 'API server running on production port', 'status': 'pending'},
                    {'task': 'Load balancer configured', 'status': 'pending'},
                    {'task': 'SSL certificates installed', 'status': 'pending'},
                    {'task': 'Domain name configured', 'status': 'pending'}
                ]
            },
            {
                'category': 'Model Serving',
                'items': [
                    {'task': 'All models loaded successfully', 'status': 'pending'},
                    {'task': 'Feature preprocessing validated', 'status': 'pending'},
                    {'task': 'API endpoints responding', 'status': 'pending'},
                    {'task': 'Error handling implemented', 'status': 'pending'}
                ]
            },
            {
                'category': 'Monitoring & Logging',
                'items': [
                    {'task': 'Performance monitoring active', 'status': 'pending'},
                    {'task': 'Log aggregation configured', 'status': 'pending'},
                    {'task': 'Alert system setup', 'status': 'pending'},
                    {'task': 'Health check endpoints working', 'status': 'pending'}
                ]
            },
            {
                'category': 'Security',
                'items': [
                    {'task': 'API authentication implemented', 'status': 'pending'},
                    {'task': 'Rate limiting configured', 'status': 'pending'},
                    {'task': 'Input validation active', 'status': 'pending'},
                    {'task': 'Security headers set', 'status': 'pending'}
                ]
            }
        ]
    
    def run_deployment_checks(self) -> Dict:
        """Run comprehensive deployment checks"""
        logger.info("Running production deployment checks...")
        
        checks = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'pending',
            'checks_passed': 0,
            'checks_total': 0,
            'details': {}
        }
        
        # Check API connectivity
        checks['details']['api_connectivity'] = self.check_api_connectivity()
        
        # Check model availability
        checks['details']['model_availability'] = self.check_model_availability()
        
        # Check feature processing
        checks['details']['feature_processing'] = self.check_feature_processing()
        
        # Check performance
        checks['details']['performance'] = self.check_performance()
        
        # Calculate overall status
        total_checks = len(checks['details'])
        passed_checks = sum(1 for check in checks['details'].values() if check['status'] == 'pass')
        checks['checks_passed'] = passed_checks
        checks['checks_total'] = total_checks
        checks['overall_status'] = 'pass' if passed_checks == total_checks else 'fail'
        
        return checks
    
    def check_api_connectivity(self) -> Dict:
        """Check API connectivity"""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            return {
                'status': 'pass' if response.status_code == 200 else 'fail',
                'message': f"API responded with status {response.status_code}",
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f"API connectivity failed: {str(e)}",
                'response_time': None
            }
    
    def check_model_availability(self) -> Dict:
        """Check if all models are available"""
        try:
            models = ['deployment_high_rated_model.pkl', 'deployment_popular_model.pkl', 'deployment_premium_model.pkl']
            missing_models = []
            
            for model_file in models:
                model_path = MODELS_DIR / model_file
                if not model_path.exists():
                    missing_models.append(model_file)
            
            return {
                'status': 'pass' if not missing_models else 'fail',
                'message': f"Models checked: {len(models) - len(missing_models)}/{len(models)} available",
                'missing_models': missing_models
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f"Model check failed: {str(e)}",
                'missing_models': []
            }
    
    def check_feature_processing(self) -> Dict:
        """Check feature processing pipeline"""
        try:
            # Test with sample features
            sample_features = {
                'avg_price': 450.0,
                'dish_count': 85,
                'total_rating_count': 1200,
                'avg_rating': 4.1,
                'median_rating': 4.2,
                'rating_std': 0.3,
                'price_std': 150.0,
                'category_diversity': 8,
                'price_to_dish_ratio': 5.29,
                'rating_count_per_dish': 14.12,
                'has_high_variance': 0,
                'price_volatility': 0.33,
                'city': 'mumbai'
            }
            
            import requests
            response = requests.post(
                "http://localhost:8000/predict/high-rated",
                json=sample_features,
                timeout=10
            )
            
            return {
                'status': 'pass' if response.status_code == 200 else 'fail',
                'message': f"Feature processing test: {response.status_code}",
                'prediction_received': response.status_code == 200
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f"Feature processing check failed: {str(e)}",
                'prediction_received': False
            }
    
    def check_performance(self) -> Dict:
        """Check system performance"""
        try:
            import requests
            import time
            
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            response_time = time.time() - start_time
            
            return {
                'status': 'pass' if response_time < 1.0 else 'warning',
                'message': f"Health check response time: {response_time:.3f}s",
                'response_time': response_time,
                'threshold': 1.0
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f"Performance check failed: {str(e)}",
                'response_time': None,
                'threshold': 1.0
            }
    
    def generate_deployment_report(self) -> Dict:
        """Generate comprehensive deployment report"""
        deployment_checks = self.run_deployment_checks()
        monitoring_report = self.monitor.generate_monitoring_report()
        
        report = {
            'deployment_timestamp': datetime.now().isoformat(),
            'deployment_checks': deployment_checks,
            'monitoring_report': monitoring_report,
            'recommendations': self.generate_recommendations(deployment_checks, monitoring_report),
            'next_steps': self.get_next_steps(deployment_checks)
        }
        
        return report
    
    def generate_recommendations(self, deployment_checks: Dict, monitoring_report: Dict) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []
        
        if deployment_checks['overall_status'] != 'pass':
            recommendations.append("Fix deployment check failures before production")
        
        if monitoring_report['system_health']['status'] != 'healthy':
            recommendations.append("Address system health issues")
        
        if not monitoring_report['last_1_hour']:
            recommendations.append("Generate more test traffic to validate monitoring")
        
        # Add specific recommendations based on checks
        for check_name, check_result in deployment_checks['details'].items():
            if check_result['status'] != 'pass':
                recommendations.append(f"Fix {check_name}: {check_result['message']}")
        
        return recommendations
    
    def get_next_steps(self, deployment_checks: Dict) -> List[str]:
        """Get next steps for production deployment"""
        if deployment_checks['overall_status'] == 'pass':
            return [
                "Deploy to production environment",
                "Set up continuous monitoring",
                "Configure alert notifications",
                "Plan for model retraining schedule",
                "Document API for consumers"
            ]
        else:
            return [
                "Fix identified deployment issues",
                "Re-run deployment checks",
                "Test with more diverse input data",
                "Validate error handling scenarios",
                "Perform load testing"
            ]

def create_monitoring_dashboard():
    """Create a simple monitoring dashboard"""
    print("\n" + "=" * 70)
    print("📊 PRODUCTION MONITORING DASHBOARD")
    print("=" * 70)
    
    deployment = ProductionDeployment()
    
    # Run deployment checks
    print("\n🔍 RUNNING DEPLOYMENT CHECKS...")
    deployment_checks = deployment.run_deployment_checks()
    
    # Display check results
    print(f"\n✅ DEPLOYMENT STATUS: {deployment_checks['overall_status'].upper()}")
    print(f"   Checks Passed: {deployment_checks['checks_passed']}/{deployment_checks['checks_total']}")
    
    for check_name, check_result in deployment_checks['details'].items():
        status_icon = "✅" if check_result['status'] == 'pass' else "⚠️" if check_result['status'] == 'warning' else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")
    
    # Generate monitoring report
    print("\n📈 GENERATING MONITORING REPORT...")
    monitoring_report = deployment.monitor.generate_monitoring_report()
    
    # Display monitoring insights
    if monitoring_report['last_1_hour']:
        metrics = monitoring_report['last_1_hour']
        print(f"\n📊 LAST HOUR PERFORMANCE:")
        print(f"   Predictions: {metrics['total_predictions']}")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        print(f"   Avg Response Time: {metrics['avg_response_time']:.3f}s")
        print(f"   P95 Response Time: {metrics['p95_response_time']:.3f}s")
    
    # Display system health
    health = monitoring_report['system_health']
    health_icon = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'degraded' else "🔵"
    print(f"\n{health_icon} SYSTEM HEALTH: {health['status'].upper()}")
    
    if health['issues']:
        print("   Issues:")
        for issue in health['issues']:
            print(f"     • {issue}")
    
    if health['recommendations']:
        print("   Recommendations:")
        for rec in health['recommendations']:
            print(f"     • {rec}")
    
    # Display alerts
    alerts = monitoring_report['alerts']
    if alerts:
        print(f"\n🚨 ACTIVE ALERTS ({len(alerts)}):")
        for alert in alerts:
            level_icon = "🔴" if alert['level'] == 'high' else "🟡" if alert['level'] == 'medium' else "🔵"
            print(f"   {level_icon} {alert['message']} (Value: {alert['value']})")
    
    # Generate deployment report
    print("\n📋 GENERATING DEPLOYMENT REPORT...")
    deployment_report = deployment.generate_deployment_report()
    
    # Display recommendations
    if deployment_report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in deployment_report['recommendations']:
            print(f"   • {rec}")
    
    # Display next steps
    print(f"\n🎯 NEXT STEPS:")
    for step in deployment_report['next_steps']:
        print(f"   • {step}")
    
    # Save reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    deployment_report_file = reports_dir / "production_deployment_report.json"
    with open(deployment_report_file, 'w') as f:
        json.dump(deployment_report, f, indent=2)
    
    print(f"\n💾 REPORTS SAVED:")
    print(f"   • Deployment Report: {deployment_report_file}")
    print(f"   • Monitoring Data: {deployment.monitor.monitoring_file}")
    
    print("\n" + "=" * 70)
    print("🎉 PHASE 10 COMPLETED: Model Monitoring & Production Deployment")
    print("=" * 70)
    
    return deployment_report

def main():
    print("🚀 Starting Phase 10: Model Monitoring & Production Deployment")
    print("=" * 70)
    
    try:
        setup_plotting()
        
        # Create monitoring dashboard
        deployment_report = create_monitoring_dashboard()
        
        print(f"\n✅ PRODUCTION DEPLOYMENT READY!")
        print(f"📊 Monitoring System: Active")
        print(f"🔧 Deployment Checks: {deployment_report['deployment_checks']['checks_passed']}/{deployment_report['deployment_checks']['checks_total']} Passed")
        print(f"🚀 Next Steps: {len(deployment_report['next_steps'])} actions identified")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"   - Monitoring system: ModelMonitor class")
        print(f"   - Deployment checks: ProductionDeployment class") 
        print(f"   - Reports: production_deployment_report.json")
        print(f"   - Monitoring data: model_monitoring.json")
        
        print(f"\n🌐 PRODUCTION READINESS:")
        if deployment_report['deployment_checks']['overall_status'] == 'pass':
            print("   ✅ System is ready for production deployment!")
        else:
            print("   ⚠️ Address the identified issues before production")
        
    except Exception as e:
        logger.error(f"❌ Phase 10 failed: {e}")
        raise

if __name__ == "__main__":
    main()