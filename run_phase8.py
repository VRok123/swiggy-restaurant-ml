# run_phase8.py
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from utils import logger, setup_plotting, load_model
from config import *

class DeploymentManager:
    def __init__(self):
        self.models = {}
        self.feature_metadata = {}

    def load_deployment_models(self):
        """Load deployment models and feature metadata"""
        logger.info("Loading deployment models...")
        
        try:
            # Load deployment models
            self.models['high_rated'] = load_model('deployment_high_rated_model.pkl')
            self.models['popular'] = load_model('deployment_popular_model.pkl')
            self.models['premium'] = load_model('deployment_premium_model.pkl')
            
            # Load feature metadata
            metadata_file = MODELS_DIR / "deployment_feature_metadata.pkl"
            self.feature_metadata = joblib.load(metadata_file)
            
            logger.info("✅ All deployment models loaded successfully")
            print(f"📦 Loaded models: {list(self.models.keys())}")
            print(f"📋 Feature sets: {list(self.feature_metadata.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load deployment models: {e}")
            raise
    
    def create_fastapi_app(self):
        """Create FastAPI application with prediction endpoints"""
        logger.info("Creating FastAPI application...")
        
        app = FastAPI(
            title="Swiggy Restaurant Prediction API",
            description="Machine Learning API for predicting restaurant characteristics",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Pydantic models for request/response
        class RestaurantFeatures(BaseModel):
            avg_price: float
            dish_count: int
            total_rating_count: int
            avg_rating: float
            median_rating: float
            rating_std: float
            price_std: float
            category_diversity: int
            price_to_dish_ratio: float
            rating_count_per_dish: float
            has_high_variance: int
            price_volatility: float
            city: str
        
        class PredictionResponse(BaseModel):
            prediction_task: str
            prediction: int
            probability: float
            confidence: str
        
        class BatchPredictionResponse(BaseModel):
            predictions: List[Dict[str, Any]]
        
        # Helper function to prepare features for each model
        def prepare_features(features: RestaurantFeatures, task: str):
            """Prepare features for specific prediction task"""
            base_features = {
                'avg_price': features.avg_price,
                'dish_count': features.dish_count,
                'total_rating_count': features.total_rating_count,
                'avg_rating': features.avg_rating,
                'median_rating': features.median_rating,
                'rating_std': features.rating_std,
                'price_std': features.price_std,
                'category_diversity': features.category_diversity,
                'price_to_dish_ratio': features.price_to_dish_ratio,
                'rating_count_per_dish': features.rating_count_per_dish,
                'has_high_variance': features.has_high_variance,
                'price_volatility': features.price_volatility
            }
            
            # Add city encoding
            city_features = {f'city_{features.city}': 1}
            for city_feature in self.feature_metadata.get(f'{task}_features', []):
                if city_feature.startswith('city_') and city_feature != f'city_{features.city}':
                    city_features[city_feature] = 0
            
            # Combine all features
            all_features = {**base_features, **city_features}
            
            # Select only the features needed for this task
            required_features = self.feature_metadata.get(f'{task}_features', [])
            prepared_features = {feature: all_features.get(feature, 0) for feature in required_features}
            
            return prepared_features
        
        # Health check endpoint
        @app.get("/")
        async def root():
            return {
                "message": "Swiggy Restaurant Prediction API",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "predict_high_rated": "/predict/high-rated",
                    "predict_popular": "/predict/popular", 
                    "predict_premium": "/predict/premium",
                    "batch_predict": "/predict/batch"
                }
            }
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "models_loaded": list(self.models.keys()),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        @app.post("/predict/high-rated", response_model=PredictionResponse)
        async def predict_high_rated(features: RestaurantFeatures):
            """Predict if a restaurant is high-rated (rating >= 4.2)"""
            try:
                # Prepare features
                prepared_features = prepare_features(features, 'high_rated')
                feature_array = np.array([list(prepared_features.values())]).reshape(1, -1)
                
                # Make prediction
                model = self.models['high_rated']
                probability = model.predict_proba(feature_array)[0, 1]
                prediction = int(probability >= 0.5)
                
                # Determine confidence level
                if probability >= 0.8:
                    confidence = "high"
                elif probability >= 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                return PredictionResponse(
                    prediction_task="high_rated",
                    prediction=prediction,
                    probability=round(probability, 4),
                    confidence=confidence
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/popular", response_model=PredictionResponse)
        async def predict_popular(features: RestaurantFeatures):
            """Predict if a restaurant is popular (rating_count >= 100)"""
            try:
                # Prepare features
                prepared_features = prepare_features(features, 'popular')
                feature_array = np.array([list(prepared_features.values())]).reshape(1, -1)
                
                # Make prediction
                model = self.models['popular']
                probability = model.predict_proba(feature_array)[0, 1]
                prediction = int(probability >= 0.5)
                
                # Determine confidence level
                if probability >= 0.9:
                    confidence = "high"
                elif probability >= 0.7:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                return PredictionResponse(
                    prediction_task="popular",
                    prediction=prediction,
                    probability=round(probability, 4),
                    confidence=confidence
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/premium", response_model=PredictionResponse)
        async def predict_premium(features: RestaurantFeatures):
            """Predict if a restaurant is premium (price in top 20%)"""
            try:
                # Prepare features
                prepared_features = prepare_features(features, 'premium')
                feature_array = np.array([list(prepared_features.values())]).reshape(1, -1)
                
                # Make prediction
                model = self.models['premium']
                probability = model.predict_proba(feature_array)[0, 1]
                prediction = int(probability >= 0.5)
                
                # Determine confidence level
                if probability >= 0.8:
                    confidence = "high"
                elif probability >= 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                return PredictionResponse(
                    prediction_task="premium",
                    prediction=prediction,
                    probability=round(probability, 4),
                    confidence=confidence
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def batch_predict(features_list: List[RestaurantFeatures]):
            """Batch prediction for multiple restaurants"""
            try:
                predictions = []
                
                for i, features in enumerate(features_list):
                    # Get predictions for all tasks
                    high_rated_pred = await predict_high_rated(features)
                    popular_pred = await predict_popular(features)
                    premium_pred = await predict_premium(features)
                    
                    predictions.append({
                        "restaurant_id": i,
                        "high_rated": {
                            "prediction": high_rated_pred.prediction,
                            "probability": high_rated_pred.probability,
                            "confidence": high_rated_pred.confidence
                        },
                        "popular": {
                            "prediction": popular_pred.prediction,
                            "probability": popular_pred.probability,
                            "confidence": popular_pred.confidence
                        },
                        "premium": {
                            "prediction": premium_pred.prediction,
                            "probability": premium_pred.probability,
                            "confidence": premium_pred.confidence
                        }
                    })
                
                return BatchPredictionResponse(predictions=predictions)
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/model-info")
        async def get_model_info():
            """Get information about loaded models"""
            model_info = {}
            for task, model in self.models.items():
                model_info[task] = {
                    "model_type": type(model).__name__,
                    "n_features": len(self.feature_metadata.get(f'{task}_features', [])),
                    "features": self.feature_metadata.get(f'{task}_features', [])[:5]  # First 5 features
                }
            return model_info
        
        return app

def main():
    print("🚀 Starting Phase 8: FastAPI Deployment")
    print("=" * 60)
    
    try:
        setup_plotting()
        
        # Initialize deployment manager
        manager = DeploymentManager()
        
        # Load deployment models
        print("\n📦 Loading Deployment Models...")
        manager.load_deployment_models()
        
        # Create FastAPI app
        print("\n🔧 Creating FastAPI Application...")
        app = manager.create_fastapi_app()
        
        print("\n✅ PHASE 8 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n🎯 FASTAPI DEPLOYMENT READY!")
        print(f"   To start the API server, run:")
        print(f"   uvicorn run_phase8:app --reload --host 0.0.0.0 --port 8000")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"   - API code: run_phase8.py")
        
        print(f"\n🌐 API ENDPOINTS:")
        print(f"   - Health check: GET /health")
        print(f"   - High-rated prediction: POST /predict/high-rated")
        print(f"   - Popular prediction: POST /predict/popular")
        print(f"   - Premium prediction: POST /predict/premium")
        print(f"   - Batch prediction: POST /predict/batch")
        print(f"   - Interactive docs: GET /docs")
        
        print(f"\n🚀 START THE API:")
        print(f"   Open a new terminal and run:")
        print(f"   cd swiggy-ml")
        print(f"   python -m uvicorn run_phase8:app --reload --host 0.0.0.0 --port 8000")
        
        print(f"\n📖 ACCESS DOCUMENTATION:")
        print(f"   Once running, visit: http://localhost:8000/docs")
        
        return app
        
    except Exception as e:
        logger.error(f"❌ Phase 8 failed: {e}")
        raise

# Create the app instance for uvicorn
manager = DeploymentManager()
manager.load_deployment_models()
app = manager.create_fastapi_app()

if __name__ == "__main__":
    main()