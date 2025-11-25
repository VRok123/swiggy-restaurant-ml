# run_phase9.py
import sys
sys.path.append('src')

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger, setup_plotting, load_model
from config import *

class StreamlitDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Swiggy Restaurant Analytics",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-card {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            background-color: #ffffff;
            border-left: 4px solid #FF6B6B;
        }
        .ml-success {
            border-left: 4px solid #28a745;
        }
        .ml-warning {
            border-left: 4px solid #ffc107;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def check_api_health(self):
        """Check if the FastAPI is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, None
        except:
            return False, None
    
    def make_ml_prediction(self, features, prediction_type):
        """Make ML prediction using the API"""
        try:
            response = requests.post(
                f"{self.api_base_url}/predict/{prediction_type}",
                json=features
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error for {prediction_type}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Prediction failed for {prediction_type}: {e}")
            return None
    
    def render_header(self):
        """Render the dashboard header"""
        st.markdown('<h1 class="main-header">🍽️ Swiggy Restaurant Analytics</h1>', unsafe_allow_html=True)
        st.markdown("### Machine Learning Powered Restaurant Insights")
        
        # API health status
        api_healthy, health_data = self.check_api_health()
        if api_healthy:
            st.success("✅ API Connection: Healthy")
            if health_data:
                st.sidebar.markdown("### 📊 API Status")
                st.sidebar.write(f"**Models Loaded:** {', '.join(health_data['models_loaded'])}")
                st.sidebar.write(f"**Feature Mapping:** {health_data.get('feature_mapping', 'Fixed 30-feature')}")
                st.sidebar.write(f"**Expected Features:** {health_data.get('expected_features_per_model', {}).get('high_rated', 'N/A')}")
                st.sidebar.write(f"**Last Check:** {health_data['timestamp']}")
        else:
            st.error("❌ API Connection: Failed - Make sure FastAPI server is running on port 8000")
            st.info("💡 Run this command in another terminal: `python -m uvicorn run_phase8_fixed:app --reload --host 0.0.0.0 --port 8000`")
            return False
        return True
    
    def render_sidebar(self):
        """Render the sidebar with input controls"""
        st.sidebar.markdown("## 🔧 Restaurant Features")
        
        # Basic restaurant information
        st.sidebar.markdown("### 📍 Basic Information")
        city = st.sidebar.selectbox(
            "City",
            ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "other"]
        )
        
        st.sidebar.markdown("### 💰 Pricing & Menu")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            avg_price = st.sidebar.number_input("Average Price (INR)", min_value=0, max_value=2000, value=450, step=10)
            dish_count = st.sidebar.number_input("Number of Dishes", min_value=1, max_value=500, value=85, step=1)
        
        with col2:
            price_std = st.sidebar.number_input("Price Standard Deviation", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
            category_diversity = st.sidebar.number_input("Category Diversity", min_value=1, max_value=20, value=8, step=1)
        
        st.sidebar.markdown("### ⭐ Ratings & Popularity")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            avg_rating = st.sidebar.slider("Average Rating", min_value=1.0, max_value=5.0, value=4.1, step=0.1)
            median_rating = st.sidebar.slider("Median Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
        
        with col2:
            total_rating_count = st.sidebar.number_input("Total Rating Count", min_value=0, max_value=10000, value=1200, step=50)
            rating_std = st.sidebar.slider("Rating Standard Deviation", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
        
        # Derived features (calculated automatically)
        price_to_dish_ratio = avg_price / max(dish_count, 1)
        rating_count_per_dish = total_rating_count / max(dish_count, 1)
        has_high_variance = 1 if rating_std > 0.5 else 0
        price_volatility = price_std / max(avg_price, 1)
        
        # Store all features
        features = {
            'avg_price': avg_price,
            'dish_count': dish_count,
            'total_rating_count': total_rating_count,
            'avg_rating': avg_rating,
            'median_rating': median_rating,
            'rating_std': rating_std,
            'price_std': price_std,
            'category_diversity': category_diversity,
            'price_to_dish_ratio': round(price_to_dish_ratio, 2),
            'rating_count_per_dish': round(rating_count_per_dish, 2),
            'has_high_variance': has_high_variance,
            'price_volatility': round(price_volatility, 2),
            'city': city
        }
        
        # Show derived features
        with st.sidebar.expander("📈 Derived Features"):
            st.write(f"**Price per Dish:** ₹{price_to_dish_ratio:.2f}")
            st.write(f"**Ratings per Dish:** {rating_count_per_dish:.1f}")
            st.write(f"**Price Volatility:** {price_volatility:.2f}")
            st.write(f"**High Rating Variance:** {'Yes' if has_high_variance else 'No'}")
        
        return features
    
    def render_prediction_cards(self, features):
        """Render prediction results as cards"""
        st.markdown("## 🎯 Restaurant Predictions")
        
        # Try to get ML predictions
        ml_predictions = {}
        ml_success = False
        
        for prediction_type in ['high-rated', 'popular', 'premium']:
            prediction = self.make_ml_prediction(features, prediction_type)
            if prediction:
                ml_predictions[prediction_type] = prediction
                ml_success = True
        
        # Show status
        if ml_success:
            st.success("✅ ML Predictions: Active - Using trained machine learning models")
        else:
            st.warning("⚠️ ML Predictions: Fallback Mode - Using rule-based analysis")
        
        # Create columns for predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'high-rated' in ml_predictions:
                self.render_ml_prediction_card(ml_predictions['high-rated'], "⭐ High-Rated Restaurant")
            else:
                self.render_rule_based_card(
                    "⭐ High-Rated Restaurant",
                    "Restaurants with rating ≥ 4.2",
                    features['avg_rating'] >= 4.2,
                    features['avg_rating'] / 5.0,
                    "High-rated probability based on current rating"
                )
        
        with col2:
            if 'popular' in ml_predictions:
                self.render_ml_prediction_card(ml_predictions['popular'], "🔥 Popular Restaurant")
            else:
                self.render_rule_based_card(
                    "🔥 Popular Restaurant", 
                    "Restaurants with 100+ ratings",
                    features['total_rating_count'] >= 100,
                    min(features['total_rating_count'] / 1000.0, 1.0),
                    "Popularity probability based on rating count"
                )
        
        with col3:
            if 'premium' in ml_predictions:
                self.render_ml_prediction_card(ml_predictions['premium'], "💎 Premium Restaurant")
            else:
                self.render_rule_based_card(
                    "💎 Premium Restaurant",
                    "Restaurants in top 20% price range",
                    features['avg_price'] > 400,
                    min(features['avg_price'] / 1000.0, 1.0),
                    "Premium probability based on pricing"
                )
        
        # Show technical details in expander
        with st.expander("🔧 Technical Details"):
            if ml_success:
                st.success("**ML Integration**: ✅ Working - Models receiving proper 30-feature format")
                st.info(f"**Features Used**: {ml_predictions.get('high-rated', {}).get('features_used', 'N/A')} features per prediction")
            else:
                st.error("**Current Issue**: Feature dimension mismatch between dashboard and ML models")
                st.info("**Solution**: Use the fixed API with proper 30-feature mapping")
                st.write("**Temporary Solution**: Using rule-based analysis based on business thresholds")
    
    def render_ml_prediction_card(self, prediction_data, title):
        """Render ML prediction card"""
        prediction = prediction_data.get('prediction', 0)
        probability = prediction_data.get('probability', 0.5)
        confidence = prediction_data.get('confidence', 'medium')
        
        if prediction == 1:
            emoji = "✅"
            message = "YES"
            bg_color = "#d4edda"
            text_color = "#155724"
            border_color = "green"
        else:
            emoji = "❌"
            message = "NO"
            bg_color = "#f8d7da"
            text_color = "#721c24"
            border_color = "red"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 10px; background-color: {bg_color}; border-left: 4px solid {border_color}; margin: 1rem 0;" class="ml-success">
            <h3 style="margin: 0; color: {text_color};">{emoji} {title}</h3>
            <p style="margin: 0.5rem 0; color: {text_color}; opacity: 0.8;">ML Model Prediction</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="font-size: 1.5rem; font-weight: bold; color: {text_color};">{message}</span>
                <span style="font-size: 1.2rem; color: {text_color};" title="ML Model Confidence">{probability:.1%}</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">Confidence: {confidence.upper()}</small>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">🎯 ML Model</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_rule_based_card(self, title, description, prediction, probability, tooltip):
        """Render rule-based prediction card"""
        # Determine emoji and message based on prediction
        if prediction:
            emoji = "✅"
            message = "YES"
            bg_color = "#e2e3e5"
            text_color = "#383d41"
            border_color = "#6c757d"
            confidence = "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"
        else:
            emoji = "❌"
            message = "NO"
            bg_color = "#e2e3e5"
            text_color = "#383d41"
            border_color = "#6c757d"
            confidence = "low"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 10px; background-color: {bg_color}; border-left: 4px solid {border_color}; margin: 1rem 0;" class="ml-warning">
            <h3 style="margin: 0; color: {text_color};">{emoji} {title}</h3>
            <p style="margin: 0.5rem 0; color: {text_color}; opacity: 0.8;">{description}</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="font-size: 1.5rem; font-weight: bold; color: {text_color};">{message}</span>
                <span style="font-size: 1.2rem; color: {text_color};" title="{tooltip}">{probability:.1%}</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">Confidence: {confidence.upper()}</small>
            </div>
            <div style="margin-top: 0.5rem;">
                <small style="color: {text_color}; opacity: 0.8;">📊 Rule-Based</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_feature_analysis(self, features):
        """Render feature analysis and insights"""
        st.markdown("## 📊 Feature Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Feature Overview", "🎯 Business Insights", "📋 Feature Data"])
        
        with tab1:
            self.render_feature_overview(features)
        
        with tab2:
            self.render_business_insights(features)
        
        with tab3:
            self.render_feature_data(features)
    
    def render_feature_overview(self, features):
        """Render feature overview with visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Price and rating metrics
            st.subheader("💰 Pricing Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Average Price", f"₹{features['avg_price']}")
                st.metric("Price per Dish", f"₹{features['price_to_dish_ratio']}")
            with metric_col2:
                st.metric("Price Volatility", f"{features['price_volatility']:.2f}")
                st.metric("Dish Count", features['dish_count'])
        
        with col2:
            # Rating metrics
            st.subheader("⭐ Rating Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Average Rating", f"{features['avg_rating']}/5")
                st.metric("Total Ratings", f"{features['total_rating_count']:,}")
            with metric_col2:
                st.metric("Rating Consistency", "High" if features['has_high_variance'] == 0 else "Low")
                st.metric("Ratings per Dish", f"{features['rating_count_per_dish']:.1f}")
        
        # Create a simple bar chart for key metrics
        st.subheader("📊 Key Metrics Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = ['avg_price', 'dish_count', 'total_rating_count', 'category_diversity']
        metric_names = ['Avg Price', 'Dish Count', 'Rating Count', 'Categories']
        values = [features[metric] for metric in metrics_to_plot]
        
        # Normalize values for better visualization
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val * 100 for v in values]
        
        bars = ax.bar(metric_names, normalized_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Normalized Score (%)')
        ax.set_title('Restaurant Metrics (Normalized)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:,}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    def render_business_insights(self, features):
        """Render business insights based on features"""
        st.subheader("💡 Business Insights")
        
        insights = []
        
        # Generate insights based on feature values
        if features['avg_rating'] >= 4.2:
            insights.append("✅ **High Potential**: Current ratings suggest this could be a high-rated restaurant")
        else:
            insights.append("📈 **Improvement Opportunity**: Ratings are below the high-rated threshold (4.2+)")
        
        if features['total_rating_count'] >= 100:
            insights.append("🔥 **Popular Spot**: Has sufficient ratings to be considered popular")
        else:
            insights.append("👥 **Growth Potential**: Needs more ratings to reach popular status")
        
        if features['avg_price'] > 400:
            insights.append("💎 **Premium Positioning**: Pricing suggests a premium restaurant")
        else:
            insights.append("💰 **Value Focus**: Competitive pricing indicates value positioning")
        
        if features['rating_std'] < 0.5:
            insights.append("🎯 **Consistent Quality**: Low rating variance indicates consistent customer experience")
        else:
            insights.append("⚠️ **Inconsistent Experience**: High rating variance suggests mixed customer feedback")
        
        if features['dish_count'] > 50:
            insights.append("📋 **Extensive Menu**: Wide variety of dishes available")
        else:
            insights.append("🍽️ **Focused Menu**: Curated selection of dishes")
        
        # Display insights
        for insight in insights:
            st.write(insight)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        if features['avg_rating'] < 4.2 and features['total_rating_count'] < 100:
            st.info("**Focus on Quality & Marketing**: Improve food quality and actively encourage customer reviews to boost ratings and popularity")
        elif features['avg_rating'] >= 4.2 and features['total_rating_count'] < 100:
            st.info("**Leverage High Ratings**: Use excellent ratings in marketing to attract more customers and increase review count")
        elif features['avg_price'] > 500:
            st.info("**Premium Experience**: Ensure service and ambiance match the premium pricing to justify the cost")
        else:
            st.info("**Solid Foundation**: Maintain current quality standards while exploring opportunities for menu expansion or premium offerings")
    
    def render_feature_data(self, features):
        """Render feature data without using dataframe to avoid serialization issues"""
        st.subheader("📋 Feature Data")
        
        # Display features in a clean format without using dataframe
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 Pricing & Menu**")
            st.write(f"- Average Price: ₹{features['avg_price']}")
            st.write(f"- Dish Count: {features['dish_count']}")
            st.write(f"- Price Std Dev: {features['price_std']}")
            st.write(f"- Category Diversity: {features['category_diversity']}")
            st.write(f"- Price per Dish: ₹{features['price_to_dish_ratio']}")
            st.write(f"- Price Volatility: {features['price_volatility']:.2f}")
        
        with col2:
            st.write("**⭐ Ratings & Popularity**")
            st.write(f"- Average Rating: {features['avg_rating']}/5")
            st.write(f"- Median Rating: {features['median_rating']}/5")
            st.write(f"- Total Ratings: {features['total_rating_count']:,}")
            st.write(f"- Rating Std Dev: {features['rating_std']}")
            st.write(f"- Ratings per Dish: {features['rating_count_per_dish']:.1f}")
            st.write(f"- Rating Consistency: {'High' if features['has_high_variance'] == 0 else 'Low'}")
        
        st.write("**📍 Location**")
        st.write(f"- City: {features['city'].title()}")
    
    def render_batch_analysis(self):
        """Render batch analysis section"""
        st.markdown("## 📊 Batch Analysis")
        
        st.info("💡 **Coming Soon**: Batch prediction functionality will be available once the ML API integration is complete.")
        
        # Placeholder for future batch functionality
        with st.expander("🔮 Future Features"):
            st.write("""
            **Planned Batch Analysis Features:**
            
            - 📁 **CSV Upload**: Upload restaurant data in bulk
            - 🚀 **Batch Predictions**: Get predictions for multiple restaurants at once
            - 📊 **Comparative Analysis**: Compare multiple restaurants side by side
            - 📥 **Export Results**: Download predictions as CSV
            - 📈 **Trend Analysis**: Identify patterns across multiple restaurants
            """)
    
    def run(self):
        """Run the Streamlit dashboard"""
        # Render header and check API
        if not self.render_header():
            return
        
        # Create main layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            features = self.render_sidebar()
        
        with col2:
            self.render_prediction_cards(features)
            self.render_feature_analysis(features)
        
        # Batch analysis at the bottom
        self.render_batch_analysis()

def main():
    print("🚀 Starting Phase 9: Streamlit Dashboard")
    print("=" * 60)
    
    try:
        # Initialize and run the dashboard
        dashboard = StreamlitDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"❌ Phase 9 failed: {e}")
        st.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()