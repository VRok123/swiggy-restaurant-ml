# 🍽️ Swiggy Restaurant Analytics & ML Prediction System

A comprehensive end-to-end machine learning system that predicts restaurant characteristics and provides real-time analytics for Swiggy data. This production-ready system includes ML models, REST API, and an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🎯 Project Overview

This project delivers a complete ML pipeline for restaurant analytics, featuring:

- **🤖 Machine Learning Models**: 3 specialized classifiers for restaurant prediction
- **🔗 REST API**: FastAPI backend with real-time predictions
- **📊 Interactive Dashboard**: Streamlit frontend with live insights
- **📈 Monitoring**: Production monitoring and performance tracking
- **🚀 Deployment**: Production-ready deployment system

### ML Prediction Tasks

1. **⭐ High-Rated Restaurant** - Predicts if a restaurant maintains ≥4.2 rating
2. **🔥 Popular Restaurant** - Identifies restaurants with high customer engagement
3. **💎 Premium Restaurant** - Classifies premium-priced establishments

## 🏗️ System Architecture

┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ 📊 Streamlit │ │ 🔧 FastAPI │ │ 🧠 ML Models │
│ Dashboard │◄──►│ Backend │◄──►│ (3 deployed) │
│ (localhost:8501)│ │ (localhost:8000) │ │ │
└─────────────────┘ └──────────────────┘ └─────────────────┘
│ │ │
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Real-time │ │ Feature │ │ High Accuracy │
│ Predictions │ │ Processing │ │ Predictions │
└─────────────────┘ └──────────────────┘ └─────────────────┘

## 📁 Project Structure

swiggy-ml/
├── 📊 data/ # Data directory
│ ├── raw/ # Raw Swiggy data files
│ └── processed/ # Processed and feature-engineered data
├── 🔬 notebooks/ # Jupyter notebooks for analysis
├── 🏗️ src/ # Source code
│ ├── utils.py # Utility functions and logging
│ └── config.py # Project configuration
├── 🤖 models/ # Trained ML models (gitignored)
│ ├── deployment_high_rated_model.pkl
│ ├── deployment_popular_model.pkl
│ └── deployment_premium_model.pkl
├── 📈 reports/ # Analysis and deployment reports
├── 🎨 figures/ # Generated visualizations
├── 🔍 monitoring/ # Model monitoring data
├── 🚀 run_phase8_optimized.py # Production API server
├── 📊 run_phase9.py # Streamlit dashboard
├── 📋 run_phase10.py # Monitoring & deployment
└── ⚙️ requirements.txt # Python dependencies

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- pip package manager

### 2. Installation & Setup

```bash
# Clone and setup environment
git clone <your-repo-url>
cd swiggy-ml

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Data Preparation

Place your Swiggy dataset as data/raw/swiggy.csv. The system will automatically process it through the ML pipeline

### 4. Start the System

python -m uvicorn run_phase8_working_final:app --host 0.0.0.0 --port 8000
python start_production.py

### 5. Access the System

🌐 API Documentation: http://localhost:8000/docs

📊 Dashboard: http://localhost:8501

🔍 API Health: http://localhost:8000/health

🛠️ Technical Details
ML Models Deployed
Model Type Algorithm Features Accuracy
High-Rated Classification LightGBM 30 features ~85%
Popular Classification XGBoost 30 features ~87%
Premium Classification LightGBM 30 features ~83%

### Deployment Checks

python run_phase10.py

### Project Phases

Phase 1-3: Data Analysis & Feature Engineering

Phase 4-7: Machine Learning Model Development

Phase 8: FastAPI Production Backend

Phase 9: Streamlit Dashboard

Phase 10: Monitoring & Deployment

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
Built with ❤️ using Python, FastAPI, Streamlit, and Scikit-learn
