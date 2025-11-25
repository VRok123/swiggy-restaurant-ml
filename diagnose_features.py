# diagnose_features.py
import pandas as pd
import joblib
import os
import sys
sys.path.append('src')

def find_model_files():
    """Find all model files in the models directory"""
    models_dir = "models"
    model_files = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.pkl', '.joblib', '.sav')):
                model_files.append(os.path.join(models_dir, file))
    
    # Also check root directory
    for file in os.listdir('.'):
        if file.endswith(('.pkl', '.joblib', '.sav')):
            model_files.append(file)
    
    return model_files

def check_model_features():
    """Check what features each model was trained with"""
    print("🔍 MODEL FEATURE DIAGNOSTIC")
    print("=" * 60)
    
    model_files = find_model_files()
    
    if not model_files:
        print("❌ No model files found! Looking for .pkl, .joblib, or .sav files")
        print("📁 Checked directories: 'models/', './'")
        return
    
    print(f"📁 Found {len(model_files)} model files:")
    
    for model_path in model_files:
        print(f"\n📊 Analyzing: {os.path.basename(model_path)}")
        print(f"📁 Model path: {model_path}")
        
        try:
            # Load the model
            model = joblib.load(model_path)
            print(f"   ✅ Model loaded successfully")
            print(f"   🏷️ Model type: {type(model).__name__}")
            
            # Check for feature names (modern scikit-learn)
            if hasattr(model, 'feature_names_in_'):
                print(f"   📋 Expected features ({len(model.feature_names_in_)}):")
                for i, feature in enumerate(model.feature_names_in_, 1):
                    print(f"      {i:2d}. {feature}")
            else:
                print(f"   ⚠️ No feature names stored in model object")
                
            # Check for feature importance if available
            if hasattr(model, 'feature_importances_'):
                print(f"   📈 Model has feature importances ({len(model.feature_importances_)} features)")
                
            # Check for coefficients (linear models)
            if hasattr(model, 'coef_'):
                num_features = len(model.coef_) if len(model.coef_.shape) == 1 else model.coef_.shape[1]
                print(f"   📐 Linear model with {num_features} coefficients")
                
            # Try to get number of features expected
            if hasattr(model, 'n_features_in_'):
                print(f"   🔢 Expected number of features: {model.n_features_in_}")
                
            # For ensemble models
            if hasattr(model, 'n_features_in_'):
                print(f"   🌳 Expected features: {model.n_features_in_}")
                
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
    
    print("\n" + "=" * 60)

def check_training_data_features():
    """Check if we have training data to understand feature sets"""
    print("\n\n🔍 CHECKING TRAINING DATA FEATURES")
    print("=" * 60)
    
    # Common training data paths
    data_paths = [
        'data/processed/train_data.csv',
        'data/processed/training_data.csv', 
        'data/processed/restaurant_data.csv',
        'data/final/swiggy_restaurant_data.csv',
        'data/processed/swiggy_features.csv',  # From your config
        'data/processed/swiggy_clean.csv'      # From your config
    ]
    
    found_data = False
    for data_path in data_paths:
        if os.path.exists(data_path):
            print(f"\n📁 Found data file: {data_path}")
            try:
                df = pd.read_csv(data_path)
                print(f"   📊 Shape: {df.shape}")
                print(f"   📋 Columns ({len(df.columns)}):")
                for col in df.columns:
                    print(f"      - {col}")
                found_data = True
                
                # Show sample of data types
                print(f"   🔍 Data types:")
                print(f"      {df.dtypes.head(10)}")
                break
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"   ❌ Data file not found: {data_path}")
    
    if not found_data:
        print("❌ No training data files found!")

def check_phase8_api():
    """Check if we have the FastAPI phase 8 files"""
    print("\n\n🔍 CHECKING PHASE 8 API FILES")
    print("=" * 60)
    
    api_files = [
        'run_phase8.py',
        'src/run_phase8.py', 
        'phase8_api.py',
        'src/phase8_api.py'
    ]
    
    for file_path in api_files:
        if os.path.exists(file_path):
            print(f"✅ Found API file: {file_path}")
            # Try to read and find model configurations
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'MODELS_CONFIG' in content:
                        print(f"   📝 MODELS_CONFIG found in {file_path}")
                    if 'predict' in content:
                        print(f"   🎯 Predict endpoints defined in {file_path}")
            except:
                pass
        else:
            print(f"❌ API file not found: {file_path}")

if __name__ == "__main__":
    check_model_features()  # This calls find_model_files() internally
    check_training_data_features()
    check_phase8_api()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Based on the model files found, we'll create MODELS_CONFIG")
    print("2. Update the feature mapping in your Streamlit dashboard")
    print("3. Fix the API endpoint feature expectations")