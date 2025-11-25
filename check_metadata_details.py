# check_metadata_details.py
import joblib
import pandas as pd

def check_metadata_details():
    """Check exactly what's in the feature metadata"""
    try:
        metadata = joblib.load('models/deployment_feature_metadata.pkl')
        print("🔍 DETAILED FEATURE METADATA ANALYSIS")
        print("=" * 60)
        
        print(f"📊 Metadata type: {type(metadata)}")
        
        if isinstance(metadata, dict):
            print("📋 All metadata keys and values:")
            for key, value in metadata.items():
                print(f"\n🎯 Key: {key}")
                print(f"   Type: {type(value)}")
                
                if isinstance(value, list):
                    print(f"   Length: {len(value)}")
                    if len(value) > 0:
                        print(f"   Values: {value}")
                elif isinstance(value, dict):
                    print(f"   Dict keys: {list(value.keys())[:10]}")  # First 10 keys
                elif isinstance(value, (str, int, float, bool)):
                    print(f"   Value: {value}")
                elif hasattr(value, 'shape'):
                    print(f"   Shape: {value.shape}")
                else:
                    print(f"   Value (repr): {repr(value)[:200]}...")  # First 200 chars
                    
        # Check if it's actually a feature mapping
        print("\n🔎 CHECKING FOR FEATURE MAPPING PATTERNS:")
        if isinstance(metadata, dict):
            # Look for feature-related keys
            feature_keys = [k for k in metadata.keys() if 'feature' in k.lower() or 'column' in k.lower()]
            if feature_keys:
                print("✅ Found feature-related keys:")
                for key in feature_keys:
                    print(f"   - {key}")
            else:
                print("❌ No feature-related keys found")
                
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")

if __name__ == "__main__":
    check_metadata_details()