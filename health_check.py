"""
Health Check and Diagnostic Script for Streamlit Deployment
===========================================================
"""

import os
import sys

def check_files():
    """Check if all required files exist"""
    required_files = [
        'streamlit_app.py',
        'disaster_prediction_model.py',
        'disaster_prediction_model.pkl',
        'public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv',
        'requirements.txt'
    ]
    
    print("=== FILE CHECK ===")
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - MISSING")
            all_files_exist = False
    
    return all_files_exist

def check_imports():
    """Check if critical imports work"""
    print("\n=== IMPORT CHECK ===")
    imports = [
        ('streamlit', 'st'),
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('xgboost', 'xgb'),
        ('joblib', 'joblib'),
        ('plotly', 'plotly')
    ]
    
    failed_imports = []
    for module, alias in imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            failed_imports.append(module)
    
    # Special check for TensorFlow
    try:
        import tensorflow
        print(f"✅ tensorflow ({tensorflow.__version__})")
    except ImportError:
        print("⚠️  tensorflow - Optional (will use safe training mode)")
    
    return len(failed_imports) == 0

def check_model():
    """Check if model can be loaded"""
    print("\n=== MODEL CHECK ===")
    try:
        from disaster_prediction_model import DisasterPredictionModel
        
        csv_file = "public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv"
        model = DisasterPredictionModel(csv_file)
        
        if os.path.exists("disaster_prediction_model.pkl"):
            model.load_model("disaster_prediction_model.pkl")
            print("✅ Pre-trained model loaded successfully")
        else:
            print("⚠️  No pre-trained model found - will train on startup")
        
        print("✅ Model class initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("🚀 India Disaster Prediction System - Health Check")
    print("=" * 60)
    
    files_ok = check_files()
    imports_ok = check_imports()
    model_ok = check_model()
    
    print("\n=== SUMMARY ===")
    if files_ok and imports_ok and model_ok:
        print("✅ All checks passed! App should work correctly.")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        if not files_ok:
            print("   - Missing required files")
        if not imports_ok:
            print("   - Missing required packages")
        if not model_ok:
            print("   - Model loading issues")

if __name__ == "__main__":
    main()