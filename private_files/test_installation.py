#!/usr/bin/env python3
"""
Test script to verify all required packages are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn imported successfully")
    except ImportError as e:
        print(f"❌ seaborn import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, IsolationForest
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from imblearn.over_sampling import SMOTE
        print("✅ imbalanced-learn imported successfully")
    except ImportError as e:
        print(f"❌ imbalanced-learn import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ xgboost imported successfully")
    except ImportError as e:
        print(f"❌ xgboost import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ joblib imported successfully")
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key packages"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print("✅ pandas DataFrame creation works")
        
        # Test numpy
        arr = np.array([1, 2, 3, 4, 5])
        print("✅ numpy array creation works")
        
        # Test scikit-learn
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"✅ scikit-learn model training works (accuracy: {score:.3f})")
        
        # Test SMOTE
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE works (original: {len(y_train)}, resampled: {len(y_resampled)})")
        
        # Test XGBoost
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        print(f"✅ XGBoost model training works (accuracy: {xgb_score:.3f})")
        
        # Test joblib
        import joblib
        joblib.dump(model, 'test_model.pkl')
        loaded_model = joblib.load('test_model.pkl')
        print("✅ joblib save/load works")
        
        # Clean up
        import os
        os.remove('test_model.pkl')
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Fraud Detection System - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! The fraud detection system is ready to use.")
            print("\n📋 Next steps:")
            print("1. Open the Jupyter notebook: fraud_detection_analysis.ipynb")
            print("2. Run all cells to train the models")
            print("3. Launch the Streamlit app: streamlit run app.py")
            print("4. Use the web interface for fraud detection")
        else:
            print("\n⚠️  Some functionality tests failed. Check the error messages above.")
    else:
        print("\n❌ Some packages failed to import. Please install missing packages.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
