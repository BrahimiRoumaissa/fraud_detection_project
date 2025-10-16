#!/usr/bin/env python3
"""
Simple test script to verify all required packages are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('plotly.express', 'px'),
        ('sklearn', 'sklearn'),
        ('imblearn.over_sampling', 'SMOTE'),
        ('xgboost', 'xgb'),
        ('joblib', 'joblib'),
        ('streamlit', 'st')
    ]
    
    success = True
    for package, alias in packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"OK - {package} imported successfully")
        except ImportError as e:
            print(f"ERROR - {package} import failed: {e}")
            success = False
    
    return success

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        print(f"OK - Model training works (accuracy: {score:.3f})")
        return True
        
    except Exception as e:
        print(f"ERROR - Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Fraud Detection System - Installation Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    if imports_ok and functionality_ok:
        print("\nSUCCESS - All tests passed!")
        print("\nNext steps:")
        print("1. Open fraud_detection_analysis.ipynb in Jupyter")
        print("2. Run all cells to train models")
        print("3. Run: streamlit run app.py")
        print("4. Use the web interface for fraud detection")
    else:
        print("\nFAILURE - Some tests failed. Check error messages above.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
