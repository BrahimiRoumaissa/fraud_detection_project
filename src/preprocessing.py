"""
Data preprocessing utilities for fraud detection system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy import stats


class DataPreprocessor:
    """
    Handles data preprocessing for fraud detection.
    
    This class provides methods for data cleaning, feature scaling,
    and handling class imbalance in fraud detection datasets.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.smote = SMOTE(random_state=random_state)
        self.is_fitted = False
        
    def clean_data(self, df):
        """
        Clean the dataset by handling outliers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Handle extreme outliers using IQR method
        for col in ['Time', 'Amount']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def prepare_features_target(self, df, target_col='Class'):
        """
        Separate features and target variable.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
            
        Returns:
            tuple: (X, y) features and target
        """
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
    
    def fit_transform_scaler(self, X_train, X_test):
        """
        Fit scaler on training data and transform both sets.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        self.is_fitted = True
        
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        X_balanced, y_balanced = self.smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        
        return X_balanced, y_balanced
    
    def get_data_quality_report(self, df):
        """
        Generate data quality report.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Data quality metrics
        """
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'extreme_outliers': (np.abs(df.select_dtypes(include=[np.number])) > 5).sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return report


def load_sample_data():
    """
    Load sample fraud detection data.
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    # This is a placeholder function
    # In a real implementation, this would load actual data
    print("Loading sample fraud detection dataset...")
    return None


def validate_data(df):
    """
    Validate input data for fraud detection.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        bool: True if data is valid
    """
    required_columns = ['Time', 'Amount', 'Class']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")
    
    if df['Class'].nunique() != 2:
        raise ValueError("Target column must be binary")
    
    return True
