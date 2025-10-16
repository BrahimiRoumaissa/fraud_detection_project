"""
Machine learning models for fraud detection.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd


class FraudDetectionModels:
    """
    Manages multiple ML models for fraud detection.
    
    This class provides a unified interface for training and evaluating
    different machine learning algorithms for fraud detection.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the models.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def initialize_models(self):
        """
        Initialize all fraud detection models.
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state, 
                eval_metric='logloss'
            ),
            'Isolation Forest': IsolationForest(
                contamination=0.001727, 
                random_state=self.random_state
            )
        }
        
        print("✅ All models initialized successfully")
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate performance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Model results
        """
        if not self.models:
            self.initialize_models()
            
        for name, model in self.models.items():
            print(f"🔄 Training {name}...")
            
            try:
                if name == 'Isolation Forest':
                    # Unsupervised model
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    # Convert predictions: -1 (anomaly) -> 1 (fraud), 1 (normal) -> 0
                    y_pred = (y_pred == -1).astype(int)
                    
                    # Calculate AUC using decision function
                    train_auc = roc_auc_score(y_train, model.decision_function(X_train))
                    test_auc = roc_auc_score(y_test, model.decision_function(X_test))
                    
                else:
                    # Supervised models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate AUC
                    if hasattr(model, 'predict_proba'):
                        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
                        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    else:
                        train_auc = roc_auc_score(y_train, model.decision_function(X_train))
                        test_auc = roc_auc_score(y_test, model.decision_function(X_test))
                
                # Calculate metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Store results
                self.results[name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'y_pred': y_pred
                }
                
                self.trained_models[name] = model
                
                print(f"✅ {name} trained successfully!")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                print(f"   F1-score: {f1:.4f}")
                print(f"   Test AUC: {test_auc:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                
        return self.results
    
    def get_best_model(self):
        """
        Get the best performing model based on F1-score.
        
        Returns:
            tuple: (model_name, model, results)
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
            
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_score'])
        best_model = self.trained_models[best_model_name]
        best_results = self.results[best_model_name]
        
        return best_model_name, best_model, best_results
    
    def get_model_comparison(self):
        """
        Get comparison of all models.
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
            
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Train AUC': results['train_auc'],
                'Test AUC': results['test_auc']
            })
        
        return pd.DataFrame(comparison_data).sort_values('F1-Score', ascending=False)
    
    def save_models(self, directory='models'):
        """
        Save trained models to disk.
        
        Args:
            directory (str): Directory to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save all models
        joblib.dump(self.trained_models, f'{directory}/all_models.pkl')
        
        # Save best model
        best_name, best_model, _ = self.get_best_model()
        joblib.dump(best_model, f'{directory}/best_model.pkl')
        
        # Save results
        joblib.dump(self.results, f'{directory}/model_results.pkl')
        
        print(f"✅ Models saved to {directory}/")
        print(f"✅ Best model ({best_name}) saved as best_model.pkl")
    
    def load_models(self, directory='models'):
        """
        Load trained models from disk.
        
        Args:
            directory (str): Directory containing models
        """
        try:
            self.trained_models = joblib.load(f'{directory}/all_models.pkl')
            self.results = joblib.load(f'{directory}/model_results.pkl')
            print(f"✅ Models loaded from {directory}/")
        except FileNotFoundError:
            print(f"❌ Models not found in {directory}/")
    
    def predict_fraud(self, X, model_name=None):
        """
        Predict fraud for new data.
        
        Args:
            X (pd.DataFrame): Features
            model_name (str): Specific model to use (None for best model)
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if not self.trained_models:
            raise ValueError("No trained models available")
            
        if model_name is None:
            model_name, model, _ = self.get_best_model()
        else:
            model = self.trained_models[model_name]
        
        predictions = model.predict(X)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            # For Isolation Forest, use decision function
            probabilities = model.decision_function(X)
            probabilities = -probabilities  # Convert to positive scale
        
        return predictions, probabilities


def create_model_summary(results):
    """
    Create a summary of model results.
    
    Args:
        results (dict): Model results
        
    Returns:
        dict: Summary statistics
    """
    if not results:
        return {}
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_results = results[best_model_name]
    
    summary = {
        'best_model': best_model_name,
        'best_f1_score': best_results['f1_score'],
        'best_precision': best_results['precision'],
        'best_recall': best_results['recall'],
        'best_auc': best_results['test_auc'],
        'total_models': len(results),
        'model_names': list(results.keys())
    }
    
    return summary
