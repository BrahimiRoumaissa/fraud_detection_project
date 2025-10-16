"""
Evaluation utilities for fraud detection models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, 
    precision_score, recall_score
)


class ModelEvaluator:
    """
    Comprehensive model evaluation for fraud detection.
    
    This class provides methods for evaluating model performance
    with fraud detection specific metrics and visualizations.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate a single model.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
        
        self.results[model_name] = metrics
        return metrics
    
    def calculate_auc(self, y_true, y_scores, model_name):
        """
        Calculate ROC AUC score.
        
        Args:
            y_true (array-like): True labels
            y_scores (array-like): Prediction scores/probabilities
            model_name (str): Name of the model
            
        Returns:
            float: AUC score
        """
        auc_score = roc_auc_score(y_true, y_scores)
        
        if model_name in self.results:
            self.results[model_name]['auc'] = auc_score
        
        return auc_score
    
    def plot_confusion_matrices(self, results_dict, figsize=(16, 12)):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            results_dict (dict): Model results dictionary
            figsize (tuple): Figure size
        """
        n_models = len(results_dict)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (name, results) in enumerate(results_dict.items()):
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[i], xticklabels=['Normal', 'Fraud'], 
                           yticklabels=['Normal', 'Fraud'])
                axes[i].set_title(f'{name}\\nConfusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_true, y_scores_dict, figsize=(12, 8)):
        """
        Plot ROC curves for multiple models.
        
        Args:
            y_true (array-like): True labels
            y_scores_dict (dict): Dictionary of model scores
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, y_scores in y_scores_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.4f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(self, y_true, y_scores_dict, figsize=(12, 8)):
        """
        Plot precision-recall curves for multiple models.
        
        Args:
            y_true (array-like): True labels
            y_scores_dict (dict): Dictionary of model scores
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, y_scores in y_scores_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            
            plt.plot(recall, precision, linewidth=2, label=f'{name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results_df, figsize=(16, 12)):
        """
        Plot model performance comparison.
        
        Args:
            results_df (pd.DataFrame): Model results dataframe
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        metrics = ['Precision', 'Recall', 'F1-Score', 'Test AUC']
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                ax = axes[i//2, i%2]
                results_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance_scores, 
                              model_name, top_n=15, figsize=(12, 8)):
        """
        Plot feature importance for tree-based models.
        
        Args:
            feature_names (list): List of feature names
            importance_scores (array-like): Feature importance scores
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def generate_classification_report(self, y_true, y_pred, model_name):
        """
        Generate detailed classification report.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            model_name (str): Name of the model
            
        Returns:
            str: Classification report
        """
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Fraud'])
        
        print(f"\\n{model_name} - Classification Report:")
        print("-" * len(model_name))
        print(report)
        
        return report
    
    def calculate_business_metrics(self, y_true, y_pred, cost_per_fraud=1000, 
                                 cost_per_false_positive=10):
        """
        Calculate business-relevant metrics for fraud detection.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            cost_per_fraud (float): Cost of a missed fraud
            cost_per_false_positive (float): Cost of a false positive
            
        Returns:
            dict: Business metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate costs
        fraud_cost = fn * cost_per_fraud
        false_positive_cost = fp * cost_per_false_positive
        total_cost = fraud_cost + false_positive_cost
        
        # Calculate savings
        prevented_fraud_savings = tp * cost_per_fraud
        net_savings = prevented_fraud_savings - total_cost
        
        business_metrics = {
            'fraud_cost': fraud_cost,
            'false_positive_cost': false_positive_cost,
            'total_cost': total_cost,
            'prevented_fraud_savings': prevented_fraud_savings,
            'net_savings': net_savings,
            'cost_benefit_ratio': net_savings / total_cost if total_cost > 0 else 0
        }
        
        return business_metrics


def create_evaluation_summary(results_dict):
    """
    Create a comprehensive evaluation summary.
    
    Args:
        results_dict (dict): Dictionary of model results
        
    Returns:
        pd.DataFrame: Summary dataframe
    """
    summary_data = []
    
    for model_name, results in results_dict.items():
        summary_data.append({
            'Model': model_name,
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'AUC': results.get('auc', 0),
            'Specificity': results.get('specificity', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1-Score', ascending=False)
    
    return summary_df
