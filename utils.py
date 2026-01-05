"""
Diabetes Prediction - Utility Functions and Helpers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


def print_metrics_report(metrics: Dict) -> None:
    """
    Print formatted metrics report
    
    Args:
        metrics: Dictionary of metrics from classifier.evaluate()
    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}  TP: {cm[1][1]}")
    print("=" * 50)


def validate_patient_data(patient_data: Tuple, num_features: int = 8) -> bool:
    """
    Validate patient input data
    
    Args:
        patient_data: Tuple of patient features
        num_features: Expected number of features
        
    Returns:
        bool: True if valid, False otherwise
    """
    if len(patient_data) != num_features:
        return False
    
    if not all(isinstance(x, (int, float)) and x >= 0 for x in patient_data):
        return False
    
    return True


def format_prediction_report(result: Dict) -> str:
    """
    Format prediction result as a readable report
    
    Args:
        result: Prediction result from predictor.predict_single()
        
    Returns:
        str: Formatted report
    """
    report = f"""
╔════════════════════════════════════════════════════╗
║          DIABETES PREDICTION REPORT               ║
╚════════════════════════════════════════════════════╝

PATIENT DATA:
{chr(10).join(f"  • {k}: {v}" for k, v in result['patient_data'].items())}

PREDICTION RESULT:
  • Status: {result['prediction_text']}
  • Confidence Score: {result['confidence_score']:.6f}

╔════════════════════════════════════════════════════╗
"""
    return report


def get_feature_importance_summary(df: pd.DataFrame, Y: pd.Series) -> Dict:
    """
    Get summary statistics of features by class
    
    Args:
        df: Feature DataFrame
        Y: Target Series
        
    Returns:
        dict: Summary statistics
    """
    summary = {}
    for col in df.columns:
        summary[col] = {
            'non_diabetic_mean': df[Y == 0][col].mean(),
            'diabetic_mean': df[Y == 1][col].mean(),
            'difference': df[Y == 1][col].mean() - df[Y == 0][col].mean()
        }
    return summary


def display_feature_comparison(summary: Dict) -> None:
    """
    Display feature comparison between classes
    
    Args:
        summary: Feature importance summary from get_feature_importance_summary()
    """
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON: Non-Diabetic vs Diabetic")
    print("=" * 80)
    print(f"{'Feature':<25} {'Non-Diabetic':<20} {'Diabetic':<20} {'Difference':<15}")
    print("-" * 80)
    
    for feature, stats in summary.items():
        print(f"{feature:<25} {stats['non_diabetic_mean']:>19.4f} "
              f"{stats['diabetic_mean']:>19.4f} {stats['difference']:>14.4f}")
    
    print("=" * 80)
