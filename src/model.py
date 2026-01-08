"""Model Training and Evaluation Module"""

import pickle
from pathlib import Path
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class DiabetesClassifier:
    """SVM Classifier for Diabetes Prediction"""
    
    def __init__(self, kernel: str = 'linear'):
        """
        Initialize classifier
        
        Args:
            kernel: SVM kernel type ('linear', 'rbf', 'poly', etc.)
        """
        self.classifier = svm.SVC(kernel=kernel)
        self.model_path = None
        
    def train(self, X_train, Y_train):
        """
        Train the classifier
        
        Args:
            X_train: Training features
            Y_train: Training labels
        """
        self.classifier.fit(X_train, Y_train)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted labels
        """
        return self.classifier.predict(X)
    
    def evaluate(self, X, Y, set_name: str = ""):
        """
        Evaluate model performance
        
        Args:
            X: Features
            Y: True labels
            set_name: Name of dataset (for display)
            
        Returns:
            dict: Metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(Y, predictions),
            'precision': precision_score(Y, predictions),
            'recall': recall_score(Y, predictions),
            'f1': f1_score(Y, predictions),
            'confusion_matrix': confusion_matrix(Y, predictions).tolist()
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Save trained model"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
        self.model_path = path
    
    def load_model(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        self.model_path = path
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'kernel': self.classifier.kernel,
            'model_type': type(self.classifier).__name__,
            'saved_path': self.model_path
        }
