"""Data Preprocessing Module"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class DataPreprocessor:
    """Handle data preprocessing and standardization"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        
    def separate_features_target(self, df: pd.DataFrame, target_column: str = 'Outcome'):
        """
        Separate features and target variable
        
        Args:
            df: DataFrame
            target_column: Name of target column
            
        Returns:
            tuple: (X, Y)
        """
        X = df.drop(columns=target_column, axis=1)
        Y = df[target_column]
        return X, Y
    
    def standardize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Standardize features
        
        Args:
            X: Feature array
            fit: If True, fit scaler; if False, only transform
            
        Returns:
            Standardized array
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def train_test_split_data(self, X: np.ndarray, Y: pd.Series, 
                              test_size: float = 0.2, random_state: int = 2):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            Y: Target
            test_size: Test set proportion
            random_state: Random seed
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size, stratify=Y, random_state=random_state
        )
    
    def get_train_test_data(self):
        """Get train/test splits"""
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def save_scaler(self, path: str):
        """Save fitted scaler"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path: str):
        """Load fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
