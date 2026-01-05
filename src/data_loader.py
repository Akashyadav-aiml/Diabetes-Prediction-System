"""Data Loading and Exploration Module"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Handle data loading and initial exploration"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def get_summary(self) -> dict:
        """Get data summary statistics"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistical_summary': self.data.describe().to_dict()
        }
    
    def get_class_distribution(self, target_column: str = 'Outcome') -> dict:
        """Get class distribution"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.data[target_column].value_counts().to_dict()
    
    def get_feature_statistics_by_class(self, target_column: str = 'Outcome') -> pd.DataFrame:
        """Get mean values of features grouped by target variable"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.data.groupby(target_column).mean()
