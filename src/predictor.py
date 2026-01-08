"""Prediction Module for Single Instances"""

import numpy as np
from typing import Union, List


class DiabetesPredictor:
    """Make predictions for individual patients"""
    
    def __init__(self, classifier, scaler):
        """
        Initialize predictor
        
        Args:
            classifier: Trained classifier model
            scaler: Fitted StandardScaler
        """
        self.classifier = classifier
        self.scaler = scaler
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    
    def predict_single(self, patient_data: Union[tuple, list]) -> dict:
        """
        Predict for a single patient
        
        Args:
            patient_data: Tuple/list of 8 features
            
        Returns:
            dict: Prediction result and probability
        """
        if len(patient_data) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(patient_data)}")
        
        # Convert to numpy array and reshape
        input_array = np.asarray(patient_data)
        input_reshaped = input_array.reshape(1, -1)
        
        # Standardize
        standardized_data = self.scaler.transform(input_reshaped)
        
        # Predict
        prediction = self.classifier.predict(standardized_data)[0]
        
        # Get decision function (distance from hyperplane)
        decision_function = self.classifier.decision_function(standardized_data)[0]
        
        return {
            'prediction': int(prediction),
            'is_diabetic': bool(prediction),
            'prediction_text': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'confidence_score': abs(decision_function),
            'patient_data': dict(zip(self.feature_names, patient_data))
        }
    
    def predict_batch(self, patient_data_list: List[Union[tuple, list]]) -> list:
        """
        Predict for multiple patients
        
        Args:
            patient_data_list: List of tuples/lists with patient data
            
        Returns:
            list: List of prediction results
        """
        results = []
        for patient_data in patient_data_list:
            results.append(self.predict_single(patient_data))
        return results
    
    def get_feature_names(self) -> list:
        """Get feature names"""
        return self.feature_names
