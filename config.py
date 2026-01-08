"""Configuration and Settings"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
NOTEBOOK_DIR = PROJECT_ROOT / 'notebooks'
SRC_DIR = PROJECT_ROOT / 'src'

# Data settings
DATA_FILE = PROJECT_ROOT / 'diabetes.csv'
TARGET_COLUMN = 'Outcome'
TEST_SIZE = 0.2
RANDOM_STATE = 2

# Model settings
MODEL_TYPE = 'SVM'
SVM_KERNEL = 'linear'
MODEL_FILE = MODEL_DIR / 'diabetes_model.pkl'
SCALER_FILE = MODEL_DIR / 'scaler.pkl'

# Feature names (in order)
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# Class mapping
CLASS_LABELS = {
    0: 'Non-Diabetic',
    1: 'Diabetic'
}

# Create necessary directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
