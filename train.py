"""Main Training Pipeline"""

import os
from pathlib import Path
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import DiabetesClassifier


def main():
    """Execute the complete training pipeline"""
    
    # Setup paths
    project_root = Path(__file__).parent
    data_path = project_root / 'diabetes.csv'
    model_path = project_root / 'models' / 'diabetes_model.pkl'
    scaler_path = project_root / 'models' / 'scaler.pkl'
    
    print("=" * 60)
    print("DIABETES PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n[1/5] Loading data...")
    loader = DataLoader(str(data_path))
    df = loader.load_data()
    summary = loader.get_summary()
    
    print(f"  ✓ Data loaded: {summary['shape'][0]} records, {summary['shape'][1]} features")
    
    # Step 2: Data Analysis
    print("\n[2/5] Analyzing data...")
    class_dist = loader.get_class_distribution()
    print(f"  ✓ Class distribution: Non-Diabetic={class_dist[0]}, Diabetic={class_dist[1]}")
    
    feature_stats = loader.get_feature_statistics_by_class()
    print("  ✓ Feature statistics by class calculated")
    
    # Step 3: Preprocessing
    print("\n[3/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, Y = preprocessor.separate_features_target(df)
    X_standardized = preprocessor.standardize_data(X, fit=True)
    preprocessor.train_test_split_data(X_standardized, Y)
    X_train, X_test, Y_train, Y_test = preprocessor.get_train_test_data()
    
    print(f"  ✓ Data standardized")
    print(f"  ✓ Train set: {X_train.shape[0]} samples")
    print(f"  ✓ Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train Model
    print("\n[4/5] Training SVM classifier...")
    classifier = DiabetesClassifier(kernel='linear')
    classifier.train(X_train, Y_train)
    print("  ✓ Model training completed")
    
    # Step 5: Evaluate Model
    print("\n[5/5] Evaluating model...")
    train_metrics = classifier.evaluate(X_train, Y_train, "Training")
    test_metrics = classifier.evaluate(X_test, Y_test, "Test")
    
    print(f"  Training Set Metrics:")
    print(f"    - Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"    - Precision: {train_metrics['precision']:.4f}")
    print(f"    - Recall:    {train_metrics['recall']:.4f}")
    print(f"    - F1-Score:  {train_metrics['f1']:.4f}")
    
    print(f"\n  Test Set Metrics:")
    print(f"    - Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    - Precision: {test_metrics['precision']:.4f}")
    print(f"    - Recall:    {test_metrics['recall']:.4f}")
    print(f"    - F1-Score:  {test_metrics['f1']:.4f}")
    
    # Save Models
    print("\n[SAVING] Persisting model and scaler...")
    os.makedirs(model_path.parent, exist_ok=True)
    classifier.save_model(str(model_path))
    preprocessor.save_scaler(str(scaler_path))
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return classifier, preprocessor


if __name__ == "__main__":
    main()
