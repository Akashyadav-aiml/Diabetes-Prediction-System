
# 🏥 Diabetes Prediction System

Predict diabetes risk instantly using 8 medical indicators. Powered by Flask, SQLite3, and an SVM model.

## 🚀 Quick Start

1. Install Python 3.7+ and pip
2. Clone this repo and enter the folder
3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
4. Train the model (first time only):
  ```bash
  python train.py
  ```
5. Start the app:
  ```bash
  python app.py
  ```
6. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## ✨ Features

- Enter 8 medical values, get instant diabetes prediction & confidence
- All predictions saved in local database
- View history, stats, and details for each prediction
- Responsive web UI, works on all devices

## 🩺 Medical Indicators

Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## 🛠️ Tech Stack

Python, Flask, scikit-learn, pandas, numpy, SQLite3, HTML/CSS/JS

## 📁 Structure

- app.py, train.py, database.py, config.py, utils.py
- src/: ML pipeline (data_loader, preprocessor, model, predictor)
- models/: Trained model & scaler
- templates/: HTML pages
- static/: CSS
- database_dir/: SQLite DB

## ⚠️ Disclaimer

For education/demo only. Not for real medical use.

## 📚 License

MIT License. (c) 2026 Akash

