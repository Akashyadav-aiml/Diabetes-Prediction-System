"""Flask Web Application for Diabetes Prediction"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
from pathlib import Path
from src.model import DiabetesClassifier
from src.preprocessor import DataPreprocessor
from src.predictor import DiabetesPredictor
from database import PredictionDatabase


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes_prediction_secret_key'

# Add min function to Jinja2
app.jinja_env.globals.update(min=min)

# Initialize database
db = PredictionDatabase()

# Load trained model
def load_model():
    """Load trained model and scaler"""
    project_root = Path(__file__).parent
    model_path = project_root / 'models' / 'diabetes_model.pkl'
    scaler_path = project_root / 'models' / 'scaler.pkl'
    
    if not model_path.exists() or not scaler_path.exists():
        return None, None, None
    
    classifier = DiabetesClassifier()
    classifier.load_model(str(model_path))
    
    preprocessor = DataPreprocessor()
    preprocessor.load_scaler(str(scaler_path))
    
    predictor = DiabetesPredictor(classifier.classifier, preprocessor.scaler)
    
    return classifier, preprocessor, predictor


# Load model on startup
classifier, preprocessor, predictor = load_model()

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

FEATURE_KEYS = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
]


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        if classifier is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please run train.py first.'
            }), 400
        
        try:
            # Get form data
            patient_data = {
                'pregnancies': float(request.form.get('pregnancies', 0)),
                'glucose': float(request.form.get('glucose', 0)),
                'blood_pressure': float(request.form.get('blood_pressure', 0)),
                'skin_thickness': float(request.form.get('skin_thickness', 0)),
                'insulin': float(request.form.get('insulin', 0)),
                'bmi': float(request.form.get('bmi', 0)),
                'diabetes_pedigree_function': float(request.form.get('diabetes_pedigree_function', 0)),
                'age': float(request.form.get('age', 0))
            }
            
            # Validate data
            for key, value in patient_data.items():
                if value < 0:
                    return jsonify({
                        'success': False,
                        'error': f'{key} cannot be negative'
                    }), 400
            
            # Make prediction
            patient_tuple = tuple(patient_data.values())
            result = predictor.predict_single(patient_tuple)
            
            # Prepare data for database
            db_data = {
                'pregnancies': patient_data['pregnancies'],
                'glucose': patient_data['glucose'],
                'blood_pressure': patient_data['blood_pressure'],
                'skin_thickness': patient_data['skin_thickness'],
                'insulin': patient_data['insulin'],
                'bmi': patient_data['bmi'],
                'diabetes_pedigree_function': patient_data['diabetes_pedigree_function'],
                'age': patient_data['age'],
                'prediction': result['prediction'],
                'prediction_text': result['prediction_text'],
                'confidence': result['confidence_score']
            }
            
            # Save to database
            record_id = db.save_prediction(db_data)
            
            return jsonify({
                'success': True,
                'record_id': record_id,
                'prediction': result['prediction'],
                'prediction_text': result['prediction_text'],
                'confidence_score': round(result['confidence_score'], 4),
                'patient_data': patient_data,
                'message': f"Patient is {result['prediction_text']}"
            })
        
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid input: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('predict.html', features=FEATURE_NAMES)


@app.route('/results/<int:record_id>')
def results(record_id):
    """View prediction results"""
    prediction = db.get_prediction(record_id)
    
    if not prediction:
        return redirect(url_for('history'))
    
    return render_template('results.html', prediction=prediction)


@app.route('/history')
def history():
    """View prediction history"""
    predictions = db.get_recent_predictions(limit=50)
    stats = db.get_statistics()
    
    return render_template('history.html', 
                         predictions=predictions,
                         stats=stats)


@app.route('/api/statistics')
def api_statistics():
    """API endpoint for statistics"""
    stats = db.get_statistics()
    return jsonify(stats)


@app.route('/api/predictions')
def api_predictions():
    """API endpoint for all predictions"""
    predictions = db.get_all_predictions(limit=100)
    return jsonify(predictions)


@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    """Delete a prediction record"""
    if db.delete_prediction(record_id):
        return jsonify({'success': True, 'message': 'Record deleted'})
    return jsonify({'success': False, 'message': 'Record not found'}), 404


@app.route('/feedback/<int:record_id>', methods=['POST'])
def save_feedback(record_id):
    """Save user feedback"""
    try:
        actual_outcome = int(request.form.get('actual_outcome', 0))
        feedback_text = request.form.get('feedback_text', '')
        
        db.save_feedback(record_id, actual_outcome, feedback_text)
        
        return jsonify({
            'success': True,
            'message': 'Feedback saved successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DIABETES PREDICTION WEB APPLICATION")
    print("="*60)
    print("\nServer starting...")
    print("Open browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
