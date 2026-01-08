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


@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Chatbot endpoint to handle user queries"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower().strip()
        
        # Knowledge base for chatbot responses
        responses = {
            'what is diabetes': """Diabetes is a chronic health condition where your body cannot properly process glucose (blood sugar). There are two main types:
• Type 1: Body doesn't produce insulin
• Type 2: Body doesn't use insulin properly
It can lead to serious health complications if not managed properly.""",
            
            'diabetes': """Diabetes is a metabolic disorder affecting how your body processes blood sugar. Key symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, and blurred vision. Early detection through prediction and screening is crucial.""",
            
            'how does this prediction work': """Our system uses Machine Learning (Support Vector Machine) trained on medical data to predict diabetes risk. It analyzes 8 key factors:
• Pregnancies
• Glucose level
• Blood pressure
• Skin thickness
• Insulin level
• BMI
• Diabetes Pedigree Function
• Age
The model has been trained on thousands of patient records to provide accurate predictions.""",
            
            'how does it work': """This prediction system uses AI to analyze your medical data and predict diabetes risk. Just click 'Predict', enter your medical information, and get instant results with confidence scores.""",
            
            'prediction': """To make a prediction:
1. Click on 'Predict' in the navigation
2. Enter all required medical measurements
3. Submit the form
4. View your result with confidence score and recommendations
All predictions are saved in your history!""",
            
            'glucose': """Glucose is blood sugar that provides energy to your cells. Normal fasting glucose levels are 70-100 mg/dL. Levels above 126 mg/dL may indicate diabetes. The glucose test in our prediction is a key indicator of diabetes risk.""",
            
            'normal glucose levels': """Normal glucose levels:
• Fasting: 70-100 mg/dL (normal)
• 100-125 mg/dL (prediabetes)
• 126+ mg/dL (diabetes)
• After meals: Less than 140 mg/dL""",
            
            'bmi': """BMI (Body Mass Index) measures body fat based on height and weight:
• Below 18.5: Underweight
• 18.5-24.9: Normal weight
• 25-29.9: Overweight
• 30+: Obese
Higher BMI increases diabetes risk. Calculate: weight(kg) / height(m)²""",
            
            'blood pressure': """Normal blood pressure is 120/80 mmHg. High blood pressure (140/90+) increases diabetes risk and complications. Our prediction uses diastolic blood pressure (the bottom number) as an input factor.""",
            
            'insulin': """Insulin is a hormone that helps cells absorb glucose. In diabetes:
• Type 1: Body doesn't produce insulin
• Type 2: Cells resist insulin
Normal fasting insulin: 2-25 μIU/mL. High insulin may indicate insulin resistance.""",
            
            'symptoms': """Common diabetes symptoms:
• Increased thirst and hunger
• Frequent urination
• Unexplained weight loss
• Fatigue
• Blurred vision
• Slow-healing wounds
• Tingling in hands/feet
If you experience these, consult a doctor.""",
            
            'prevention': """Diabetes prevention tips:
• Maintain healthy weight
• Exercise regularly (30 min/day)
• Eat balanced diet (less sugar, more fiber)
• Monitor blood sugar regularly
• Avoid smoking
• Manage stress
• Regular health check-ups""",
            
            'treatment': """Diabetes treatment includes:
• Type 1: Insulin therapy
• Type 2: Lifestyle changes, oral medications, insulin if needed
• Blood sugar monitoring
• Healthy diet
• Regular exercise
• Weight management
Always consult healthcare professionals for personalized treatment.""",
            
            'history': """To view your prediction history:
1. Click 'History' in the navigation
2. See all past predictions with dates
3. View statistics and trends
4. Track your health progress over time""",
            
            'accuracy': """Our model is trained on the Pima Indians Diabetes Database with thousands of medical records. The system provides a confidence score with each prediction. However, always consult healthcare professionals for medical advice.""",
            
            'help': """I can help you with:
• Understanding diabetes and its symptoms
• How our prediction system works
• Normal ranges for medical measurements
• Prevention and treatment information
• Navigating the website
• Viewing prediction history
Just ask me anything!""",
            
            'hi': """Hello! 👋 I'm here to help you with diabetes information and guide you through making predictions. What would you like to know?""",
            
            'hello': """Hi there! 👋 How can I assist you today? I can help with diabetes questions or guide you through the prediction process.""",
            
            'thank': """You're welcome! If you have any more questions, feel free to ask. Stay healthy! 😊""",
        }
        
        # Find best matching response
        response = None
        for key, value in responses.items():
            if key in user_message:
                response = value
                break
        
        # Default response if no match found
        if not response:
            if '?' in user_message:
                response = """I'm here to help! You can ask me about:
• What is diabetes and its symptoms
• How our prediction system works
• Normal glucose, BMI, blood pressure levels
• Prevention and treatment tips
• How to make a prediction
• Your prediction history

Try asking one of these questions!"""
            else:
                response = "I'm not sure I understand. Could you rephrase your question? I can help with diabetes information, prediction guidance, and health tips."
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("\n" + "="*60)
    print("DIABETES PREDICTION WEB APPLICATION")
    print("="*60)
    print(f"\nServer starting on port {port}...")
    if debug:
        print("Open browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
