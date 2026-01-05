"""SQLite Database Module for Prediction Records"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class PredictionDatabase:
    """Handle SQLite database for prediction records"""
    
    def __init__(self, db_path: str = 'database_dir/diabetes_predictions.db'):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pregnancies REAL,
                glucose REAL,
                blood_pressure REAL,
                skin_thickness REAL,
                insulin REAL,
                bmi REAL,
                diabetes_pedigree_function REAL,
                age REAL,
                prediction INTEGER,
                prediction_text TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                actual_outcome INTEGER,
                feedback_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, patient_data: Dict) -> int:
        """
        Save prediction to database
        
        Args:
            patient_data: Dictionary with prediction data
                {
                    'pregnancies': float,
                    'glucose': float,
                    'blood_pressure': float,
                    'skin_thickness': float,
                    'insulin': float,
                    'bmi': float,
                    'diabetes_pedigree_function': float,
                    'age': float,
                    'prediction': int (0 or 1),
                    'prediction_text': str,
                    'confidence': float
                }
        
        Returns:
            int: ID of inserted record
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree_function, age,
                prediction, prediction_text, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data['pregnancies'],
            patient_data['glucose'],
            patient_data['blood_pressure'],
            patient_data['skin_thickness'],
            patient_data['insulin'],
            patient_data['bmi'],
            patient_data['diabetes_pedigree_function'],
            patient_data['age'],
            patient_data['prediction'],
            patient_data['prediction_text'],
            patient_data['confidence']
        ))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        return record_id
    
    def get_prediction(self, record_id: int) -> Dict:
        """Get prediction by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions WHERE id = ?', (record_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_predictions(self, limit: int = 100) -> List[Dict]:
        """Get all predictions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total FROM predictions')
        total = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as diabetic FROM predictions WHERE prediction = 1')
        diabetic = cursor.fetchone()['diabetic']
        
        cursor.execute('SELECT COUNT(*) as non_diabetic FROM predictions WHERE prediction = 0')
        non_diabetic = cursor.fetchone()['non_diabetic']
        
        cursor.execute('SELECT AVG(confidence) as avg_confidence FROM predictions')
        avg_confidence = cursor.fetchone()['avg_confidence'] or 0
        
        conn.close()
        
        return {
            'total_predictions': total,
            'diabetic_count': diabetic,
            'non_diabetic_count': non_diabetic,
            'diabetic_percentage': (diabetic / total * 100) if total > 0 else 0,
            'average_confidence': avg_confidence
        }
    
    def save_feedback(self, prediction_id: int, actual_outcome: int, feedback_text: str = ""):
        """
        Save user feedback on prediction
        
        Args:
            prediction_id: ID of prediction
            actual_outcome: Actual outcome (0 or 1)
            feedback_text: Optional feedback text
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (prediction_id, actual_outcome, feedback_text)
            VALUES (?, ?, ?)
        ''', (prediction_id, actual_outcome, feedback_text))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """Get recent predictions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def delete_prediction(self, record_id: int) -> bool:
        """Delete prediction record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM feedback WHERE prediction_id = ?', (record_id,))
        cursor.execute('DELETE FROM predictions WHERE id = ?', (record_id,))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
