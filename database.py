#!/usr/bin/env python3
"""
Database module for ThyroCheck health history storage
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

class HealthDatabase:
    def __init__(self, db_path: str = 'thyrocheck.db'):
        """Initialize database connection"""
        # Ensure the database is created in the backend directory
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(__file__), db_path)
        self.db_path = db_path
        print(f"Database path: {self.db_path}")
        self.init_database()

    def init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Health assessments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    score INTEGER NOT NULL,
                    health_status TEXT NOT NULL,
                    symptom_details TEXT,  -- JSON string
                    answers TEXT,  -- JSON string of questionnaire answers
                    ml_prediction TEXT,  -- JSON string of ML prediction results
                    recommendations TEXT,  -- JSON string of recommendations
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # ML Predictions table (for tracking model performance)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assessment_id INTEGER NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL,
                    model_version TEXT,
                    input_features TEXT,  -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (assessment_id) REFERENCES health_assessments (id) ON DELETE CASCADE
                )
            ''')

            conn.commit()

    def create_user(self, name: str, email: str, password_hash: str) -> int:
        """Create a new user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
            ''', (name, email, password_hash))
            conn.commit()
            return cursor.lastrowid

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'password_hash': row[3],
                    'created_at': row[4],
                    'updated_at': row[5]
                }
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'password_hash': row[3],
                    'created_at': row[4],
                    'updated_at': row[5]
                }
            return None

    def save_health_assessment(self, user_id: int, assessment_data: Dict[str, Any]) -> int:
        """Save a health assessment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO health_assessments
                (user_id, score, health_status, symptom_details, answers, recommendations)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                assessment_data['score'],
                assessment_data['health_status'],
                json.dumps(assessment_data.get('symptom_details', [])),
                json.dumps(assessment_data.get('answers', {})),
                json.dumps(assessment_data.get('recommendations', []))
            ))

            assessment_id = cursor.lastrowid
            conn.commit()
            return assessment_id

    def save_ml_prediction(self, assessment_id: int, prediction_data: Dict[str, Any]):
        """Save ML prediction results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO ml_predictions
                (assessment_id, prediction, confidence, model_version, input_features)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                assessment_id,
                prediction_data.get('prediction', ''),
                prediction_data.get('confidence', 0),
                prediction_data.get('model_version', '1.0'),
                json.dumps(prediction_data.get('input_features', {}))
            ))

            conn.commit()

    def get_user_health_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's health assessment history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    ha.id,
                    ha.assessment_date,
                    ha.score,
                    ha.health_status,
                    ha.symptom_details,
                    ha.answers,
                    ha.recommendations,
                    mp.prediction,
                    mp.confidence
                FROM health_assessments ha
                LEFT JOIN ml_predictions mp ON ha.id = mp.assessment_id
                WHERE ha.user_id = ?
                ORDER BY ha.assessment_date DESC
                LIMIT ?
            ''', (user_id, limit))

            assessments = []
            for row in cursor.fetchall():
                assessments.append({
                    'id': row[0],
                    'assessment_date': row[1],
                    'score': row[2],
                    'health_status': row[3],
                    'symptom_details': json.loads(row[4]) if row[4] else [],
                    'answers': json.loads(row[5]) if row[5] else {},
                    'recommendations': json.loads(row[6]) if row[6] else [],
                    'ml_prediction': row[7],
                    'ml_confidence': row[8]
                })

            return assessments

    def get_health_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get health statistics for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get assessment count and average score
            cursor.execute('''
                SELECT
                    COUNT(*) as total_assessments,
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score,
                    MAX(assessment_date) as last_assessment
                FROM health_assessments
                WHERE user_id = ?
            ''', (user_id,))

            row = cursor.fetchone()
            if not row or row[0] == 0:
                return {
                    'total_assessments': 0,
                    'average_score': 0,
                    'min_score': 0,
                    'max_score': 0,
                    'last_assessment': None,
                    'score_trend': [],
                    'common_symptoms': []
                }

            # Get score trend (last 10 assessments)
            cursor.execute('''
                SELECT score, assessment_date
                FROM health_assessments
                WHERE user_id = ?
                ORDER BY assessment_date DESC
                LIMIT 10
            ''', (user_id,))

            score_trend = []
            for score_row in cursor.fetchall():
                score_trend.append({
                    'score': score_row[0],
                    'date': score_row[1]
                })

            # Get common symptoms
            cursor.execute('''
                SELECT symptom_details
                FROM health_assessments
                WHERE user_id = ? AND symptom_details IS NOT NULL
                ORDER BY assessment_date DESC
                LIMIT 20
            ''', (user_id,))

            symptom_counts = {}
            for symptom_row in cursor.fetchall():
                symptoms = json.loads(symptom_row[0]) if symptom_row[0] else []
                for symptom in symptoms:
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1

            common_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                'total_assessments': row[0],
                'average_score': round(row[1], 1) if row[1] else 0,
                'min_score': row[2] or 0,
                'max_score': row[3] or 0,
                'last_assessment': row[4],
                'score_trend': score_trend,
                'common_symptoms': [{'symptom': s[0], 'count': s[1]} for s in common_symptoms]
            }

    def update_user(self, user_id: int, updates: Dict[str, Any]):
        """Update user information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            update_fields = []
            values = []

            for field, value in updates.items():
                if field in ['name', 'email', 'password_hash']:
                    update_fields.append(f"{field} = ?")
                    values.append(value)

            if update_fields:
                values.append(user_id)
                cursor.execute(f'''
                    UPDATE users
                    SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', values)
                conn.commit()

# Global database instance
db = HealthDatabase()

def get_database():
    """Get database instance"""
    return db

if __name__ == "__main__":
    # Test the database
    db = HealthDatabase()

    # Create a test user
    user_id = db.create_user("Test User", "test@example.com", "hashed_password")
    print(f"Created user with ID: {user_id}")

    # Save a test assessment
    assessment_data = {
        'score': 8,
        'health_status': 'Mild Thyroid Symptoms',
        'symptom_details': ['fatigue', 'weight_changes'],
        'answers': {'tiredness': 'often', 'sleep': 'mild'},
        'recommendations': ['Track symptoms', 'Consult doctor']
    }

    assessment_id = db.save_health_assessment(user_id, assessment_data)
    print(f"Saved assessment with ID: {assessment_id}")

    # Save ML prediction
    prediction_data = {
        'prediction': 'hypothyroid',
        'confidence': 0.85,
        'model_version': '1.0',
        'input_features': {'age': 30, 'sex': 'F'}
    }

    db.save_ml_prediction(assessment_id, prediction_data)

    # Get user history
    history = db.get_user_health_history(user_id)
    print(f"User has {len(history)} assessments")

    # Get statistics
    stats = db.get_health_statistics(user_id)
    print(f"User statistics: {stats}")

    print("Database test completed successfully!")