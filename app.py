from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

app = Flask(__name__)

# Serve static files
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/styles.css')
def styles():
    response = app.send_static_file('styles.css')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/script.js')
def script():
    response = app.send_static_file('script.js')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# API endpoint for predictions (optional - for future integration with actual model)
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Convert frontend data to model format
        features = {
            'Time_spent_Alone': float(data.get('timeAlone', 0)),
            'Stage_fear': 1 if data.get('stageFear') == 'Yes' else 0,
            'Social_event_attendance': int(data.get('socialEvents', 0)),
            'Going_outside': int(data.get('goingOutside', 1)),
            'Drained_after_socializing': 1 if data.get('drainedAfterSocializing') == 'Yes' else 0,
            'Friends_circle_size': int(data.get('friendsCircle', 1)),
            'Post_frequency': int(data.get('postFrequency', 0))
        }
        
        # For now, return a simple prediction based on the frontend logic
        # In the future, this could load the actual trained model
        prediction = simple_predict(features)
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def simple_predict(features):
    """Simple prediction logic matching the frontend"""
    
    # Calculate derived features (matching the model's feature engineering)
    social_activity_score = features['Social_event_attendance'] + features['Going_outside'] + features['Post_frequency']
    social_energy = features['Time_spent_Alone'] * 0.5 + features['Stage_fear'] * 2 + features['Drained_after_socializing'] * 2
    introversion_score = features['Time_spent_Alone'] * 0.3 + features['Stage_fear'] * 1.5 + features['Drained_after_socializing'] * 1.5 - features['Social_event_attendance'] * 0.2
    extroversion_score = features['Social_event_attendance'] * 0.3 + features['Going_outside'] * 0.3 + features['Post_frequency'] * 0.2 + features['Friends_circle_size'] * 0.2 - features['Time_spent_Alone'] * 0.1
    
    # Simple prediction logic
    introversion_indicators = [
        features['Time_spent_Alone'] > 8,
        features['Stage_fear'] == 1,
        features['Drained_after_socializing'] == 1,
        features['Social_event_attendance'] < 2,
        features['Going_outside'] <= 2,
        features['Friends_circle_size'] <= 2,
        features['Post_frequency'] < 3
    ]
    
    extroversion_indicators = [
        features['Time_spent_Alone'] < 4,
        features['Stage_fear'] == 0,
        features['Drained_after_socializing'] == 0,
        features['Social_event_attendance'] > 4,
        features['Going_outside'] >= 3,
        features['Friends_circle_size'] >= 3,
        features['Post_frequency'] > 5
    ]
    
    introversion_count = sum(introversion_indicators)
    extroversion_count = sum(extroversion_indicators)
    
    # Determine personality type
    if introversion_score > extroversion_score or introversion_count > extroversion_count:
        personality_type = 'Introvert'
        confidence = max((introversion_count / len(introversion_indicators)) * 100, 60)
    else:
        personality_type = 'Extrovert'
        confidence = max((extroversion_count / len(extroversion_indicators)) * 100, 60)
    
    return {
        'personalityType': personality_type,
        'confidence': min(confidence, 95),
        'features': features
    }

if __name__ == '__main__':
    # Move static files to the correct location for Flask
    import shutil
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Copy files to static directory
    files_to_copy = ['index.html', 'styles.css', 'script.js']
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, f'static/{file}')
    
    print("🚀 Starting Personality Predictor Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("✨ The frontend is now running!")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 