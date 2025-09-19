from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
import google.generativeai as genai
import re

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel('gemini-1.5-flash')


try:
    model = load_model('youth_wellness_model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


ordinal_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}
sleep_nights_map = {'0-1 night': 0.5, '2-3 nights': 2.5, '4 nights': 4, '5-7 nights': 6}


feature_columns = [
    'Gender_Male', 'Gender_Others', 'Student_Status_Postgraduate',
    'Nervous_Stressed', 'Skip_Meals', 'Problem_Sleep_Nights_Numeric',
    'Age', 'Sleep_Quality', 'Student_Status_Undergraduate', 'Gender_Female'
]


scaler = MinMaxScaler()

def preprocess_user_data(user_data):
    try:
        input_df = pd.DataFrame([user_data])

        
        for col in ['Nervous_Stressed', 'Skip_Meals']:
            if col in input_df.columns:
                input_df[col] = input_df[col].map(ordinal_map).fillna(2)

        
        if 'Problem_Sleep_Nights' in input_df.columns:
            input_df['Problem_Sleep_Nights_Numeric'] = input_df['Problem_Sleep_Nights'].map(sleep_nights_map).fillna(4)

        
        if 'Age' in input_df.columns:
            input_df['Age'] = pd.to_numeric(input_df['Age'], errors='coerce').fillna(25)

        
        if 'Sleep_Quality' in input_df.columns:
            input_df['Sleep_Quality'] = input_df['Sleep_Quality'].map(ordinal_map).fillna(2)

        
        if 'Gender' in input_df.columns:
            input_df['Gender_Male'] = (input_df['Gender'] == 'Male').astype(int)
            input_df['Gender_Female'] = (input_df['Gender'] == 'Female').astype(int)
            input_df['Gender_Others'] = (input_df['Gender'] == 'Others').astype(int)

        
        if 'Student_Status' in input_df.columns:
            input_df['Student_Status_Postgraduate'] = (input_df['Student_Status'] == 'Postgraduate').astype(int)
            input_df['Student_Status_Undergraduate'] = (input_df['Student_Status'] == 'Undergraduate').astype(int)

        
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        
        input_df = input_df[feature_columns]

        
        numerical_features = ['Nervous_Stressed', 'Skip_Meals', 'Problem_Sleep_Nights_Numeric', 'Age', 'Sleep_Quality']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        return input_df
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def predict_stress_level(user_data):
    if model is None:
        return None
    try:
        processed_data = preprocess_user_data(user_data)
        if processed_data is None:
            return None
        prediction = model.predict(processed_data, verbose=0)
        
        return max(0, min(5, prediction[0][0]))
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def interpret_stress_score(score):
    """
    Interpret the stress score according to the 0-5 Sleep Quality & Stress Cycle Scale
    """
    if score >= 4.5:
        return "excellent", "very good sleep, low stress, healthy eating patterns"
    elif score >= 3.5:
        return "good", "overall balance, but mild stress may appear"
    elif score >= 2.5:
        return "acceptable", "moderate stress, average sleep, mild eating changes; needs lifestyle adjustments"
    elif score >= 1.5:
        return "poor", "stress disrupting eating and sleep, reduced concentration, fatigue, early burnout risk"
    else:
        return "very poor", "high stress, severe sleep and eating disruption; urgent intervention needed"

def format_gemini_response(response_text):
    """
    Format the Gemini response for better readability in HTML
    """
   
    response_text = re.sub(r'^Response:\s*', '', response_text)
    
    
    response_text = re.sub(r'(\d+\.)\s+(.*?)(?=\d+\.|$)', r'<br><strong>\1</strong> \2<br>', response_text)
    
    
    response_text = re.sub(r'\*\s+(.*?)(?=\*|$)', r'<br>• \1<br>', response_text)
    
    
    response_text = re.sub(r'\n\n+', '<br><br>', response_text)
    
    
    response_text = response_text.replace('\n', '<br>')
    
    return response_text

def get_gemini_analysis(user_data, stress_score, stress_level):
    try:
        prompt = f"""You are a mental health support assistant. 
        You must interpret the user's predicted score ({stress_score:.2f}) on the Sleep Quality & Stress Cycle Scale.

        Scale meaning:
        5.0 → Excellent ✅ (very good sleep, low stress, healthy eating patterns)  
        4.0 → Good (overall balance, but mild stress may appear)  
        3.0 → Acceptable ✅ (moderate stress, average sleep, mild eating changes; needs lifestyle adjustments)  
        2.0 → Poor ⚠ (stress disrupting eating and sleep, reduced concentration, fatigue, early burnout risk)  
        1.0 → Very Poor 🚨 (high stress, severe sleep and eating disruption; urgent intervention needed)

        This scale represents the cyclic relationship between:
        - Academic Stress
        - Changes in Eating Patterns
        - Lack of Sleep
        which reinforce each other into a Cycle of Stress & Anxiety.

        The user's score is: {stress_score:.2f} ({stress_level})

        User's Profile:
        - Gender: {user_data.get('Gender', 'Not specified')}
        - Student Status: {user_data.get('Student_Status', 'Not specified')}
        - Nervous/Stressed Frequency: {user_data.get('Nervous_Stressed', 'Not specified')}
        - Meal Skipping Frequency: {user_data.get('Skip_Meals', 'Not specified')}
        - Sleep Problems: {user_data.get('Problem_Sleep_Nights', 'Not specified')}
        - Sleep Quality: {user_data.get('Sleep_Quality', 'Not specified')}

        Please:
        1. Explain what this score means in simple words.  
        2. Describe how stress, eating, and sleep might be connected for this score.  
        3. Suggest practical strategies and precautions to break the cycle (stress management, nutrition, sleep hygiene).  
        4. Give empathetic encouragement and motivate the user.  

        Keep the tone supportive, non-judgmental, and concise.

        Response:"""

        response = gemini_model.generate_content(prompt)
        return format_gemini_response(response.text)
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."

def get_gemini_chat_response(user_message, stress_context=None):
    try:
        if stress_context:
            prompt = f"""As a mental health support assistant, you have previously analyzed this user's stress test results. 
            Their stress score was {stress_context['score']:.2f} indicating {stress_context['level']} stress levels.
            
            This score represents their position in the stress-eating-sleep cycle:
            - Academic Stress → Changes in Eating Patterns → Lack of Sleep → (cycle continues)
            
            Now respond to this follow-up question:
            User: {user_message}
            
            Please provide a helpful, empathetic response that considers their stress level and previous assessment.
            Format your response with clear paragraphs and proper spacing for readability.
            Assistant:"""
        else:
            prompt = f"""You are a mental health support assistant. Provide a helpful, empathetic response to this question:
            
            User: {user_message}
            
            Format your response with clear paragraphs and proper spacing for readability.
            Assistant:"""
        
        response = gemini_model.generate_content(prompt)
        return format_gemini_response(response.text)
    except Exception as e:
        print(f"Error getting Gemini chat response: {e}")
        return "I'm sorry, I'm having trouble responding right now. Please try again later."

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mental')
def mental():
    return render_template('mental.html')

@app.route('/stress')
def stress():
    return render_template('stress.html')

@app.route('/chatbot')
def chatbot():
    stress_results = session.get('stress_results', None)
    return render_template('chatbot.html', stress_results=stress_results)

@app.route('/api/save-mental-result', methods=['POST'])
def save_mental_result():
    data = request.get_json()
    session['mental_test_result'] = data
    stress_score = predict_stress_level(data)

    if stress_score is not None:
        stress_level, interpretation = interpret_stress_score(stress_score)
        gemini_analysis = get_gemini_analysis(data, stress_score, stress_level)
        
        session['stress_results'] = {
            'score': float(stress_score),
            'level': stress_level,
            'interpretation': interpretation,
            'gemini_analysis': gemini_analysis,
            'user_data': data
        }
        
        return jsonify({
            'status': 'success', 
            'redirect': url_for('chatbot')
        })

    return jsonify({'status': 'error', 'message': 'Prediction failed'})

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    user_message = data.get('message', '')
    stress_results = session.get('stress_results', None)

    # Get response from Gemini
    gemini_response = get_gemini_chat_response(user_message, stress_results)
    
    return jsonify({'response': gemini_response})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    
    dummy_data = pd.DataFrame({
        'Nervous_Stressed': [0, 1, 2, 3, 4],
        'Skip_Meals': [0, 1, 2, 3, 4],
        'Problem_Sleep_Nights_Numeric': [0.5, 2.5, 4, 6, 6],
        'Age': [18, 22, 25, 30, 35],
        'Sleep_Quality': [0, 1, 2, 3, 4]
    })
    scaler.fit(dummy_data)

    app.run(debug=True, host='0.0.0.0', port=5000)
