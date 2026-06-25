import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

app = Flask(__name__)

MODEL_PATH = 'models/intrusion_model.pkl'
DATA_PATH = 'data/network_traffic.csv'

# Ensure directories exist
os.makedirs('models', exist_ok=True)

def train_and_save_model():
    print("🔄 Training model pipeline...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Please run generate_data.py first to construct the dataset.")
        
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=['intrusion_type'])
    y = df['intrusion_type']
    
    # Define preprocessing steps
    numeric_features = ['duration', 'packet_size', 'source_ip_patterns', 'port_number']
    categorical_features = ['protocol']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Bundle preprocessing and Random Forest classifier into a production Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data to generate metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    # Generate Evaluation Metrics for Dashboard
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model_pipeline.classes_.tolist()).tolist()
    
    # Save the pipeline along with metrics metadata
    payload = {
        'pipeline': model_pipeline,
        'classes': model_pipeline.classes_.tolist(),
        'accuracy': round(acc * 100, 2),
        'confusion_matrix': cm
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print("✅ Model trained and saved successfully.")

# Initialize or load model
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

with open(MODEL_PATH, 'rb') as f:
    saved_meta = pickle.load(f)
    model_pipeline = saved_meta['pipeline']

# --- FLASK WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read parameters from the form submission
        input_data = pd.DataFrame([{
            'duration': float(request.form['duration']),
            'protocol': request.form['protocol'],
            'packet_size': int(request.form['packet_size']),
            'source_ip_patterns': int(request.form['source_ip_patterns']),
            'port_number': int(request.form['port_number'])
        }])
        
        prediction = model_pipeline.predict(input_data)[0]
        probabilities = model_pipeline.predict_proba(input_data)[0]
        max_prob = round(np.max(probabilities) * 100, 2)
        
        return render_template('index.html', 
                               prediction=prediction, 
                               confidence=max_prob,
                               submitted=True)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    # Feeds accuracy and matrix metadata directly into the frontend interface
    return render_template('dashboard.html', 
                           accuracy=saved_meta['accuracy'], 
                           matrix=saved_meta['confusion_matrix'],
                           classes=saved_meta['classes'])

if __name__ == '__main__':
    app.run(debug=True)