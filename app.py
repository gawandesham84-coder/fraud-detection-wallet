from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this for production!

# Global variables
model = None
scaler = None

def load_model():
    """Load your trained fraud detection model"""
    global model
    try:
        model_path = os.path.join('model', 'fraud_model.pkl')
        model = joblib.load(model_path)
        print("✅ Fraud detection model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model when app starts
model_loaded = load_model()

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        flash('Model not loaded. Please check the model file.', 'error')
        return redirect(url_for('home'))
    
    try:
        # Get form data
        form_data = request.form
        
        # Extract features - adjust based on your model's features
        # The creditcard.csv typically has V1-V28, Time, Amount
        features = []
        
        # Add V1-V28 features
        for i in range(1, 29):
            feature_name = f'V{i}'
            features.append(float(form_data.get(feature_name, 0)))
        
        # Add Time and Amount (these need scaling like in your training)
        time_val = float(form_data.get('Time', 0))
        amount_val = float(form_data.get('Amount', 0))
        
        # For demo, we'll use placeholder scaling - you should use the same scaler from training
        # In production, you should save and load your scaler
        time_scaled = (time_val - 0) / 1  # Replace with actual scaling
        amount_scaled = (amount_val - 0) / 1  # Replace with actual scaling
        
        features.extend([time_scaled, amount_scaled])
        
        # Create feature array
        feature_array = np.array([features])
        
        # Make prediction
        probability = model.predict_proba(feature_array)[0][1]  # Probability of fraud
        prediction = model.predict(feature_array)[0]
        
        # Format results
        risk_level = "HIGH RISK" if probability > 0.7 else "MEDIUM RISK" if probability > 0.3 else "LOW RISK"
        
        result_data = {
            'fraud_probability': f"{probability:.4f}",
            'prediction': "FRAUD" if prediction == 1 else "LEGITIMATE",
            'risk_level': risk_level,
            'confidence': f"{(max(probability, 1-probability)):.2%}",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'features_used': len(features)
        }
        
        return render_template('result.html', result=result_data)
    
    except Exception as e:
        flash(f'Error processing request: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/batch_analyze', methods=['GET', 'POST'])
def batch_analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Read and process the CSV file
                df = pd.read_csv(file)
                
                # Validate columns (should match your model's expected features)
                expected_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
                
                if not all(col in df.columns for col in expected_columns):
                    flash('CSV file must contain columns: V1-V28, Time, Amount', 'error')
                    return redirect(request.url)
                
                # Process each transaction
                results = []
                for idx, row in df.iterrows():
                    features = row[expected_columns].values.reshape(1, -1)
                    probability = model.predict_proba(features)[0][1]
                    prediction = model.predict(features)[0]
                    
                    results.append({
                        'transaction_id': idx + 1,
                        'amount': row['Amount'],
                        'fraud_probability': probability,
                        'prediction': "FRAUD" if prediction == 1 else "LEGITIMATE",
                        'risk_level': "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
                    })
                
                # Summary statistics
                fraud_count = sum(1 for r in results if r['prediction'] == 'FRAUD')
                total_amount = sum(r['amount'] for r in results if r['prediction'] == 'FRAUD')
                
                summary = {
                    'total_transactions': len(results),
                    'fraudulent_count': fraud_count,
                    'fraud_percentage': f"{(fraud_count/len(results))*100:.2f}%",
                    'total_fraud_amount': f"${total_amount:,.2f}",
                    'average_risk_score': f"{sum(r['fraud_probability'] for r in results)/len(results):.4f}"
                }
                
                return render_template('analyze.html', results=results, summary=summary)
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
    
    return render_template('analyze.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        # Extract features from JSON
        features = []
        for i in range(1, 29):
            features.append(float(data.get(f'V{i}', 0)))
        
        features.extend([float(data.get('Time', 0)), float(data.get('Amount', 0))])
        feature_array = np.array([features])
        
        probability = model.predict_proba(feature_array)[0][1]
        prediction = model.predict(feature_array)[0]
        
        return jsonify({
            "fraud_probability": float(probability),
            "prediction": "fraud" if prediction == 1 else "legitimate",
            "risk_level": "high" if probability > 0.7 else "medium" if probability > 0.3 else "low",
            "confidence": float(max(probability, 1-probability)),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy" if model_loaded else "error",
        "model_loaded": model_loaded,
        "service": "fraud-detection-api",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # For production, use: waitress.serve(app, host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000)