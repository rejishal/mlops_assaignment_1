# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the ML Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.json
    features = np.array(data['features']).reshape(1, -1)  # Ensure the input is 2D
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
