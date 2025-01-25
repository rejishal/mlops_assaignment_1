# FlaskApplication.py
import pandas as pd
from flask import Flask, request, jsonify
#from Model.DataPreprocessing import load_and_preprocess_data
#from Model.Model_new import train_model, evaluate_model
import mlflow

# Load and preprocess data
#X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train model
#best_model = train_model(X_train, y_train)

# Evaluate model
#accuracy = evaluate_model(best_model, X_test, y_test)
#print(f"Model Accuracy: {accuracy:.2f}")

model_uri = "runs:/70bea5b86bae4029b1df311ba12ef83f/random_forest_model"
best_model = mlflow.pyfunc.load_model(model_uri)

# Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        input_features = pd.DataFrame([data])

        # Make prediction
        prediction = best_model.predict(input_features)
        result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)