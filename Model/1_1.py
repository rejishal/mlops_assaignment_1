# train_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import mlflow
import mlflow.sklearn

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Features
y = 2.6 * X + np.random.randn(100, 1)  # Target with noise

# Split the data
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)




#mlflow.set_tracking_uri("file:///C:\\Users\\rejis\\OneDrive - wilp.bits-pilani.ac.in\\bits\\mlops\\ass1\\MLFlow")
#print(mlflow.get_tracking_uri())

# Check if the experiment exists, create if not
#experiment_name = "my_experiment_name"
#mlflow.set_experiment(experiment_name)
#experiment = mlflow.get_experiment_by_name(experiment_name)
#print(f"Using experiment: {experiment}")

# Start MLflow tracking
with mlflow.start_run():
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("model_type", "LinearRegression")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    # Ensure the correct artifact path
    artifact_path = "linear_model.pkl"
    with open(artifact_path, 'wb') as f:
        pickle.dump(model, f)
    mlflow.log_artifact(artifact_path)
    print("artifact_path:",artifact_path)
print("MLflow tracking complete. Run logged.")