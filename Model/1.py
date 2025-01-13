# train_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Features
y = 2.5 * X + np.random.randn(100, 1)  # Target with noise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'linear_model.pkl'")
