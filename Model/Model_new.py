import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from DataPreprocessing import load_and_preprocess_data

load_and_preprocess_data()

def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    param_grid = {
    'n_estimators': [10, 50, 150, 300],  # Includes a smaller value and a higher one
    'max_depth': [5, 15, 25, None],     # Added a smaller depth for comparison
    'min_samples_split': [3, 6, 12],    # Modified split values
    'min_samples_leaf': [1, 3, 5]       # Slightly larger range for leaf size
    }


    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def run_pipeline():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Start MLflow experiment
    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 3)
        
        # Train model
        model = train_model(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Evaluate model and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):  # Log scalar values
                mlflow.log_metric(metric, value)
            else:  # Log confusion matrix as artifact
                confusion_matrix_file = "confusion_matrix.txt"
                with open(confusion_matrix_file, "w") as f:
                    f.write(str(value))
                mlflow.log_artifact(confusion_matrix_file)

        # Print evaluation metrics
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            if metric == "Confusion Matrix":
                print(f"{metric}:\n{value}")
            else:
                print(f"{metric}: {value:.4f}")

# Trigger the pipeline run
if __name__ == "__main__":
    run_pipeline()