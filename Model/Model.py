from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
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

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = train_model(X_train, y_train)
metrics  = evaluate_model(model, X_test, y_test)

 # Print results
print("Model Evaluation Metrics:")
for metric, value in metrics.items():
    if metric == "Confusion Matrix":
        print(f"{metric}:\n{value}")
    else:
        print(f"{metric}: {value:.4f}")