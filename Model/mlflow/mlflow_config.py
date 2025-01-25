import mlflow
import mlflow.sklearn

# Start tracking
mlflow.set_tracking_uri("file:///C:\\Users\\rejis\\OneDrive - wilp.bits-pilani.ac.in\\bits\\mlops\\ass1\\MLFlow")

print(mlflow.get_tracking_uri())

mlflow.set_experiment("my_experiment_name")
print(mlflow.get_experiment_by_name("my_experiment_name"))



# Log an example run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_artifact(r"C:\Users\rejis\OneDrive - wilp.bits-pilani.ac.in\bits\mlops\ass1\mlops_assaignment_1\linear_model.pkl")

