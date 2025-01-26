import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(input_file="data/raw_data/raw_data.csv", output_file="data/preprocessed_data/processed_data.csv"):
    #data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    #dataset = pd.read_csv(input_file, header=None, na_values="?")
    dataset = pd.read_csv(input_file)
    dataset.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    dataset = dataset.dropna()

    X = dataset.drop("target", axis=1)
    y = dataset["target"].apply(lambda x: 1 if x > 0 else 0)  # Binary classification
    print(output_file)
    dataset.to_csv(output_file, index=False)

    return train_test_split(X, y, test_size=0.2, random_state=42)

load_and_preprocess_data()