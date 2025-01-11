import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset from DVC-managed file
dataset_path = "data/iris.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"{dataset_path} not found! Did you run `dvc checkout`?")
data = pd.read_csv(dataset_path)

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameters to experiment with
hyperparameters = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 15},
]

# Start MLflow experiment
mlflow.set_experiment("Iris Classification Experiment")

for params in hyperparameters:
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy with params {params}: {accuracy:.2f}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_metric("accuracy", accuracy)

        # Save the model locally
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"rf_model_{params['n_estimators']}_{params['max_depth']}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Log the model and artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
