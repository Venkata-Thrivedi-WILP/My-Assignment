import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature


# Load dataset from DVC-managed file
dataset_path = "data/iris.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"{dataset_path} not found!")
data = pd.read_csv(dataset_path)


# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define the objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
    max_depth = trial.suggest_int("max_depth", 5, 20)

    # Train model with the suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# Start MLflow experiment
mlflow.set_experiment("Iris Classification Experiment")


# Start Optuna study for hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)


# Get the best hyperparameters from Optuna
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")


# Train the model with the best hyperparameters
best_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    random_state=42,
)
best_model.fit(X_train, y_train)


# Evaluate the best model
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Best Model Accuracy: {:.2f}".format(accuracy))


# Log the best hyperparameters and accuracy to MLflow
mlflow.log_param("n_estimators", best_params["n_estimators"])
mlflow.log_param("max_depth", best_params["max_depth"])
mlflow.log_metric("accuracy", accuracy)


# Save the best model locally
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "best_rf_model.pkl")
joblib.dump(best_model, model_path)
print("Best model saved to {}".format(model_path))


# Prepare input example and infer signature
input_example = pd.DataFrame(X_train[:1])  # Example input (single row)
signature = infer_signature(X_train, best_model.predict(X_train))


# Log the best model with MLflow
mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    input_example=input_example,
    signature=signature,
)
