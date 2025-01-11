import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from model import train_model, evaluate_model

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "iris_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
