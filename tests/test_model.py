import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.model import train_model, evaluate_model

def test_train_model():
    data = load_iris()
    X, y = data.data, data.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.5  # Ensure accuracy is reasonable
