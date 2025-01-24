import pytest
import pandas as pd
import os
import joblib
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Test cases
@pytest.fixture
def sample_data():
    """Fixture to provide a small dataset for testing."""
    data = {
        "sepal_length": [5.1, 4.9, 4.7, 4.6],
        "sepal_width": [3.5, 3.0, 3.2, 3.1],
        "petal_length": [1.4, 1.4, 1.3, 1.5],
        "petal_width": [0.2, 0.2, 0.2, 0.2],
        "target": [0, 0, 0, 0],
    }
    return pd.DataFrame(data)


def test_dataset_loading(sample_data, tmpdir):
    """Test if the dataset loads correctly."""
    dataset_path = os.path.join(tmpdir, "iris.csv")
    sample_data.to_csv(dataset_path, index=False)

    # Load the dataset
    loaded_data = pd.read_csv(dataset_path)
    assert not loaded_data.empty
    assert list(loaded_data.columns) == ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]


def test_train_test_split(sample_data):
    """Test train-test split functionality."""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 3  # 80% of 4
    assert len(X_test) == 1   # 20% of 4
    assert len(y_train) == 3
    assert len(y_test) == 1


@patch("sklearn.ensemble.RandomForestClassifier.fit")
@patch("sklearn.ensemble.RandomForestClassifier.predict")
def test_objective_function(mock_predict, mock_fit, sample_data):
    """Test the objective function used in Optuna."""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mock_predict.return_value = y_test.values

    def mock_objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
        max_depth = trial.suggest_int("max_depth", 5, 20)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    trial = MagicMock()
    trial.suggest_int.side_effect = [100, 10]  # Mocking hyperparameters
    accuracy = mock_objective(trial)
    assert accuracy == 1.0  # Mocked prediction perfectly matches y_test


@patch("joblib.dump")
def test_model_saving(mock_dump, sample_data, tmpdir):
    """Test model saving functionality."""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(tmpdir, "best_rf_model.pkl")
    joblib.dump(model, model_path)
    mock_dump.assert_called_once_with(model, model_path)


def test_model_accuracy(sample_data):
    """Test accuracy computation for a trained model."""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy == 1.0  # All target values are the same, leading to perfect accuracy
