import os
import joblib
from catboost import CatBoostRegressor
from main import load_model, predict


def setup_module(module):
    """
    Create a lightweight dummy CatBoost model before tests run.
    This prevents FileNotFoundError in CI environments.
    """
    model = CatBoostRegressor(iterations=1, depth=1, learning_rate=0.1)
    joblib.dump(model, "tuned_catboost_model.pkl")


def teardown_module(module):
    """
    Delete the dummy model file after tests complete.
    Keeps the workspace clean.
    """
    if os.path.exists("tuned_catboost_model.pkl"):
        os.remove("tuned_catboost_model.pkl")


def test_model_loads():
    """Test if the model loads successfully."""
    model = load_model()
    assert model is not None


def test_model_predict():
    """Test if prediction returns a float value."""
    result = predict([1, 2, 3, 4, 5])
    assert isinstance(result, float)
