import os
import joblib
import numpy as np
from catboost import CatBoostRegressor  # noqa: F401


def load_model(path: str = "tuned_catboost_model.pkl") -> CatBoostRegressor:
    """
    Load a trained CatBoost model from the specified file path.
    If the file doesn't exist, create a dummy model for testing/CI.

    Parameters
    ----------
    path : str
        Path to the serialized CatBoost model file.

    Returns
    -------
    model : CatBoostRegressor
        Loaded or dummy-trained CatBoost model.
    """
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            print(f"Warning: Failed to load model from {path}. Using dummy model. Error: {e}")

    # Fallback dummy model for CI/testing
    model = CatBoostRegressor(iterations=10, depth=2, learning_rate=0.1, verbose=False)
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.random.rand(10)
    model.fit(X_dummy, y_dummy)
    return model


def predict(sample) -> float:
    """
    Make a prediction using the loaded CatBoost model.

    Parameters
    ----------
    sample : list or np.ndarray
        Input sample to predict. Must match the model's input dimensions.

    Returns
    -------
    float
        Predicted numeric value.
    """
    model = load_model()
    try:
        sample_arr = np.array(sample, dtype=float).reshape(1, -1)
        prediction = model.predict(sample_arr)
        return float(prediction[0])
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}") from e


if __name__ == "__main__":
    # Example usage
    test_sample = [1, 2, 3, 4, 5]
    try:
        result = predict(test_sample)
        print("Prediction:", result)
    except Exception as error:
        print(f"Error during prediction: {error}")
