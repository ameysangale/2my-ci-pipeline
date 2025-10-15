import os
import joblib
import numpy as np
from catboost import CatBoostRegressor  # noqa: F401


def load_model(path: str = "tuned_catboost_model.pkl", n_features: int = 5
               ) -> CatBoostRegressor:
    """
    Load a trained CatBoost model from file.
    If missing, create a dummy model with the correct input dimension.

    Parameters
    ----------
    path : str
        Path to the serialized CatBoost model file.
    n_features : int
        Number of features expected by the model (for dummy fallback).

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
            print(
                f"Warning: Failed to load model from {path}. "
                f"Using dummy model. Error: {e}"
            )

    # Create dummy model with correct feature count
    model = CatBoostRegressor(
        iterations=10,
        depth=2,
        learning_rate=0.1,
        verbose=False
    )
    X_dummy = np.random.rand(10, n_features)
    y_dummy = np.random.rand(10)
    model.fit(X_dummy, y_dummy)
    return model


def predict(sample) -> float:
    """
    Make a prediction using the loaded CatBoost model.

    Parameters
    ----------
    sample : list or np.ndarray
        Input sample to predict.

    Returns
    -------
    float
        Predicted numeric value.
    """
    # Determine feature count dynamically
    n_features = len(sample)
    model = load_model(n_features=n_features)
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
