import joblib
import numpy as np
from catboost import CatBoostRegressor  # noqa: F401


def load_model(path: str = "tuned_catboost_model.pkl"):
    """
    Load a trained CatBoost model from the specified file path.

    Parameters
    ----------
    path : str
        Path to the serialized CatBoost model file.

    Returns
    -------
    model : object
        Loaded model object.
    """
    try:
        model = joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    return model


def predict(sample):
    """
    Make a prediction using the loaded CatBoost model.

    Parameters
    ----------
    sample : list or np.ndarray
        Input sample to predict. Must match the model’s input dimensions.

    Returns
    -------
    float
        Predicted numeric value.
    """
    model = load_model()
    try:
        sample = np.array(sample, dtype=float).reshape(1, -1)
        prediction = model.predict(sample)
        return float(prediction[0])
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")


if __name__ == "__main__":
    # Example usage
    test_sample = [1, 2, 3, 4, 5]
    try:
        result = predict(test_sample)
        print("Prediction:", result)
    except Exception as error:
        print(f"Error: {error}")
