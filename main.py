import joblib
import numpy as np
from catboost import CatBoostRegressor  # noqa: F401



def load_model(path="tuned_catboost_model.pkl"):
    """Load trained CatBoost model"""
    model = joblib.load(path)
    return model


def predict(sample):
    """Make prediction using the loaded model"""
    model = load_model()
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)
    return float(prediction[0])  # Ensure output is a float


if __name__ == "__main__":
    # Example usage
    result = predict([1, 2, 3, 4, 5])
    print("Prediction:", result)
