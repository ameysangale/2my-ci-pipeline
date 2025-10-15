
from main import load_model, predict


def test_model_loads():
    """
    Test that the model loads successfully, either from file or dummy.
    """
    model = load_model(n_features=5)
    # Check the object type
    from catboost import CatBoostRegressor
    assert isinstance(model, CatBoostRegressor)


def test_model_predict():
    """
    Test that prediction returns a float and works with dummy model.
    """
    test_sample = [1, 2, 3, 4, 5]
    result = predict(test_sample)
    assert isinstance(result, float)
