from main import load_model, predict


def test_model_loads():
    """Test if the model loads successfully"""
    model = load_model()
    assert model is not None


def test_model_predict():
    """Test if the model predicts correctly"""
    result = predict([1, 2, 3, 4, 5])
    assert isinstance(result, (float, int))
