import pytest
import numpy as np
import os
import tempfile
from polybot.ml.models import LogisticModel, XGBoostModel, LightGBMModel

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.rand(200, 35).astype(np.float32)
    y = np.random.choice([0, 1], size=200).astype(np.float32)
    return X, y

@pytest.fixture
def eval_data():
    np.random.seed(43)
    X_val = np.random.rand(50, 35).astype(np.float32)
    y_val = np.random.choice([0, 1], size=50).astype(np.float32)
    return X_val, y_val

# --- LogisticModel Tests ---
def test_logistic_fit_predict(dummy_data, eval_data):
    """Verify that LogisticModel can fit binary labels and return valid probabilities."""
    X_train, y_train = dummy_data
    X_test, _ = eval_data
    model = LogisticModel()
    model.fit(X_train, y_train)
    probs = model.predict(X_test)
    assert probs.shape == (50,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

def test_logistic_predict_before_fit(eval_data):
    """Verify prediction prior to fitting returns neutral 0.5 without crashing."""
    X_test, _ = eval_data
    model = LogisticModel()
    probs = model.predict(X_test)
    assert probs.shape == (50,)
    assert np.all(probs == 0.5)

def test_logistic_save_load(dummy_data, eval_data):
    """Verify serialization and deserialization of the model preserve exact behavior."""
    X_train, y_train = dummy_data
    X_test, _ = eval_data
    model = LogisticModel()
    model.fit(X_train, y_train)
    probs_orig = model.predict(X_test)
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    
    try:
        model.save(path)
        loaded_model = LogisticModel.load(path)
        probs_loaded = loaded_model.predict(X_test)
        np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-6)
    finally:
        os.remove(path)

def test_logistic_feature_importances(dummy_data):
    """Verify feature importances are extracted safely and correctly shaped."""
    X_train, y_train = dummy_data
    model = LogisticModel()
    model.fit(X_train, y_train)
    importances = model.feature_importances()
    assert importances.shape == (35,)
    assert np.all(importances >= 0)

# --- XGBoostModel Tests ---
def test_xgboost_fit_predict(dummy_data, eval_data):
    """Verify XGBoostModel trains on binary outcomes correctly (regression labels will raise ValueError)."""
    X_train, y_train = dummy_data
    # Re-verify that passing continuous y to fit() throws a ValueError, as models strictly require binary.
    continuous_y = np.random.rand(200).astype(np.float32)
    model = XGBoostModel()
    # Continuous y should result in ValueError internally, though model code catches it or raises it depending on location.
    # The new version of XGBoostModel uses _ensure_binary which explicitly raises ValueError.
    with pytest.raises(ValueError):
        model.fit(X_train, continuous_y)

    # Now verify with correct binary labels
    model.fit(X_train, y_train)
    probs = model.predict(eval_data[0])
    assert probs.shape == (50,)
    assert np.all((probs > 0.0) & (probs < 1.0))

def test_xgboost_early_stopping(dummy_data, eval_data):
    """Verify XGBoost uses early stopping with a validation set to prevent overfitting."""
    X_train, y_train = dummy_data
    X_val, y_val = eval_data
    model = XGBoostModel(n_estimators=50, early_stopping_rounds=5)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model.is_fitted

def test_xgboost_predict_before_fit(eval_data):
    """Verify predict before fit fallback for XGBoost returns 0.5 safely."""
    X_test, _ = eval_data
    model = XGBoostModel()
    probs = model.predict(X_test)
    assert probs.shape == (50,)
    assert np.all(probs == 0.5)

def test_xgboost_save_load(dummy_data, eval_data):
    """Verify XGBoost serialization saves both scaler and native json model cleanly."""
    X_train, y_train = dummy_data
    X_test, _ = eval_data
    model = XGBoostModel()
    model.fit(X_train, y_train)
    probs_orig = model.predict(X_test)
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
        
    try:
        model.save(path)
        loaded_model = XGBoostModel.load(path)
        probs_loaded = loaded_model.predict(X_test)
        np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5)
    finally:
        os.remove(path)
        json_path = path.replace(".pkl", ".json")
        if os.path.exists(json_path):
            os.remove(json_path)

def test_xgboost_clipping(dummy_data):
    """Verify XGBoost strictly complains about out-of-bounds y values, ensuring we don't regress to regression on [0,1]."""
    X_train, _ = dummy_data
    y_out_of_bounds = np.array([-0.5] * 100 + [1.5] * 100)
    model = XGBoostModel()
    with pytest.raises(ValueError):
        model.fit(X_train, y_out_of_bounds)

# --- LSTM Tests (Skipped) ---
@pytest.mark.skip(reason="LSTMModel has been completely removed in the new causal architecture to prevent bidirectionality leakage.")
def test_lstm_sequence_building():
    """Verify sequence building shapes."""
    pass

@pytest.mark.skip(reason="LSTMModel is deprecated.")
def test_lstm_fit_predict():
    pass

@pytest.mark.skip(reason="LSTMModel is deprecated.")
def test_lstm_padding():
    pass

@pytest.mark.skip(reason="LSTMModel is deprecated.")
def test_lstm_save_load():
    pass

# --- Edge Cases ---
def test_empty_input(dummy_data):
    """Verify predicting on empty arrays doesn't crash but returns appropriately."""
    _, y_train = dummy_data
    model = LogisticModel()
    empty_X = np.empty((0, 35))
    probs = model.predict(empty_X)
    assert probs.shape == (0,)

def test_single_sample(dummy_data):
    """Verify single-sample predictions return correct scalar-like boundaries."""
    X_train, y_train = dummy_data
    model = LogisticModel()
    model.fit(X_train, y_train)
    prob = model.predict_single(X_train[0])
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_all_same_labels(dummy_data):
    """Verify models do not crash if trained on uniform labels, keeping probabilistic properties intact."""
    X_train, _ = dummy_data
    y_same = np.zeros(200)
    model = LogisticModel()
    model.fit(X_train, y_same)
    probs = model.predict(X_train[:10])
    assert len(probs) == 10
    assert np.all((probs >= 0.0) & (probs <= 1.0))
