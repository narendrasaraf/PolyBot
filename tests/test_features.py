import pytest
import numpy as np
import pandas as pd
import math
from polybot.ml.features import (
    build_features,
    build_feature_matrix,
    build_multi_market_dataset,
    feature_importance_table,
    FEATURE_COLS,
)

def test_build_features_minimal_data():
    """Verify that build_features returns None if there are fewer than 5 rows, avoiding crashes on new markets."""
    df = pd.DataFrame({"mid_price": [0.5, 0.51, 0.52, 0.53]}) # 4 rows
    assert build_features(df) is None

def test_build_features_known_values():
    """Verify that feature engineering computes exact known mathematical values to prevent silent calculation errors."""
    np.random.seed(42)
    prices = np.linspace(0.30, 0.80, 20)
    df = pd.DataFrame({
        "mid_price": prices,
        "volume": [1000] * 20,
        "liquidity": [5000] * 20,
        "spread": [0.05] * 20,
        "days_to_close": [30.0] * 20,
    })
    
    feat = build_features(df)
    assert feat is not None
    assert len(feat) == len(FEATURE_COLS)
    
    feat_dict = dict(zip(FEATURE_COLS, feat))
    
    # price
    expected_price = float(np.clip(prices[-1], 0.001, 0.999))
    np.testing.assert_allclose(feat_dict["price"], expected_price, rtol=1e-4)
    
    # logit
    expected_log_price = math.log(expected_price / (1 - expected_price))
    np.testing.assert_allclose(feat_dict["log_price"], expected_log_price, rtol=1e-4)
    
    # time features
    expected_urgency = math.exp(-30.0 / 30.0)  # ~0.368
    np.testing.assert_allclose(feat_dict["time_urgency"], expected_urgency, rtol=1e-4)
    assert feat_dict["near_resolution"] == 0.0

def test_build_features_nan_inf_handling():
    """Verify that feature building gracefully handles missing and infinite inputs by coercing them to valid floats."""
    df = pd.DataFrame({
        "mid_price": [0.5, np.nan, np.inf, -np.inf, 0.6],
        "volume": [1000, 1000, 1000, 1000, 1000],
    })
    # Will forward/backward fill or output nan_to_num
    feat = build_features(df)
    assert feat is not None
    assert np.all(np.isfinite(feat))

def test_build_feature_matrix_shapes_and_defaults():
    """Verify that build_feature_matrix generates the correct shape and handles missing optional columns."""
    df = pd.DataFrame({"mid_price": np.random.rand(50), "resolution": [1.0] * 50})
    X, y = build_feature_matrix(df)
    assert X.shape == (50, 35)
    assert len(y) == 50
    assert np.all((y == 0.0) | (y == 1.0))
    assert np.all(np.isfinite(X))

def test_build_multi_market_dataset():
    """Verify combining multiple markets correctly aggregates shapes and ignores invalid data."""
    df1 = pd.DataFrame({"mid_price": np.random.rand(20), "resolution": [1.0] * 20})
    df2 = pd.DataFrame({"mid_price": np.random.rand(30), "resolution": [0.0] * 30})
    df3 = pd.DataFrame({"mid_price": np.random.rand(5)})  # Too short, should be skipped
    
    X, y, cids = build_multi_market_dataset({"m1": df1, "m2": df2, "m3": df3}, min_rows=10)
    assert X.shape == (50, 35)
    assert len(y) == 50
    assert len(cids) == 50
    assert cids.count("m1") == 20
    assert cids.count("m2") == 30

def test_feature_importance_table():
    """Verify feature_importance_table returns a properly formatted, descending sorted DataFrame."""
    importances = np.random.rand(35)
    df = feature_importance_table(importances)
    assert list(df.columns) == ["feature", "importance"]
    assert len(df) == 35
    assert df["importance"].is_monotonic_decreasing

def test_causal_constraint():
    """CRITICAL: Verify that the feature vector at index i relies ONLY on data from rows 0..i, preventing algorithmic data leakage."""
    np.random.seed(42)
    df = pd.DataFrame({
        "mid_price": np.random.rand(50),
        "volume": [1000] * 50,
        "liquidity": [5000] * 50,
        "spread": [0.05] * 50,
        "days_to_close": [30.0] * 50,
        "resolution": [1.0] * 50
    })
    
    # Extract features normally
    X_normal, _ = build_feature_matrix(df)
    
    # Pick a row i
    i = 25
    # Corrupt all FUTURE rows (i+1 onwards)
    df_corrupted = df.copy()
    df_corrupted.loc[i+1:, "mid_price"] = 999.9  # Future data leak test
    
    X_corrupted, _ = build_feature_matrix(df_corrupted)
    
    # The feature row `i` should be EXACTLY identical, meaning it didn't look ahead.
    np.testing.assert_allclose(X_normal[i], X_corrupted[i], atol=1e-5)
    
    # Also verify with single-row feature builder
    feat_normal = build_features(df, row_idx=i)
    feat_corrupted = build_features(df_corrupted, row_idx=i)
    np.testing.assert_allclose(feat_normal, feat_corrupted, atol=1e-5)
