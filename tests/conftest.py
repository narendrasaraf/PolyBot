import pytest
import pandas as pd
import numpy as np
import datetime
from polybot.data_layer import MarketSnapshot

@pytest.fixture
def mock_snap():
    """Provides a standard MarketSnapshot for testing."""
    return MarketSnapshot(
        condition_id="test_btc_001",
        question="Will Bitcoin reach $100k?",
        volume=50000,
        liquidity=25000,
        days_to_close=30,
        best_yes_ask=0.47,
        best_yes_bid=0.43,
        mid_price=0.45,
        spread=0.05,
        yes_token_id="test_btc_001",
        no_token_id="test_btc_001",
    )

@pytest.fixture
def mock_history():
    """Provides a 30-row historical DataFrame trending from price 0.30 to 0.50 for testing."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = np.linspace(0.30, 0.50, 30)
    return pd.DataFrame({
        "fetched_at": dates,
        "mid_price": prices,
        "volume": [10000.0] * 30,
        "liquidity": [5000.0] * 30,
        "days_to_close": np.linspace(60, 31, 30).astype(int),
        "spread": [0.05] * 30,
        "best_yes_bid": prices - 0.025,
        "best_yes_ask": prices + 0.025,
    })
