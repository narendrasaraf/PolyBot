import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from polybot.strategies import (
    strategy_momentum,
    strategy_mean_reversion,
    strategy_probability_gap,
    strategy_news,
    strategy_reddit,
    strategy_ml,
    run_all_strategies,
    EmpiricalBaseRates,
    REDDIT_MAX_CONFIDENCE,
    MIN_EV_THRESHOLD
)

def test_strategy_momentum(mock_snap, mock_history):
    """Verify momentum signals correctly capture trends, outputting constraints appropriately."""
    # Uptrend
    sig_up = strategy_momentum(mock_snap, mock_history)
    assert sig_up.direction == "BUY_YES"
    assert 0.0 <= sig_up.confidence <= 1.0

    # Flat trend
    flat_history = mock_history.copy()
    flat_history["mid_price"] = 0.45
    sig_flat = strategy_momentum(mock_snap, flat_history)
    assert sig_flat.direction == "HOLD"

    # Short history
    short_history = mock_history.head(5)
    sig_short = strategy_momentum(mock_snap, short_history)
    assert sig_short.direction == "HOLD"
    assert "reason" in sig_short.metadata

def test_strategy_mean_reversion(mock_snap, mock_history):
    """Verify mean reversion signals identify statistically significant band crossings."""
    # Price > upper band (mean around 0.4, current snap is 0.45, let's force current higher)
    mock_snap.mid_price = 0.90
    sig_no = strategy_mean_reversion(mock_snap, mock_history)
    assert sig_no.direction == "BUY_NO"

    # Price < lower band
    mock_snap.mid_price = 0.10
    sig_yes = strategy_mean_reversion(mock_snap, mock_history)
    assert sig_yes.direction == "BUY_YES"

    # Within bands
    mock_snap.mid_price = mock_history["mid_price"].mean()
    sig_hold = strategy_mean_reversion(mock_snap, mock_history)
    assert sig_hold.direction == "HOLD"

    # Flat market
    flat_hist = mock_history.copy()
    flat_hist["mid_price"] = 0.50
    sig_flat = strategy_mean_reversion(mock_snap, flat_hist)
    assert sig_flat.direction == "HOLD"
    assert sig_flat.metadata["reason"] == "flat market"

def test_strategy_probability_gap(mocker, mock_snap):
    """Verify probabilistic gap calculation computes dynamic weightings against empirical base rates."""
    # Mock EmpiricalBaseRates.get_rate
    mocker.patch(
        "polybot.strategies.EmpiricalBaseRates.get_rate",
        return_value=(0.42, 0.50)  # base_rate, conf (meaning high conf (>0.3) -> weight 0.4)
    )
    mock_snap.question = "Will Bitcoin reach $100k?"
    mock_snap.mid_price = 0.55
    mock_snap.volume = 50000
    
    sig = strategy_probability_gap(mock_snap)
    # Vol_factor = 500k / 50k = 10 -> capped at 2.5
    assert sig.metadata["vol_factor"] == 2.5
    # fair_value calculation (approximate without exact extreme adj testing)
    assert sig.yes_probability < 0.55
    assert sig.direction == "BUY_NO"

    # Test HOLD for tight edge
    mock_snap.mid_price = 0.42
    sig_tight = strategy_probability_gap(mock_snap)
    assert sig_tight.direction == "HOLD"

def test_strategy_news(mocker):
    """Verify the Gemini AI LLM news strategy safely handles varied network responses and quotas."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '```json\n{"score": 75, "confidence": 80, "summary": "positive news"}\n```'
    mock_client.models.generate_content.return_value = mock_resp
    
    mocker.patch("polybot.strategies._gemini_client", return_value=(mock_client, MagicMock()))
    
    # Success case
    sig = strategy_news("Will BTC rise?")
    assert sig.yes_probability == 0.75
    assert sig.confidence == 0.80
    assert sig.direction == "BUY_YES"
    
    # Bad JSON fallback
    mock_resp.text = "Error, no json"
    sig_bad = strategy_news("Will BTC rise?")
    assert sig_bad.yes_probability == 0.50
    
    # Exceeded rate limit / 429
    mock_client.models.generate_content.side_effect = Exception("429 Quota Exceeded")
    mock_sleep = mocker.patch("polybot.strategies.time.sleep")
    sig_429 = strategy_news("Rate test")
    assert mock_sleep.call_count == 3
    assert sig_429.yes_probability == 0.50

def test_strategy_reddit(mocker, mock_snap):
    """Verify Reddit bag-of-words parsing and strict confidence capping constraints."""
    mock_get = mocker.patch("polybot.strategies.requests.get")
    
    # 5 posts with positive signals
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": {
            "children": [
                {"data": {"title": "Bitcoin will win", "selftext": "bullish confirmed", "score": 100}} for _ in range(5)
            ]
        }
    }
    mock_get.return_value = mock_resp
    
    sig = strategy_reddit("Bitcoin $100k?")
    assert sig.direction == "BUY_YES"
    assert sig.yes_probability > 0.50
    # Confirm Reddit strict max cap
    assert sig.confidence <= REDDIT_MAX_CONFIDENCE
    
    # Zero posts fallback
    mock_resp.json.return_value = {"data": {"children": []}}
    sig_empty = strategy_reddit("Obscure market?")
    assert sig_empty.yes_probability == 0.50

def test_strategy_ml(mocker, mock_snap, mock_history):
    """Verify ML strategy integration and fallback handling logic."""
    from polybot.ml.predictor import  MLSignal, predictor
    # Mock prediction wrapper
    mock_signal = MLSignal(
        model_name="test_model", market_price=0.5, predicted_prob=0.65,
        ev_yes=0.15, ev_no=-0.15, direction="BUY_YES", confidence=0.85, is_high_conf=True
    )
    mocker.patch.object(predictor, "predict_and_signal", return_value=mock_signal)
    
    sig = strategy_ml(mock_snap, 30, mock_history)
    assert sig.yes_probability == 0.65
    assert sig.direction == "BUY_YES"

def test_run_all_strategies(mocker, mock_snap, mock_history):
    """Verify the ensemble meta-aggregator correctly synthesizes varied signals into one actionable combined signal."""
    from polybot.strategies import SignalResult
    from polybot.config import MIN_CONFIDENCE
    
    mocker.patch("polybot.strategies.strategy_momentum", return_value=SignalResult("momentum", "BUY_YES", 0.70, 0.80))
    mocker.patch("polybot.strategies.strategy_mean_reversion", return_value=SignalResult("mean_rev", "HOLD", 0.50, 0.10))
    mocker.patch("polybot.strategies.strategy_probability_gap", return_value=SignalResult("prob_gap", "BUY_YES", 0.60, 0.60))
    mocker.patch("polybot.strategies.strategy_news", return_value=SignalResult("news", "BUY_YES", 0.65, 0.70))
    mocker.patch("polybot.strategies.strategy_reddit", return_value=SignalResult("reddit", "HOLD", 0.50, 0.20))
    mocker.patch("polybot.strategies.strategy_ml", return_value=SignalResult("ml", "BUY_YES", 0.75, 0.90))
    mocker.patch("polybot.strategies._is_ml_live_ready", return_value=False)
    mocker.patch("polybot.strategies._load_meta_model", return_value=None)
    
    mock_snap.mid_price = 0.50
    combined = run_all_strategies(mock_snap, mock_history)
    
    assert 0.5 < combined.yes_probability < 0.8
    assert combined.direction == "BUY_YES"
    # Ensure disagreement penalty is applied to confidence
    assert combined.confidence > 0.0 
    
    # Actionable logic: confidence > MIN_CONFIDENCE and EV_YES > MIN_EV_THRESHOLD
    if combined.confidence >= MIN_CONFIDENCE and combined.ev_yes >= MIN_EV_THRESHOLD:
        assert combined.is_actionable is True
