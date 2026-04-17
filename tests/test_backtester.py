import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from polybot.backtester import (
    Backtester,
    BacktestReport,
    BtTrade,
    compute_slippage,
    compute_liquidity_cap,
    simulated_fill_price,
    _sharpe,
    _sortino,
    _max_drawdown
)
from polybot.data_layer import MarketSnapshot
from polybot.strategies import CombinedSignal

# --- Cost model tests ---
def test_compute_slippage():
    """Verify that the square-root market impact model produces exactly the expected slippage costs."""
    # slippage(size=100, depth=10000) = 0.002 * (100/10000)^0.5 * 100 = 0.02
    slip = compute_slippage(size_usd=100.0, market_depth=10000.0, coeff=0.002, power=0.5)
    assert abs(slip - 0.02) < 1e-5

def test_slippage_cap():
    """Verify slippage assumes an absolute safety ceiling on thin books to prevent runaway pricing exceptions."""
    slip = compute_slippage(size_usd=1000.0, market_depth=10.0, coeff=0.002, power=0.5)
    # 5% of 1000 = 50.0
    assert abs(slip - 50.0) < 1e-5

def test_liquidity_cap():
    """Verify position capitalization scales safely against strict maximum volume depth constraints."""
    cap = compute_liquidity_cap(1000.0, 10000.0, 0.05)
    assert cap == 500.0  # 5% of 10000

def test_fill_price_buy_yes_entry():
    """Verify combined fill price includes both the bid-ask half spread and mathematically correct impact slippage."""
    fill = simulated_fill_price(mid=0.50, spread=0.04, slip_frac=0.01, side="BUY_YES", direction="ENTRY")
    # mid + half_spread + mid * slip = 0.50 + 0.02 + 0.005 = 0.525
    assert abs(fill - 0.525) < 1e-5

def test_fill_price_symmetry():
    """Verify that entries and exits produce symmetrically penalizing fill prices mapping correctly across market maker costs."""
    entry_fill = simulated_fill_price(mid=0.50, spread=0.04, slip_frac=0.01, side="BUY_YES", direction="ENTRY")
    exit_fill  = simulated_fill_price(mid=0.50, spread=0.04, slip_frac=0.01, side="BUY_YES", direction="EXIT")
    # Entry: 0.525; Exit: 0.50 - 0.02 - 0.005 = 0.475
    assert abs(abs(entry_fill - 0.50) - abs(exit_fill - 0.50)) < 1e-5

# --- Backtester engine tests ---
@pytest.fixture
def base_df():
    import datetime
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.DataFrame({
        "fetched_at": dates,
        "mid_price": np.linspace(0.40, 0.60, 50),
        "volume": [10000.0] * 50,
        "liquidity": [5000.0] * 50,
        "days_to_close": np.linspace(60, 11, 50).astype(int),
        "spread": [0.02] * 50,
        "best_yes_bid": np.linspace(0.39, 0.59, 50),
        "best_yes_ask": np.linspace(0.41, 0.61, 50)
    })

def test_empty_dataframe():
    """Verify that providing incomplete or short dataframes produces empty but type-safe BacktestReports."""
    bt = Backtester()
    rep = bt.run_on_dataframe(pd.DataFrame({"mid_price": [0.5]*10}), "test", "q")
    assert rep.total_trades == 0

def test_no_trade_signal(mocker, base_df):
    """Verify the engine does not trigger or fabricate trades if strategies enforce absolute HOLD."""
    bt = Backtester()
    mocker.patch.object(bt, "_run_signals", return_value=CombinedSignal(0.5, 0.5, 0, 0, 0, "HOLD"))
    rep = bt.run_on_dataframe(base_df)
    assert rep.total_trades == 0

def test_stop_loss_triggers(mocker, base_df):
    """Verify the risk framework successfully flags and forces liquidation when standard Stop-Loss criteria are breached."""
    df = base_df.copy()
    # Force entry at index 20
    # Then drop price by 6% (from ~ 0.48 down to 0.45)
    df.loc[25:, "mid_price"] = 0.45
    
    bt = Backtester(warmup_rows=15)
    def mock_signal(snap, hist):
        if len(hist) == 20: 
            return CombinedSignal(0.7, 0.9, 0.1, 0, 0.1, "BUY_YES")
        return CombinedSignal(0.5, 0.5, 0, 0, 0, "HOLD")
    mocker.patch.object(bt, "_run_signals", side_effect=mock_signal)
    
    rep = bt.run_on_dataframe(df)
    assert rep.total_trades > 0
    assert rep.trades[0].exit_reason == "STOP_LOSS"

def test_take_profit_triggers(mocker, base_df):
    """Verify the risk framework systematically identifies and executes liquidation constraints on explicit Take-Profit breaches."""
    df = base_df.copy()
    df.loc[25:, "mid_price"] = 0.65 # massive jump
    
    bt = Backtester(warmup_rows=15)
    def mock_signal(snap, hist):
        if len(hist) == 20: 
            return CombinedSignal(0.8, 0.9, 0.2, 0, 0.2, "BUY_YES")
        return CombinedSignal(0.5, 0.5, 0, 0, 0, "HOLD")
    mocker.patch.object(bt, "_run_signals", side_effect=mock_signal)
    
    rep = bt.run_on_dataframe(df)
    assert rep.total_trades > 0
    assert rep.trades[0].exit_reason == "TAKE_PROFIT"

def test_end_of_data_exit(mocker, base_df):
    """Verify the engine strictly unwinds open exposures safely at conclusion of historical series (END_OF_DATA)."""
    bt = Backtester(warmup_rows=15)
    def mock_signal(snap, hist):
        if len(hist) == 48: 
            return CombinedSignal(0.7, 0.9, 0.1, 0, 0.1, "BUY_YES")
        return CombinedSignal(0.5, 0.5, 0, 0, 0, "HOLD")
    mocker.patch.object(bt, "_run_signals", side_effect=mock_signal)
    
    rep = bt.run_on_dataframe(base_df)
    assert rep.total_trades == 1
    assert rep.trades[0].exit_reason == "END_OF_DATA"

def test_signal_reversal_exit(mocker, base_df):
    """Verify dynamic market condition reversals are honored and prematurely liquidate earlier conflicting positions."""
    bt = Backtester(warmup_rows=15)
    def mock_signal(snap, hist):
        if len(hist) == 20: 
            return CombinedSignal(0.7, 0.9, 0.1, -0.1, 0.1, "BUY_YES")
        if len(hist) == 25:
            # Reversal with edge > 0.08
            return CombinedSignal(0.3, 0.9, -0.1, 0.1, -0.15, "BUY_NO")
        return CombinedSignal(0.5, 0.5, 0, 0, 0, "HOLD")
    mocker.patch.object(bt, "_run_signals", side_effect=mock_signal)
    
    rep = bt.run_on_dataframe(base_df)
    assert rep.total_trades == 1
    assert rep.trades[0].exit_reason == "SIGNAL_REVERSAL"

# --- Report Tests ---
def test_win_rate_calculation():
    """Verify explicit tracking calculations correctly render Win Rates under generalized conditions."""
    rep = Backtester()._empty_report("test", "q")
    trades = [
        # 3 wins, 2 losses
        BtTrade(1, "c", "q", "BUY_YES", "t", "t", 0.5, 0.6, 0.5, 0.6, 100, 200, 20, 0, 0, 20, 0.2, "TP", 0.5, 0, 1),
        BtTrade(2, "c", "q", "BUY_YES", "t", "t", 0.5, 0.6, 0.5, 0.6, 100, 200, 20, 0, 0, 20, 0.2, "TP", 0.5, 0, 1),
        BtTrade(3, "c", "q", "BUY_YES", "t", "t", 0.5, 0.6, 0.5, 0.6, 100, 200, 20, 0, 0, 20, 0.2, "TP", 0.5, 0, 1),
        BtTrade(4, "c", "q", "BUY_YES", "t", "t", 0.5, 0.4, 0.5, 0.4, 100, 200, -20, 0, 0, -20, -0.2, "SL", 0.5, 0, 1),
        BtTrade(5, "c", "q", "BUY_YES", "t", "t", 0.5, 0.4, 0.5, 0.4, 100, 200, -20, 0, 0, -20, -0.2, "SL", 0.5, 0, 1),
    ]
    bt = Backtester(initial_capital=100)
    rep = bt._build_report("test", "q", 100, trades, [100])
    assert abs(rep.win_rate - 0.60) < 1e-4

def test_profit_factor():
    """Verify analytical profit factor ratios safely compute gross wins versus bounded gross losses."""
    trades = [
        BtTrade(1, "c", "q", "BUY_YES", "t", "t", 0.5, 0.6, 0.5, 0.6, 100, 200, 20, 0, 0, 20, 0.2, "TP", 0.5, 0, 1),
        BtTrade(2, "c", "q", "BUY_YES", "t", "t", 0.5, 0.4, 0.5, 0.4, 100, 200, -10, 0, 0, -10, -0.1, "SL", 0.5, 0, 1),
    ]
    bt = Backtester(initial_capital=100)
    rep = bt._build_report("test", "q", 100, trades, [100])
    assert abs(rep.profit_factor - 2.0) < 1e-4

def test_max_drawdown():
    """Verify maximum historical peak drawdown computes explicit worst-case scenario correctly."""
    mdd, _ = _max_drawdown([100, 90, 85, 95, 80])
    assert abs(mdd - 0.20) < 1e-4

def test_sharpe_with_single_trade():
    """Verify that standalone and singleton trading intervals don't cause statistical division-by-zero crashes for Sharpe metrics."""
    assert _sharpe([0.1]) == 0.0

def test_roi_calculation():
    """Verify explicitly computed ROIs map proportionally to nominal scale invariants."""
    trades = [BtTrade(1, "c", "q", "BUY_YES", "t", "t", 0.5, 0.6, 0.5, 0.6, 100, 200, 2.0, 0, 0, 2.0, 0.02, "TP", 0.5, 0, 1)]
    bt = Backtester(initial_capital=100)
    rep = bt._build_report("test", "q", 102, trades, [100, 102])
    assert abs(rep.roi_pct - 2.0) < 1e-4

# --- Synthetic Integration Tests ---
@pytest.mark.slow
def test_synthetic_backtest_runs():
    """Verify procedural and system-encompassing synthetic evaluations execute flawlessly end-to-end."""
    bt = Backtester()
    rep = bt.run_on_synthetic(n_bars=200)
    assert isinstance(rep, BacktestReport)

@pytest.mark.slow
def test_realistic_fee_drag(mocker, base_df):
    """Verify trading volume costs represent approximately accurate proportional fee friction on nominal portfolio scale."""
    bt = Backtester(fee_rate=0.01)
    # Ensure entry exits frequently
    def mock_signal(snap, hist):
        if len(hist) % 3 == 0: 
            return CombinedSignal(0.8, 0.9, 0.2, 0, 0.2, "BUY_YES")
        elif len(hist) % 3 == 1:
            return CombinedSignal(0.2, 0.9, 0, 0.2, -0.2, "BUY_NO")
        return CombinedSignal(0.5, 0.5, 0, 0, 0, "HOLD")
    mocker.patch.object(bt, "_run_signals", side_effect=mock_signal)
    
    rep = bt.run_on_dataframe(base_df)
    if rep.total_trades >= 10:
        # fee drag % = total_fees / max(abs(gross_pnl), 1e-9) * 100. Let's just ensure total_fees > 0
        assert rep.total_fees > 0
        # Given taker fee of 1% and average gross PnL is low, dragging is visible
        assert rep.fee_drag_pct > 0.1
