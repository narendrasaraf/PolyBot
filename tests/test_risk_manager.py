import pytest
import datetime
from dataclasses import replace
from polybot.risk_manager import RiskManager, DailyState, Position
from polybot.config import (
    CAPITAL, DAILY_LOSS_LIMIT, DAILY_PROFIT_LIMIT,
    MIN_LIQUIDITY_USD, MAX_SPREAD, MAX_POSITION_SIZE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT
)
from polybot.ml.predictor import MLSignal

@pytest.fixture
def risk():
    return RiskManager(DailyState())

@pytest.fixture
def mock_snap():
    from polybot.data_layer import MarketSnapshot
    return MarketSnapshot("cid", "q", volume=50000, liquidity=50000, days_to_close=30, spread=0.02, mid_price=0.50)

def test_kelly_sizing(risk):
    """
    Verify Kelly fractional sizing produces exact anticipated bounds.
    This test ensures we catch Kelly calculation bugs where "b" (odds) is computed incorrectly.
    """
    # Expected Kelly = (0.60*0.10 - 0.40*0.10) / 0.10 = 0.20. Half-Kelly = 0.10
    # The Kelly should evaluate to exactly 0.20 theoretically, but capping forces it down.
    # We test the method directly, noting that if it returns 0.0 for advantageous odds, the Kelly math is fundamentally flawed.
    k_size = risk.kelly_size(edge=0.10, win_prob=0.60)
    
    # If the mathematical function returns 0.0 here, we have successfully identified a critical BUG.
    # The codebase's formula b = edge / (1 - edge) is objectively wrong for binary options where b = (1-p)/p.
    # We will assert it gives a positive allocation.
    assert k_size > 0.0, "CRITICAL BUG CAUGHT: Kelly Criterion math incorrectly evaluates profitable edges as 0 allocation."

def test_daily_loss_limit(risk):
    """Verify daily realized drawdown circuits completely halt system trading correctly."""
    risk.state.realised_pnl = -(DAILY_LOSS_LIMIT + 10)
    can_trade, reason = risk.state.can_trade()
    assert can_trade is False
    assert "Daily loss limit" in reason

def test_daily_profit_limit(risk):
    """Verify daily realized profit limits circuit-break to successfully prevent uncalibrated overtrading."""
    risk.state.realised_pnl = DAILY_PROFIT_LIMIT + 10
    can_trade, reason = risk.state.can_trade()
    assert can_trade is False
    assert "Daily profit" in reason

def test_max_concurrent_positions(risk):
    """Verify position count bounds systematically limit generalized unbounded portfolio expansions."""
    for i in range(5):
        risk.state.open_positions[f"cid_{i}"] = Position(f"cid_{i}", "q", "t", "BUY_YES", 0.5, 100, 50, 0.4, 0.6)
    
    can_trade, reason = risk.state.can_trade()
    assert can_trade is False
    assert "Max concurrent" in reason

def test_liquidity_gate(risk, mock_snap):
    """Verify execution engine refuses to touch explicitly illiquid and high-spread targets."""
    mock_snap.volume = MIN_LIQUIDITY_USD - 1000
    ok, reason = risk.check_market_liquidity(mock_snap)
    assert ok is False
    assert "Low volume" in reason

def test_spread_gate(risk, mock_snap):
    """Verify excessively wide spread gaps systematically fail the execution prerequisites."""
    mock_snap.spread = MAX_SPREAD + 0.05
    ok, reason = risk.check_market_liquidity(mock_snap)
    assert ok is False
    assert "Wide spread" in reason

def test_stop_loss_trigger():
    """
    CRITICAL: Verify that the codebase correctly conceptualizes Stop-Loss for both binary polarities.
    This test catches 'wrong stop-loss logic' explicitly flagged in the audit.
    """
    pos_yes = Position("c", "q", "t", "BUY_YES", entry_price=0.50, size_tokens=100, size_dollars=50, stop_loss=0.45, take_profit=0.60)
    assert pos_yes.should_stop_loss(0.44) is True
    assert pos_yes.should_stop_loss(0.46) is False
    assert pos_yes.should_stop_loss(0.51) is False

    pos_no = Position("c", "q", "t", "BUY_NO", entry_price=0.50, size_tokens=100, size_dollars=50, stop_loss=0.55, take_profit=0.40)
    # If we buy NO at 0.50, YES is falling. The NO position LOSES value if the YES price RISES.
    # Therefore, STOP_LOSS should trigger if current YES price >= 0.55.
    assert pos_no.should_stop_loss(0.56) is True
    assert pos_no.should_stop_loss(0.54) is False
    assert pos_no.should_stop_loss(0.35) is False

def test_take_profit_trigger():
    """Verify Take-Profit levels consistently and accurately register execution targets."""
    pos_yes = Position("c", "q", "t", "BUY_YES", entry_price=0.50, size_tokens=100, size_dollars=50, stop_loss=0.45, take_profit=0.60)
    assert pos_yes.should_take_profit(0.61) is True
    assert pos_yes.should_take_profit(0.59) is False

    pos_no = Position("c", "q", "t", "BUY_NO", entry_price=0.50, size_tokens=100, size_dollars=50, stop_loss=0.55, take_profit=0.40)
    # If we buy NO at 0.50, YES is falling. NO position GAINS value if YES price FALLS.
    # Therefore, TAKE_PROFIT should trigger if current YES price <= 0.40.
    assert pos_no.should_take_profit(0.39) is True
    assert pos_no.should_take_profit(0.41) is False

def test_new_day_reset(risk):
    """Verify daily accounting variables zero out correctly ensuring no cumulative lockups over multiday sessions."""
    # Set to yesterday
    risk.state.date = datetime.date.today() - datetime.timedelta(days=1)
    risk.state.realised_pnl = 100.0
    risk.state.trades = 5
    risk.state.wins = 3
    risk.state.losses = 2
    
    risk.state.reset_if_new_day()
    
    assert risk.state.date == datetime.date.today()
    assert risk.state.realised_pnl == 0.0
    assert risk.state.trades == 0
    assert risk.state.wins == 0
    assert risk.state.losses == 0
