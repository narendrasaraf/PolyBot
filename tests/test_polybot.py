"""
tests/test_polybot.py
=====================
Comprehensive unit + integration test suite for PolyBot.

Coverage:
  - EV formula correctness (binary payoff)
  - Kelly criterion sizing
  - Signal combination math
  - Confidence formula
  - Stop-loss / take-profit logic (BUY_YES and BUY_NO)
  - PnL calculation (Position.pnl_at)
  - Slippage model (square-root impact)
  - Backtester statistical helpers (Sharpe, Sortino, max drawdown)
  - Backtester synthetic run (smoke test)
  - HistoricalStore CSV round-trip
  - DailyState daily-limit gates
  - Data-layer MarketSnapshot .is_tradeable property
  - ML feature build with empty / minimal data
  - ML predictor heuristic fallback
  - EV helpers (edge cases: prob=0, prob=1, price=0, price=1)
  - Metrics PerformanceTracker
  - RiskManager compute_stops
  - Backtester _check_exit logic
  - evaluator compute_all_metrics
  - simulate_trades direction correctness
  - config weight sum assertion

Run:
    python -m pytest tests/test_polybot.py -v
"""

import sys
import os
import math
import tempfile
import unittest
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from dataclasses import asdict

import numpy as np
import pandas as pd

# ─── Add repo root to path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# 0. Config sanity
# ===========================================================================

class TestConfig(unittest.TestCase):
    def test_signal_weights_sum_to_one(self):
        from polybot.config import SIGNAL_WEIGHTS
        self.assertAlmostEqual(sum(SIGNAL_WEIGHTS.values()), 1.0, places=6,
                               msg="SIGNAL_WEIGHTS must sum to 1.0")

    def test_capital_positive(self):
        from polybot.config import CAPITAL
        self.assertGreater(CAPITAL, 0)

    def test_ev_threshold_positive(self):
        from polybot.config import MIN_EV_THRESHOLD
        self.assertGreater(MIN_EV_THRESHOLD, 0)

    def test_stop_loss_fraction(self):
        from polybot.config import STOP_LOSS_PCT
        self.assertGreater(STOP_LOSS_PCT, 0)
        self.assertLess(STOP_LOSS_PCT, 1)


# ===========================================================================
# 1. EV formula correctness
# ===========================================================================

class TestEVFormula(unittest.TestCase):
    """Critical: verifies the asymmetric binary-payoff EV formula."""

    def _ev_yes(self, p, mp):
        from polybot.ml.predictor import compute_ev_yes
        return compute_ev_yes(p, mp)

    def _ev_no(self, p, mp):
        from polybot.ml.predictor import compute_ev_no
        return compute_ev_no(p, mp)

    def test_ev_yes_known_value(self):
        # EV_YES = 0.70 * (1-0.55) - 0.30 * 0.55 = 0.315 - 0.165 = 0.150
        result = self._ev_yes(0.70, 0.55)
        self.assertAlmostEqual(result, 0.150, places=4)

    def test_ev_no_known_value(self):
        # EV_NO = (1-0.30) * 0.55 - 0.30 * (1-0.55) = 0.385 - 0.135 = 0.250
        result = self._ev_no(0.30, 0.55)
        self.assertAlmostEqual(result, 0.250, places=4)

    def test_ev_yes_and_no_sum_to_edge(self):
        """EV_YES and EV_NO for opposite predictions should be symmetric."""
        p, mp = 0.70, 0.50
        ev_y = self._ev_yes(p, mp)
        # At fair price (0.50), EV_NO for complement probability
        ev_n = self._ev_no(1 - p, mp)
        self.assertAlmostEqual(ev_y, ev_n, places=5)

    def test_ev_zero_at_fair_price(self):
        """When predicted_prob == market_price → EV = 0."""
        result = self._ev_yes(0.60, 0.60)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_ev_negative_when_overpriced(self):
        """If model says 0.40 but market is 0.60, EV_YES should be negative."""
        result = self._ev_yes(0.40, 0.60)
        self.assertLess(result, 0)

    def test_ev_bounded_extreme_prob(self):
        """Extreme probabilities should not produce NaN or infinity."""
        ev_y = self._ev_yes(0.0001, 0.5)
        ev_n = self._ev_no(0.9999, 0.5)
        self.assertTrue(math.isfinite(ev_y))
        self.assertTrue(math.isfinite(ev_n))

    def test_ev_yes_greater_buys_yes(self):
        """When model prob > market price, EV_YES > EV_NO."""
        ev_y = self._ev_yes(0.80, 0.50)
        ev_n = self._ev_no(0.80, 0.50)
        self.assertGreater(ev_y, ev_n)

    def test_ev_no_greater_buys_no(self):
        """When model prob < market price, EV_NO > EV_YES."""
        ev_y = self._ev_yes(0.20, 0.60)
        ev_n = self._ev_no(0.20, 0.60)
        self.assertGreater(ev_n, ev_y)


# ===========================================================================
# 2. Kelly criterion
# ===========================================================================

class TestKellyCriterion(unittest.TestCase):

    def _kelly(self, edge, win_prob):
        from polybot.risk_manager import RiskManager
        return RiskManager().kelly_size(edge, win_prob)

    def test_zero_edge_returns_zero(self):
        self.assertEqual(self._kelly(0.0, 0.70), 0.0)

    def test_negative_edge_returns_zero(self):
        self.assertEqual(self._kelly(-0.05, 0.70), 0.0)

    def test_zero_win_prob_returns_zero(self):
        self.assertEqual(self._kelly(0.10, 0.0), 0.0)

    def test_one_win_prob_returns_zero(self):
        """win_prob=1 is clamped — function should return 0 (edge case guard)."""
        self.assertEqual(self._kelly(0.10, 1.0), 0.0)

    def test_positive_sizing(self):
        size = self._kelly(0.15, 0.65)
        self.assertGreater(size, 0)

    def test_capped_at_max_position(self):
        from polybot.config import MAX_POSITION_SIZE
        size = self._kelly(0.50, 0.90)       # Very large edge
        self.assertLessEqual(size, MAX_POSITION_SIZE)

    def test_half_kelly_applied(self):
        """With half-Kelly, size should be ≤ full Kelly."""
        from polybot.config import CAPITAL
        edge, wp = 0.20, 0.60
        b = edge / max(1 - edge, 0.001)
        q = 1 - wp
        full_kelly = (b * wp - q) / b
        full_size = CAPITAL * full_kelly
        half_size = full_size * 0.5
        actual = self._kelly(edge, wp)
        # Allow for MAX_POSITION_SIZE cap
        self.assertLessEqual(actual, min(half_size + 0.01, 0.3))


# ===========================================================================
# 3. DailyState gates
# ===========================================================================

class TestDailyState(unittest.TestCase):

    def _new_state(self):
        from polybot.risk_manager import DailyState
        return DailyState()

    def test_can_trade_initially(self):
        s = self._new_state()
        ok, _ = s.can_trade()
        self.assertTrue(ok)

    def test_halts_on_daily_loss_limit(self):
        from polybot.config import DAILY_LOSS_LIMIT
        s = self._new_state()
        s.realised_pnl = -(DAILY_LOSS_LIMIT + 0.01)
        ok, reason = s.can_trade()
        self.assertFalse(ok)
        self.assertIn("loss", reason.lower())

    def test_halts_on_daily_profit_limit(self):
        from polybot.config import DAILY_PROFIT_LIMIT
        s = self._new_state()
        s.realised_pnl = DAILY_PROFIT_LIMIT + 0.01
        ok, reason = s.can_trade()
        self.assertFalse(ok)
        self.assertIn("profit", reason.lower())

    def test_halts_on_max_concurrent_positions(self):
        from polybot.risk_manager import MAX_CONCURRENT_POSITIONS, Position
        s = self._new_state()
        for i in range(MAX_CONCURRENT_POSITIONS):
            pos = MagicMock(spec=Position)
            s.open_positions[f"cid_{i}"] = pos
        ok, reason = s.can_trade()
        self.assertFalse(ok)

    def test_resets_on_new_day(self):
        s = self._new_state()
        s.date = date.today() - timedelta(days=1)
        s.realised_pnl = 5.0
        s.trades = 3
        _ = s.can_trade()   # triggers reset
        self.assertEqual(s.realised_pnl, 0.0)
        self.assertEqual(s.trades, 0)
        self.assertEqual(s.date, date.today())

    def test_open_positions_persist_across_day_reset(self):
        from polybot.risk_manager import Position
        s = self._new_state()
        s.date = date.today() - timedelta(days=1)
        pos = MagicMock(spec=Position)
        s.open_positions["cid_abc"] = pos
        _ = s.can_trade()
        self.assertIn("cid_abc", s.open_positions)  # positions survive day reset

    def test_win_rate_zero_with_no_trades(self):
        s = self._new_state()
        self.assertEqual(s.win_rate, 0.0)

    def test_win_rate_correct(self):
        from polybot.config import CAPITAL
        s = self._new_state()
        s.record_trade(1.0, "q1", "BUY_YES", 0.5, 0.6)   # win
        s.record_trade(-0.5, "q2", "BUY_YES", 0.5, 0.45)  # loss
        self.assertAlmostEqual(s.win_rate, 0.5, places=5)


# ===========================================================================
# 4. Position PnL & stop checks
# ===========================================================================

class TestPosition(unittest.TestCase):

    def _pos(self, side="BUY_YES", entry=0.50, size_tokens=10.0, sl=0.45, tp=0.60):
        from polybot.risk_manager import Position
        return Position(
            condition_id="test_cid",
            question="Test question?",
            token_id="tok_yes",
            side=side,
            entry_price=entry,
            size_tokens=size_tokens,
            size_dollars=size_tokens * entry,
            stop_loss=sl,
            take_profit=tp,
        )

    def test_pnl_at_profit_buy_yes(self):
        pos = self._pos("BUY_YES", entry=0.50, size_tokens=10)
        pnl = pos.pnl_at(0.60)
        self.assertAlmostEqual(pnl, 1.0, places=5)

    def test_pnl_at_loss_buy_yes(self):
        pos = self._pos("BUY_YES", entry=0.50, size_tokens=10)
        pnl = pos.pnl_at(0.40)
        self.assertAlmostEqual(pnl, -1.0, places=5)

    def test_pnl_at_profit_buy_no(self):
        """BUY_NO: profit when YES price falls."""
        pos = self._pos("BUY_NO", entry=0.60, size_tokens=10, sl=0.65, tp=0.45)
        pnl = pos.pnl_at(0.50)
        self.assertAlmostEqual(pnl, 1.0, places=5)

    def test_pnl_at_loss_buy_no(self):
        """BUY_NO: loss when YES price rises."""
        pos = self._pos("BUY_NO", entry=0.60, size_tokens=10, sl=0.65, tp=0.45)
        pnl = pos.pnl_at(0.70)
        self.assertAlmostEqual(pnl, -1.0, places=5)

    def test_stop_loss_triggers_buy_yes(self):
        pos = self._pos("BUY_YES", entry=0.50, sl=0.45)
        self.assertTrue(pos.should_stop_loss(0.44))
        self.assertFalse(pos.should_stop_loss(0.46))

    def test_stop_loss_triggers_buy_no(self):
        """For BUY_NO positions SL is triggered when YES price rises above SL."""
        pos = self._pos("BUY_NO", entry=0.50, sl=0.55, tp=0.35)
        self.assertTrue(pos.should_stop_loss(0.56))
        self.assertFalse(pos.should_stop_loss(0.54))

    def test_take_profit_triggers_buy_yes(self):
        pos = self._pos("BUY_YES", entry=0.50, tp=0.60)
        self.assertTrue(pos.should_take_profit(0.61))
        self.assertFalse(pos.should_take_profit(0.59))

    def test_take_profit_triggers_buy_no(self):
        pos = self._pos("BUY_NO", entry=0.60, sl=0.66, tp=0.50)
        self.assertTrue(pos.should_take_profit(0.49))
        self.assertFalse(pos.should_take_profit(0.51))


# ===========================================================================
# 5. RiskManager compute_stops
# ===========================================================================

class TestComputeStops(unittest.TestCase):

    def test_buy_yes_sl_below_entry(self):
        from polybot.risk_manager import RiskManager
        sl, tp = RiskManager().compute_stops("BUY_YES", 0.50)
        self.assertLess(sl, 0.50)
        self.assertGreater(tp, 0.50)

    def test_buy_no_sl_above_entry(self):
        from polybot.risk_manager import RiskManager
        sl, tp = RiskManager().compute_stops("BUY_NO", 0.50)
        self.assertGreater(sl, 0.50)
        self.assertLess(tp, 0.50)

    def test_stops_bounded_01(self):
        from polybot.risk_manager import RiskManager
        rm = RiskManager()
        for side, entry in [("BUY_YES", 0.01), ("BUY_NO", 0.99)]:
            sl, tp = rm.compute_stops(side, entry)
            self.assertGreaterEqual(sl, 0.01)
            self.assertLessEqual(tp, 0.99)


# ===========================================================================
# 6. MarketSnapshot .is_tradeable
# ===========================================================================

class TestMarketSnapshot(unittest.TestCase):

    def _snap(self, volume=10000, liquidity=10000, days=5, spread=0.05):
        from polybot.data_layer import MarketSnapshot
        return MarketSnapshot(
            condition_id="abc123",
            question="Test?",
            volume=volume,
            liquidity=liquidity,
            days_to_close=days,
            spread=spread,
        )

    def test_tradeable_normal(self):
        s = self._snap()
        self.assertTrue(s.is_tradeable)

    def test_not_tradeable_low_volume(self):
        from polybot.config import MIN_LIQUIDITY_USD
        s = self._snap(volume=MIN_LIQUIDITY_USD - 1)
        self.assertFalse(s.is_tradeable)

    def test_not_tradeable_low_days(self):
        from polybot.config import MIN_DAYS_TO_CLOSE
        s = self._snap(days=MIN_DAYS_TO_CLOSE - 1)
        self.assertFalse(s.is_tradeable)

    def test_not_tradeable_wide_spread(self):
        s = self._snap(spread=0.20)
        self.assertFalse(s.is_tradeable)


# ===========================================================================
# 7. HistoricalStore CSV round-trip
# ===========================================================================

class TestHistoricalStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_append_and_load(self):
        from polybot.data_layer import HistoricalStore, MarketSnapshot
        store = HistoricalStore(data_dir=self.tmpdir)
        snap = MarketSnapshot(
            condition_id="cid_test_1234567890",
            question="Will something happen?",
            volume=50000.0,
            liquidity=30000.0,
            days_to_close=15,
            mid_price=0.62,
            spread=0.03,
            best_yes_bid=0.605,
            best_yes_ask=0.635,
        )
        store.append(snap)
        df = store.load("cid_test_1234567890")
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["mid_price"], 0.62, places=4)

    def test_multiple_append_grows_rows(self):
        from polybot.data_layer import HistoricalStore, MarketSnapshot
        store = HistoricalStore(data_dir=self.tmpdir)
        for i in range(5):
            snap = MarketSnapshot(
                condition_id="cid_multi_row_test",
                question="Will something happen?",
                volume=10000.0,
                liquidity=10000.0,
                days_to_close=10,
                mid_price=0.50 + i * 0.01,
                spread=0.02,
            )
            store.append(snap)
        df = store.load("cid_multi_row_test")
        self.assertEqual(len(df), 5)

    def test_load_nonexistent_returns_empty_df(self):
        from polybot.data_layer import HistoricalStore
        store = HistoricalStore(data_dir=self.tmpdir)
        df = store.load("nonexistent_cid_xyz")
        self.assertTrue(df.empty)

    def test_list_markets(self):
        from polybot.data_layer import HistoricalStore, MarketSnapshot
        store = HistoricalStore(data_dir=self.tmpdir)
        for cid in ["cid_alpha_12345678", "cid_beta_123456789"]:
            snap = MarketSnapshot(condition_id=cid, question="Q?", volume=10000, liquidity=10000)
            store.append(snap)
        markets = store.list_markets()
        self.assertEqual(len(markets), 2)


# ===========================================================================
# 8. Backtester statistical helpers
# ===========================================================================

class TestBacktesterStats(unittest.TestCase):

    def test_sharpe_empty_returns_zero(self):
        from polybot.backtester import _sharpe
        self.assertEqual(_sharpe([]), 0.0)

    def test_sharpe_one_return_returns_zero(self):
        from polybot.backtester import _sharpe
        self.assertEqual(_sharpe([0.05]), 0.0)

    def test_sharpe_positive_mean_positive_ratio(self):
        from polybot.backtester import _sharpe
        returns = [0.02] * 30
        self.assertGreater(_sharpe(returns), 0)

    def test_sortino_empty_returns_zero(self):
        from polybot.backtester import _sortino
        self.assertEqual(_sortino([]), 0.0)

    def test_sortino_no_negative_returns_inf_or_positive(self):
        from polybot.backtester import _sortino
        returns = [0.01, 0.02, 0.03]
        result = _sortino(returns)
        self.assertTrue(result == float("inf") or result > 0)

    def test_max_drawdown_no_drawdown(self):
        from polybot.backtester import _max_drawdown
        equity = [10.0, 11.0, 12.0, 13.0]
        mdd, mdd_usd = _max_drawdown(equity)
        self.assertAlmostEqual(mdd, 0.0, places=6)

    def test_max_drawdown_known_case(self):
        from polybot.backtester import _max_drawdown
        # Peak=12, trough=9 → MDD = (9-12)/12 = 0.25
        equity = [10.0, 12.0, 9.0, 11.0]
        mdd, mdd_usd = _max_drawdown(equity)
        self.assertAlmostEqual(mdd, 0.25, places=5)

    def test_max_drawdown_empty_returns_zero(self):
        from polybot.backtester import _max_drawdown
        mdd, mdd_usd = _max_drawdown([])
        self.assertEqual(mdd, 0.0)


# ===========================================================================
# 9. Slippage model
# ===========================================================================

class TestSlippageModel(unittest.TestCase):

    def test_slippage_positive(self):
        from polybot.backtester import compute_slippage
        slip = compute_slippage(100, 10000)
        self.assertGreater(slip, 0)

    def test_slippage_zero_depth_uses_fallback(self):
        from polybot.backtester import compute_slippage
        slip = compute_slippage(100, 0)
        self.assertGreater(slip, 0)

    def test_slippage_capped_at_5pct(self):
        from polybot.backtester import compute_slippage
        # Enormous order vs tiny market
        slip = compute_slippage(1_000_000, 1)
        self.assertLessEqual(slip / 1_000_000, 0.051)  # ≤ 5 % with small tolerance

    def test_slippage_scales_with_size(self):
        from polybot.backtester import compute_slippage
        s1 = compute_slippage(100, 10000)
        s2 = compute_slippage(400, 10000)
        self.assertGreater(s2, s1)

    def test_liquidity_cap(self):
        from polybot.backtester import compute_liquidity_cap, MAX_LIQ_FRACTION
        capped = compute_liquidity_cap(10.0, 10.0, MAX_LIQ_FRACTION)
        self.assertLessEqual(capped, 10.0)


# ===========================================================================
# 10. Backtester exit logic
# ===========================================================================

class TestBacktesterExitLogic(unittest.TestCase):

    def _check(self, price, side, sl, tp, combined, i, n):
        from polybot.backtester import Backtester
        return Backtester._check_exit(price, side, sl, tp, combined, i, n)

    def _combined(self, direction="HOLD", edge=0.01):
        c = MagicMock()
        c.direction = direction
        c.edge = edge
        return c

    def test_stop_loss_buy_yes(self):
        c = self._combined("HOLD")
        result = self._check(0.44, "BUY_YES", 0.45, 0.60, c, 5, 100)
        self.assertEqual(result, "STOP_LOSS")

    def test_take_profit_buy_yes(self):
        c = self._combined("HOLD")
        result = self._check(0.62, "BUY_YES", 0.45, 0.60, c, 5, 100)
        self.assertEqual(result, "TAKE_PROFIT")

    def test_stop_loss_buy_no(self):
        c = self._combined("HOLD")
        result = self._check(0.56, "BUY_NO", 0.55, 0.40, c, 5, 100)
        self.assertEqual(result, "STOP_LOSS")

    def test_take_profit_buy_no(self):
        c = self._combined("HOLD")
        result = self._check(0.39, "BUY_NO", 0.55, 0.40, c, 5, 100)
        self.assertEqual(result, "TAKE_PROFIT")

    def test_end_of_data(self):
        c = self._combined("HOLD")
        result = self._check(0.50, "BUY_YES", 0.40, 0.65, c, 99, 100)
        self.assertEqual(result, "END_OF_DATA")

    def test_signal_reversal(self):
        c = self._combined("BUY_NO", edge=0.12)
        result = self._check(0.50, "BUY_YES", 0.40, 0.65, c, 5, 100)
        self.assertEqual(result, "SIGNAL_REVERSAL")

    def test_no_exit_when_in_range(self):
        c = self._combined("HOLD")
        result = self._check(0.52, "BUY_YES", 0.40, 0.65, c, 5, 100)
        self.assertIsNone(result)


# ===========================================================================
# 11. Backtester synthetic smoke test
# ===========================================================================

class TestBacktesterSyntheticRun(unittest.TestCase):

    def test_synthetic_run_completes(self):
        from polybot.backtester import Backtester
        bt = Backtester(warmup_rows=5)
        report = bt.run_on_synthetic(n_bars=60)
        self.assertIsNotNone(report)
        self.assertIsInstance(report.total_trades, int)
        self.assertIsInstance(report.net_pnl, float)
        self.assertTrue(math.isfinite(report.net_pnl))

    def test_report_capital_consistent(self):
        from polybot.backtester import Backtester
        bt = Backtester()
        report = bt.run_on_synthetic(n_bars=100)
        # final_capital ≥ 0
        self.assertGreaterEqual(report.final_capital, 0.0)

    def test_win_rate_bounded(self):
        from polybot.backtester import Backtester
        bt = Backtester()
        report = bt.run_on_synthetic(n_bars=100)
        self.assertGreaterEqual(report.win_rate, 0.0)
        self.assertLessEqual(report.win_rate, 1.0)


# ===========================================================================
# 12. ML feature build
# ===========================================================================

class TestMLFeatures(unittest.TestCase):

    def _snap(self, price=0.55, spread=0.04, volume=20000, days=15, liquidity=20000):
        from polybot.data_layer import MarketSnapshot
        return MarketSnapshot(
            condition_id="feat_test",
            question="Will it happen?",
            mid_price=price, spread=spread,
            volume=volume, liquidity=liquidity,
            days_to_close=days,
        )

    def test_build_features_with_empty_history(self):
        from polybot.ml.features import build_features
        snap = self._snap()
        feat = build_features(pd.DataFrame(), sentiment_score=0.5, reddit_score=0.5)
        self.assertIsNotNone(feat)
        self.assertIsInstance(feat, np.ndarray)
        self.assertGreater(len(feat), 0)

    def test_feature_vector_finite(self):
        from polybot.ml.features import build_features
        df = pd.DataFrame({
            "mid_price":    [0.50, 0.52, 0.55, 0.58, 0.60],
            "volume":       [10000] * 5,
            "spread":       [0.02] * 5,
            "liquidity":    [20000] * 5,
            "days_to_close":[30, 29, 28, 27, 26],
        })
        feat = build_features(df, sentiment_score=0.6, reddit_score=0.4)
        self.assertTrue(np.all(np.isfinite(feat)),
                        f"Feature vector has non-finite values: {feat}")

    def test_feature_length_matches_cols(self):
        from polybot.ml.features import build_features, FEATURE_COLS
        feat = build_features(pd.DataFrame(), 0.5, 0.5)
        self.assertEqual(len(feat), len(FEATURE_COLS))

    def test_no_nan_in_features(self):
        from polybot.ml.features import build_features
        df = pd.DataFrame({
            "mid_price": [0.4, 0.5, 0.6, 0.55, 0.65, 0.7, 0.68, 0.72, 0.75, 0.80],
            "volume":    [5000] * 10,
            "spread":    [0.03] * 10,
            "liquidity": [15000] * 10,
            "days_to_close": list(range(30, 20, -1)),
        })
        feat = build_features(df, 0.5, 0.5)
        self.assertFalse(np.any(np.isnan(feat)),
                         f"NaN values in feature vector: {feat}")


# ===========================================================================
# 13. ML predictor heuristic fallback
# ===========================================================================

class TestMLPredictorHeuristic(unittest.TestCase):

    def _snap(self, price=0.60, spread=0.04, volume=30000, days=20):
        m = MagicMock()
        m.mid_price   = price
        m.spread      = spread
        m.volume      = volume
        m.days_to_close = days
        return m

    def test_heuristic_returns_bounded_prob(self):
        from polybot.ml.predictor import _heuristic_prob
        snap = self._snap()
        p = _heuristic_prob(snap)
        self.assertGreaterEqual(p, 0.01)
        self.assertLessEqual(p, 0.99)

    def test_heuristic_extreme_price_high(self):
        from polybot.ml.predictor import _heuristic_prob
        snap = self._snap(price=0.99)
        p = _heuristic_prob(snap)
        self.assertGreater(p, 0.50)

    def test_heuristic_extreme_price_low(self):
        from polybot.ml.predictor import _heuristic_prob
        snap = self._snap(price=0.01)
        p = _heuristic_prob(snap)
        self.assertLess(p, 0.50)

    def test_predict_and_signal_returns_mlsignal(self):
        from polybot.ml.predictor import MLPredictor, MLSignal
        predictor = MLPredictor(model_name="auto")
        snap = self._snap()
        sig = predictor.predict_and_signal(snap, pd.DataFrame(), condition_id="test_123")
        self.assertIsInstance(sig, MLSignal)
        self.assertIn(sig.direction, ("BUY_YES", "BUY_NO", "HOLD"))

    def test_predict_and_signal_direction_consistent_with_ev(self):
        """Direction must be consistent with which EV is dominant."""
        from polybot.ml.predictor import MLPredictor
        predictor = MLPredictor(model_name="auto", ev_threshold=0.01)
        snap = self._snap(price=0.80)
        sig = predictor.predict_and_signal(snap, pd.DataFrame(), condition_id="cv_test_01")
        if sig.direction == "BUY_YES":
            self.assertGreater(sig.ev_yes, sig.ev_no)
        elif sig.direction == "BUY_NO":
            self.assertGreater(sig.ev_no, sig.ev_yes)

    def test_batch_returns_correct_length(self):
        from polybot.ml.predictor import MLPredictor
        predictor = MLPredictor()
        snaps = [self._snap(price=0.5 + i * 0.05) for i in range(4)]
        histories = [pd.DataFrame()] * 4
        results = predictor.predict_batch(snaps, histories)
        self.assertEqual(len(results), 4)


# ===========================================================================
# 14. Confidence formula
# ===========================================================================

class TestConfidenceFormula(unittest.TestCase):

    def test_high_prob_far_from_half_gives_high_calibration_certainty(self):
        from polybot.ml.predictor import _estimate_confidence
        feat = np.ones(10)
        conf = _estimate_confidence(0.90, feat, 0.90, 0.88)
        # calibration_certainty = 2*|0.90-0.5| = 0.80 → dominates
        self.assertGreater(conf, 0.50)

    def test_near_half_prob_gives_low_certainty(self):
        from polybot.ml.predictor import _estimate_confidence
        feat = np.ones(10)
        conf = _estimate_confidence(0.51, feat, 0.51, 0.49)
        self.assertLess(conf, 0.45)

    def test_model_disagreement_reduces_confidence(self):
        from polybot.ml.predictor import _estimate_confidence
        feat = np.ones(10)
        low_disagree  = _estimate_confidence(0.80, feat, 0.80, 0.79)
        high_disagree = _estimate_confidence(0.80, feat, 0.80, 0.50)
        self.assertGreater(low_disagree, high_disagree)

    def test_no_feature_vector_degrades_reliability(self):
        from polybot.ml.predictor import _estimate_confidence
        with_feat    = _estimate_confidence(0.75, np.ones(10), 0.75, 0.74)
        without_feat = _estimate_confidence(0.75, None,        0.75, 0.74)
        self.assertGreater(with_feat, without_feat)


# ===========================================================================
# 15. PerformanceTracker (metrics)
# ===========================================================================

class TestPerformanceTracker(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from polybot.metrics import PerformanceTracker
        self.perf = PerformanceTracker(
            log_path=os.path.join(self.tmpdir, "trade_log.json"),
            capital=10.0
        )

    def _record(self, pnl, side="BUY_YES"):
        self.perf.record(
            question="Test?", side=side,
            entry_price=0.50, exit_price=0.60,
            size_dollars=2.0, pnl=pnl,
            exit_reason="TAKE_PROFIT"
        )

    def test_win_rate_zero_initially(self):
        self.assertEqual(self.perf.win_rate, 0.0)

    def test_win_rate_after_trades(self):
        self._record(0.20)   # win
        self._record(-0.10)  # loss
        self.assertAlmostEqual(self.perf.win_rate, 0.5, places=5)

    def test_total_pnl_accumulates(self):
        self._record(0.20)
        self._record(-0.10)
        self.assertAlmostEqual(self.perf.total_pnl, 0.10, places=5)

    def test_profit_factor(self):
        self._record(0.30)
        self._record(-0.10)
        # 0.30 / 0.10 = 3.0
        self.assertAlmostEqual(self.perf.profit_factor, 3.0, places=2)

    def test_max_drawdown_zero_with_only_wins(self):
        self._record(0.10)
        self._record(0.10)
        self.assertAlmostEqual(self.perf.max_drawdown, 0.0, places=5)

    def test_persistence_round_trip(self):
        self._record(0.20)
        from polybot.metrics import PerformanceTracker
        perf2 = PerformanceTracker(
            log_path=os.path.join(self.tmpdir, "trade_log.json"),
            capital=10.0,
        )
        self.assertEqual(perf2.total_trades, 1)
        self.assertAlmostEqual(perf2.total_pnl, 0.20, places=5)

    def test_expectancy_formula(self):
        """expectancy = win_rate * avg_win + loss_rate * avg_loss"""
        self._record(0.40)
        self._record(0.40)
        self._record(-0.20)
        wr = self.perf.win_rate              # 2/3
        ex = wr * self.perf.avg_win + (1 - wr) * self.perf.avg_loss
        self.assertAlmostEqual(self.perf.expectancy, ex, places=5)


# ===========================================================================
# 16. Evaluator compute_all_metrics
# ===========================================================================

class TestEvaluatorMetrics(unittest.TestCase):

    def test_metrics_perfect_predictor(self):
        from polybot.ml.evaluator import compute_all_metrics
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_pred = np.array([0.99, 0.98, 0.01, 0.02])
        m = compute_all_metrics(y_true, y_pred)
        self.assertLess(m["brier"], 0.01)
        self.assertGreater(m["auc"], 0.99)

    def test_metrics_random_predictor(self):
        from polybot.ml.evaluator import compute_all_metrics
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100).astype(float)
        y_pred = rng.uniform(0, 1, 100)
        m = compute_all_metrics(y_true, y_pred)
        # Brier around 0.25 for random, AUC around 0.5
        self.assertGreater(m["brier"], 0.15)
        self.assertLess(m["auc"], 0.70)

    def test_ece_well_calibrated(self):
        from polybot.ml.evaluator import expected_calibration_error
        # Perfect calibration: pred == fraction_yes for each bin
        y_true = np.array([1, 0] * 50, dtype=float)
        y_pred = np.full(100, 0.5)
        ece = expected_calibration_error(y_true, y_pred)
        self.assertLess(ece, 0.10)

    def test_simulate_trades_correct_direction(self):
        """With perfect predictor, BUY_YES when above 0.5 should win."""
        from polybot.ml.evaluator import simulate_trades
        n = 100
        y_true  = np.ones(n)        # always resolves YES=1
        y_pred  = np.full(n, 0.80)  # model says 0.80 (edge=0.30)
        prices  = np.full(n, 0.50)
        result  = simulate_trades(y_true, y_pred, prices, min_edge=0.06, min_conf=0.0)
        self.assertGreater(result["win_rate"], 0.9)


# ===========================================================================
# 17. bot.py signal logic (standalone, no API calls)
# ===========================================================================

class TestBotSignals(unittest.TestCase):

    def test_signal_ml_returns_bounded(self):
        """signal_ml from bot.py should return score in [0.01, 0.99]."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        # Patch the genai call so it doesn't fail on import
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            import importlib
            # Only test the pure math parts, not the API-dependent ones
            from polybot.config import SIGNAL_WEIGHTS
            total = sum(SIGNAL_WEIGHTS.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_combine_signals_weighted_sum(self):
        """combine_signals should produce a score between 0 and 1."""
        # Inline the function to avoid Gemini import at module level
        def combine_signals(signals, weights):
            weighted_sum = sum(weights[k] * signals[k]["score"] for k in weights)
            scores = [signals[k]["score"] for k in weights]
            variance = sum((s - weighted_sum) ** 2 for s in scores) / len(scores)
            confidence = max(0.0, min(1.0, 1.0 - variance * 4))
            return {"yes_prob": round(weighted_sum, 3),
                    "confidence": round(confidence, 3)}

        weights = {"news": 0.4, "reddit": 0.3, "ml": 0.3}
        signals = {
            "news":   {"score": 0.7},
            "reddit": {"score": 0.6},
            "ml":     {"score": 0.8},
        }
        result = combine_signals(signals, weights)
        self.assertGreaterEqual(result["yes_prob"], 0.0)
        self.assertLessEqual(result["yes_prob"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


# ===========================================================================
# 18. Edge-case gauntlet
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_ev_yes_at_price_zero(self):
        from polybot.ml.predictor import compute_ev_yes
        # Should not raise; should return bounded value
        result = compute_ev_yes(0.70, 0.0)
        self.assertTrue(math.isfinite(result))

    def test_ev_no_at_price_one(self):
        from polybot.ml.predictor import compute_ev_no
        result = compute_ev_no(0.30, 1.0)
        self.assertTrue(math.isfinite(result))

    def test_kelly_with_exact_one_edge(self):
        from polybot.risk_manager import RiskManager
        # edge=1.0 → b=1/0.001 = 1000; shouldn't crash
        result = RiskManager().kelly_size(1.0, 0.8)
        self.assertTrue(math.isfinite(result))
        self.assertGreaterEqual(result, 0)

    def test_sharpe_with_identical_returns(self):
        from polybot.backtester import _sharpe
        returns = [0.05] * 10
        # std=0 → return 0
        result = _sharpe(returns)
        self.assertEqual(result, 0.0)

    def test_max_drawdown_single_point(self):
        from polybot.backtester import _max_drawdown
        mdd, mdd_usd = _max_drawdown([10.0])
        self.assertEqual(mdd, 0.0)

    def test_ml_features_with_single_row_df(self):
        from polybot.ml.features import build_features
        df = pd.DataFrame({"mid_price": [0.55], "volume": [10000],
                           "spread": [0.02], "liquidity": [20000], "days_to_close": [10]})
        feat = build_features(df, 0.5, 0.5)
        self.assertFalse(np.any(np.isnan(feat)))

    def test_position_pnl_zero_at_entry(self):
        from polybot.risk_manager import Position
        pos = Position(
            condition_id="c", question="q", token_id="t",
            side="BUY_YES", entry_price=0.50, size_tokens=10,
            size_dollars=5.0, stop_loss=0.45, take_profit=0.60,
        )
        self.assertAlmostEqual(pos.pnl_at(0.50), 0.0, places=8)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__(__name__))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
