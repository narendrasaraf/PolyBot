"""
polybot/backtester.py
=====================
Production-Grade Backtesting Engine for Polymarket trading strategies.

Design principles:
  ─ Walk-forward replay:  no look-ahead bias, each bar sees only past data
  ─ Realistic cost model: transaction fees + price-impact slippage
  ─ Liquidity gates:      position sizes capped by available market depth
  ─ Rich metrics:         PnL, win rate, Sharpe, Sortino, Calmar, max drawdown
  ─ Full trade log:       every fill recorded with fees, slippage, exit reason
  ─ CSV & plot export:    audit-trail and equity-curve visualisation

Market cost assumptions (Polymarket CLOB, 2024):
  Fee:       ~1 % of notional (maker 0.0 %, taker ~1.0 %)
  Slippage:  market-impact model — scales with (position_usd / market_depth)
  Liquidity: max position = min(configured_cap, 5 % of 24h volume)
"""

from __future__ import annotations

import math
import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from polybot.data_layer import HistoricalStore, MarketSnapshot, store
from polybot.strategies import run_all_strategies
from polybot.config import (
    CAPITAL, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MAX_POSITION_SIZE, MIN_CONFIDENCE, MIN_EDGE,
)
from polybot.logger import get_logger

log = get_logger("backtester")


# ─────────────────────────────────────────────────────────────────────────────
# Cost model constants
# ─────────────────────────────────────────────────────────────────────────────

TAKER_FEE_RATE    = 0.010      # 1.0 % taker fee (Polymarket CLOB, 2024)
MAKER_FEE_RATE    = 0.000      # 0.0 % maker fee
MAX_LIQ_FRACTION  = 0.05       # max position = 5 % of 24h volume
SLIPPAGE_POWER    = 0.5        # slippage ∝ (size / depth)^0.5  (square-root impact)
SLIPPAGE_COEFF    = 0.002      # coefficient: 0.2 % for 100 % of depth


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BtFill:
    """A single simulated order fill, including all realistic costs."""
    time:          str
    side:          str           # "BUY_YES" | "BUY_NO"
    direction:     str           # "ENTRY" | "EXIT"
    mid_price:     float         # market mid at fill time
    fill_price:    float         # actual fill after slippage
    size_usd:      float         # notional in USD
    size_tokens:   float         # number of tokens
    fee_usd:       float         # transaction fee
    slippage_usd:  float         # cost of market impact
    total_cost_usd: float        # fee + slippage


@dataclass
class BtTrade:
    """A complete round-trip trade (entry fill → exit fill)."""
    trade_id:      int
    condition_id:  str
    question:      str
    side:          str

    entry_time:    str
    exit_time:     str
    entry_price:   float         # mid-price at entry
    exit_price:    float         # mid-price at exit
    entry_fill:    float         # actual fill price (includes slippage)
    exit_fill:     float         # actual fill price at exit
    size_dollars:  float         # notional allocated
    size_tokens:   float

    gross_pnl:     float         # price change × tokens
    total_fees:    float         # entry_fee + exit_fee
    total_slip:    float         # entry_slip + exit_slip
    net_pnl:       float         # gross_pnl - total_fees - total_slip
    net_return:    float         # net_pnl / size_dollars

    exit_reason:   str           # STOP_LOSS | TAKE_PROFIT | SIGNAL_REVERSAL | END_OF_DATA
    confidence:    float
    edge:          float
    hold_bars:     int           # how many data rows the position was held

    def is_win(self) -> bool:
        return self.net_pnl > 0

    def to_row(self) -> dict:
        return {
            "trade_id":    self.trade_id,
            "condition_id": self.condition_id,
            "question":    self.question[:60],
            "side":        self.side,
            "entry_time":  self.entry_time,
            "exit_time":   self.exit_time,
            "entry_price": round(self.entry_price, 4),
            "exit_price":  round(self.exit_price,  4),
            "entry_fill":  round(self.entry_fill,  4),
            "exit_fill":   round(self.exit_fill,   4),
            "size_usd":    round(self.size_dollars, 4),
            "gross_pnl":   round(self.gross_pnl,   4),
            "fees":        round(self.total_fees,   4),
            "slippage":    round(self.total_slip,   4),
            "net_pnl":     round(self.net_pnl,      4),
            "net_return":  round(self.net_return,   4),
            "exit_reason": self.exit_reason,
            "confidence":  round(self.confidence,   3),
            "edge":        round(self.edge,          4),
            "hold_bars":   self.hold_bars,
            "win":         self.is_win(),
        }


@dataclass
class BacktestReport:
    """Aggregated performance report for one or more markets."""
    condition_id:   str
    question:       str

    # ── Core P&L ─────────────────────────────────────────────────────────────
    initial_capital: float
    final_capital:   float
    gross_pnl:       float
    total_fees:      float
    total_slippage:  float
    net_pnl:         float
    roi_pct:         float

    # ── Trade counts ─────────────────────────────────────────────────────────
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    win_rate:        float
    avg_win_usd:     float
    avg_loss_usd:    float
    profit_factor:   float        # gross_wins / gross_losses
    expectancy:      float        # expected net PnL per trade

    # ── Risk metrics ─────────────────────────────────────────────────────────
    max_drawdown:    float        # peak-to-trough as fraction
    max_drawdown_usd: float
    sharpe_ratio:    float        # annualised, daily returns
    sortino_ratio:   float        # downside-only standard deviation
    calmar_ratio:    float        # ROI / max_drawdown
    avg_edge:        float
    avg_confidence:  float

    # ── Costs breakdown ──────────────────────────────────────────────────────
    fee_drag_pct:    float        # fees as % of gross PnL
    slip_drag_pct:   float

    # ── Equity series (for plotting) ─────────────────────────────────────────
    equity_curve:    list[float] = field(default_factory=list)
    trades:          list[BtTrade] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        G = "\033[32m"; R = "\033[31m"; E = "\033[0m"
        pnl_col  = G if self.net_pnl   >= 0 else R
        roi_col  = G if self.roi_pct   >= 0 else R
        wr_col   = G if self.win_rate  >= 0.55 else R
        sh_col   = G if self.sharpe_ratio >= 0.5 else R
        return {
            "condition_id":   self.condition_id[:20],
            "question":       self.question[:60],
            "─ Capital":      "─────────────────────",
            "initial":        f"${self.initial_capital:.2f}",
            "final":          f"${self.final_capital:.2f}",
            "gross_pnl":      f"${self.gross_pnl:+.4f}",
            "total_fees":     f"${self.total_fees:.4f}  ({self.fee_drag_pct:.1f}% of gross)",
            "total_slippage": f"${self.total_slippage:.4f}  ({self.slip_drag_pct:.1f}% of gross)",
            "net_pnl":        f"{pnl_col}${self.net_pnl:+.4f}{E}",
            "roi":            f"{roi_col}{self.roi_pct:+.2f}%{E}",
            "─ Trades":       "─────────────────────",
            "total_trades":   self.total_trades,
            "win/loss":       f"{self.winning_trades} / {self.losing_trades}",
            "win_rate":       f"{wr_col}{self.win_rate:.1%}{E}",
            "avg_win":        f"${self.avg_win_usd:+.4f}",
            "avg_loss":       f"${self.avg_loss_usd:+.4f}",
            "profit_factor":  f"{self.profit_factor:.3f}",
            "expectancy":     f"${self.expectancy:+.4f} / trade",
            "─ Risk":         "─────────────────────",
            "max_drawdown":   f"{self.max_drawdown:.2%}  (${self.max_drawdown_usd:.4f})",
            "sharpe_ratio":   f"{sh_col}{self.sharpe_ratio:.3f}{E}",
            "sortino_ratio":  f"{self.sortino_ratio:.3f}",
            "calmar_ratio":   f"{self.calmar_ratio:.3f}",
            "avg_edge":       f"{self.avg_edge:.4f}",
            "avg_confidence": f"{self.avg_confidence:.3f}",
        }

    def print(self):
        sep = "═" * 65
        print(f"\n{sep}")
        print(f"  BACKTEST REPORT")
        print(f"  {self.question[:60]}")
        print(sep)
        for k, v in self.to_dict().items():
            if k.startswith("─"):
                print(f"\n  {k} {v}")
            else:
                print(f"  {k:<20} {v}")
        print()
        self._print_trade_log(last_n=8)
        print(sep + "\n")

    def _print_trade_log(self, last_n: int = 8):
        trades = self.trades[-last_n:]
        if not trades:
            print("  No trades.\n")
            return
        print(f"\n  Last {len(trades)} trades:")
        print(f"  {'#':>4} {'Side':<10} {'Entry':>7} {'Exit':>7} "
              f"{'Gross':>8} {'Fees':>7} {'Slip':>7} {'Net':>8} {'Exit reason':<18}")
        print("  " + "─" * 82)
        for t in trades:
            marker = "✓" if t.is_win() else "✗"
            print(f"  {t.trade_id:>4} {marker}{t.side:<9} {t.entry_price:>7.4f} {t.exit_price:>7.4f} "
                  f"{t.gross_pnl:>+8.4f} {t.total_fees:>7.4f} {t.total_slip:>7.4f} "
                  f"{t.net_pnl:>+8.4f} {t.exit_reason:<18}")
        print()

    def save_trade_log(self, path: str = "backtest_trades.csv"):
        if not self.trades:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.trades[0].to_row().keys()))
            writer.writeheader()
            for t in self.trades:
                writer.writerow(t.to_row())
        log.info(f"Trade log saved: {path}")
        print(f"  Trade log saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Market-impact slippage model
# ─────────────────────────────────────────────────────────────────────────────

def compute_slippage(
    size_usd:    float,
    market_depth: float,      # 24h volume as proxy for depth
    coeff:       float = SLIPPAGE_COEFF,
    power:       float = SLIPPAGE_POWER,
    direction:   str   = "BUY",
) -> float:
    """
    Square-root market-impact model.
    slippage_fraction = coeff × (size / depth)^power

    Args:
        size_usd:     Order size in USD
        market_depth: Proxy for depth (24h volume)
        coeff:        Calibration constant (0.2 % at 100 % depth)
        power:        Impact exponent (0.5 = square-root model)
        direction:    "BUY" → positive slippage (pay more); "SELL" → negative

    Returns:
        Slippage in USD (always positive cost)
    """
    if market_depth <= 0:
        market_depth = size_usd * 10    # assume thin market: 10x size as depth
    fraction = coeff * (size_usd / max(market_depth, 1)) ** power
    fraction  = min(fraction, 0.05)      # cap at 5 % to prevent absurd impact
    return size_usd * fraction


def compute_liquidity_cap(
    size_usd:    float,
    volume_24h:  float,
    max_frac:    float = MAX_LIQ_FRACTION,
) -> float:
    """
    Cap position size to a fraction of 24h market volume.
    Prevents bot from moving the market against itself.
    """
    liq_cap = volume_24h * max_frac
    return min(size_usd, max(liq_cap, 0.10))


def simulated_fill_price(
    mid:       float,
    spread:    float,
    slip_frac: float,
    side:      str,     # "BUY_YES" or "BUY_NO"
    direction: str,     # "ENTRY" or "EXIT"
) -> float:
    """
    Estimate the actual fill price combining:
      ① Half-spread cost (crossing bid-ask)
      ② Market-impact slippage

    For BUY orders we pay mid + half_spread + slip (worse price)
    For SELL orders we receive mid - half_spread - slip
    """
    half_spread = spread / 2
    # On entry:  BUY_YES = buy YES token; BUY_NO = buy NO token
    # On exit:   we effectively sell the same token
    if direction == "ENTRY":
        if side == "BUY_YES":
            return mid + half_spread + mid * slip_frac
        else:  # BUY_NO: buying the NO token, which means selling YES
            return mid - half_spread - mid * slip_frac
    else:   # EXIT
        if side == "BUY_YES":
            # Selling YES token back
            return mid - half_spread - mid * slip_frac
        else:
            return mid + half_spread + mid * slip_frac


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder from historical CSV row
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_snapshot(row: pd.Series, condition_id: str, question: str) -> MarketSnapshot:
    return MarketSnapshot(
        condition_id  = condition_id,
        question      = question,
        volume        = float(row.get("volume",       10_000)),
        liquidity     = float(row.get("liquidity",    10_000)),
        days_to_close = int(row.get("days_to_close",  30)),
        best_yes_ask  = float(row.get("best_yes_ask", row.get("mid_price", 0.5))),
        best_yes_bid  = float(row.get("best_yes_bid", row.get("mid_price", 0.5))),
        mid_price     = float(row.get("mid_price",    0.5)),
        spread        = float(row.get("spread",       0.02)),
        yes_token_id  = condition_id,
        no_token_id   = condition_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core Backtester
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester with realistic cost model.

    Usage:
        bt = Backtester(fee_rate=0.01, slippage_coeff=0.002)
        report = bt.run_on_dataframe(df, condition_id="abc", question="Will X?")
        report.print()
        report.save_trade_log("results/abc_trades.csv")

    Key parameters:
        fee_rate:         Taker fee as fraction of notional (default 1 %)
        slippage_coeff:   Market-impact calibration constant (default 0.2 %)
        liquidity_cap:    Max position as fraction of 24h volume (default 5 %)
        position_pct:     Capital fraction per trade (default 2 %)
        warmup_rows:      Rows needed before signals trusted (default 15)
        use_news:         Whether to call Gemini news signal (slow; default False in BT)
    """

    def __init__(
        self,
        data_store:       HistoricalStore = store,
        fee_rate:          float = TAKER_FEE_RATE,
        slippage_coeff:    float = SLIPPAGE_COEFF,
        slippage_power:    float = SLIPPAGE_POWER,
        liquidity_cap:     float = MAX_LIQ_FRACTION,
        position_pct:      float = 0.02,
        warmup_rows:       int   = 15,
        use_news:          bool  = False,     # disable slow Gemini calls in BT
        initial_capital:   float = CAPITAL,
    ):
        self.store           = data_store
        self.fee_rate        = fee_rate
        self.slippage_coeff  = slippage_coeff
        self.slippage_power  = slippage_power
        self.liquidity_cap   = liquidity_cap
        self.position_pct    = position_pct
        self.warmup_rows     = warmup_rows
        self.use_news        = use_news
        self.initial_capital = initial_capital

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        condition_id: str,
        question:     str = "Unknown market",
    ) -> BacktestReport:
        """Load data from store and run backtest for one market."""
        df = self.store.load(condition_id)
        if df.empty or len(df) < self.warmup_rows + 5:
            log.warning(f"Insufficient data ({len(df)} rows) for: {condition_id[:20]}")
            return self._empty_report(condition_id, question)
        return self.run_on_dataframe(df, condition_id, question)

    def run_on_dataframe(
        self,
        df:           pd.DataFrame,
        condition_id: str = "synthetic",
        question:     str = "Unknown market",
    ) -> BacktestReport:
        """
        Execute the full walk-forward backtest on a pre-loaded DataFrame.

        The DataFrame must contain at minimum:
            mid_price, volume, liquidity, spread, days_to_close
        """
        if df.empty or len(df) < self.warmup_rows + 5:
            return self._empty_report(condition_id, question)

        capital      = self.initial_capital
        trades:      list[BtTrade] = []
        equity_curve = [capital]
        position     = None
        trade_id     = 0

        log.info(f"Backtesting {condition_id[:20]} | {len(df)} bars | "
                 f"fee={self.fee_rate:.1%} slip_coeff={self.slippage_coeff:.4f}")

        for i in range(self.warmup_rows, len(df)):
            row     = df.iloc[i]
            history = df.iloc[:i]
            snap    = _row_to_snapshot(row, condition_id, question)

            # ── Run strategy engine (news disabled for speed in BT) ───────────
            combined = self._run_signals(snap, history)
            mid      = snap.mid_price
            vol      = snap.volume          # 24h volume used as depth proxy
            spread   = snap.spread

            # ── EXIT check ────────────────────────────────────────────────────
            if position is not None:
                entry     = position["entry_price"]
                side      = position["side"]
                sl        = position["stop_loss"]
                tp        = position["take_profit"]
                size_usd  = position["size_usd"]
                n_tok     = position["n_tokens"]
                entry_bar = position["entry_bar"]

                exit_reason = self._check_exit(mid, side, sl, tp, combined, i, len(df))

                if exit_reason:
                    # Compute exit slippage + fill
                    slip_usd   = compute_slippage(size_usd, vol, self.slippage_coeff, self.slippage_power)
                    slip_frac  = slip_usd / max(size_usd, 1e-9)
                    exit_fill  = simulated_fill_price(mid, spread, slip_frac, side, "EXIT")
                    exit_fill  = max(0.001, min(0.999, exit_fill))
                    exit_fee   = size_usd * self.fee_rate

                    # Gross PnL based on actual fill prices
                    entry_fill = position["entry_fill"]
                    if side == "BUY_YES":
                        gross_pnl  = (exit_fill - entry_fill) * n_tok
                    else:
                        gross_pnl  = (entry_fill - exit_fill) * n_tok

                    total_fees = position["entry_fee"] + exit_fee
                    total_slip = position["entry_slip"] + slip_usd
                    net_pnl    = gross_pnl - total_fees - total_slip

                    capital   += size_usd + net_pnl   # return capital + net P&L
                    capital    = max(capital, 0.0)

                    trade_id += 1
                    trades.append(BtTrade(
                        trade_id      = trade_id,
                        condition_id  = condition_id,
                        question      = question,
                        side          = side,
                        entry_time    = position["entry_time"],
                        exit_time     = str(row.get("fetched_at", i)),
                        entry_price   = entry,
                        exit_price    = mid,
                        entry_fill    = entry_fill,
                        exit_fill     = exit_fill,
                        size_dollars  = size_usd,
                        size_tokens   = n_tok,
                        gross_pnl     = round(gross_pnl, 6),
                        total_fees    = round(total_fees, 6),
                        total_slip    = round(total_slip, 6),
                        net_pnl       = round(net_pnl, 6),
                        net_return    = round(net_pnl / max(size_usd, 1e-9), 6),
                        exit_reason   = exit_reason,
                        confidence    = position["confidence"],
                        edge          = position["edge"],
                        hold_bars     = i - entry_bar,
                    ))
                    equity_curve.append(capital)
                    position = None
                    continue

            # ── ENTRY check ───────────────────────────────────────────────────
            if position is None and combined.is_actionable and capital > 0.10:
                raw_size    = capital * self.position_pct
                liq_size    = compute_liquidity_cap(raw_size, vol, self.liquidity_cap)
                size_usd    = min(liq_size, MAX_POSITION_SIZE, capital * 0.20)

                if size_usd < 0.05:
                    continue

                # Entry fill price (slippage applied at entry)
                slip_usd    = compute_slippage(size_usd, vol, self.slippage_coeff, self.slippage_power)
                slip_frac   = slip_usd / max(size_usd, 1e-9)
                entry_fill  = simulated_fill_price(mid, spread, slip_frac, combined.direction, "ENTRY")
                entry_fill  = max(0.001, min(0.999, entry_fill))
                entry_fee   = size_usd * self.fee_rate
                n_tok       = size_usd / max(entry_fill, 0.001)

                sl, tp = self._stops(combined.direction, mid)
                capital -= size_usd   # reserve capital for the position

                position = {
                    "entry_price":  mid,
                    "entry_fill":   entry_fill,
                    "entry_fee":    entry_fee,
                    "entry_slip":   slip_usd,
                    "side":         combined.direction,
                    "size_usd":     size_usd,
                    "n_tokens":     n_tok,
                    "stop_loss":    sl,
                    "take_profit":  tp,
                    "entry_time":   str(row.get("fetched_at", i)),
                    "confidence":   combined.confidence,
                    "edge":         combined.edge,
                    "entry_bar":    i,
                }
                equity_curve.append(capital)

        # ── Aggregate metrics ─────────────────────────────────────────────────
        return self._build_report(condition_id, question, capital, trades, equity_curve)

    def run_all(self, verbose: bool = True) -> list[BacktestReport]:
        """Run backtest over every market in the data store."""
        reports = []
        market_ids = self.store.list_markets()
        log.info(f"Running backtest on {len(market_ids)} markets")
        for cid in market_ids:
            rep = self.run(cid)
            reports.append(rep)
            if verbose:
                rep.print()
        return reports

    def run_on_synthetic(self, n_bars: int = 500) -> BacktestReport:
        """
        Run a backtest on procedurally generated price data.
        Useful for verifying the engine without needing real data.
        """
        df = _generate_synthetic_ohlcv(n_bars=n_bars)
        return self.run_on_dataframe(df, "synthetic", "Synthetic market: will price exceed 0.60?")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_signals(self, snap, history):
        """Run strategy engine, suppressing slow Gemini calls during backtests."""
        if self.use_news:
            return run_all_strategies(snap, history)

        # Fast path: patch out slow signals
        import unittest.mock as mock
        dummy = lambda *a, **kw: __import__("polybot.strategies", fromlist=["SignalResult"]).SignalResult(
            strategy="news", yes_probability=0.5, metadata={"reason": "disabled_in_bt"}
        )
        from polybot import strategies as strat_mod
        with mock.patch.object(strat_mod, "strategy_news", dummy), \
             mock.patch.object(strat_mod, "strategy_reddit", dummy):
            return run_all_strategies(snap, history)

    @staticmethod
    def _stops(side: str, price: float) -> tuple[float, float]:
        if side == "BUY_YES":
            sl = max(0.01, price * (1 - STOP_LOSS_PCT))
            tp = min(0.99, price * (1 + TAKE_PROFIT_PCT))
        else:
            sl = min(0.99, price * (1 + STOP_LOSS_PCT))
            tp = max(0.01, price * (1 - TAKE_PROFIT_PCT))
        return round(sl, 4), round(tp, 4)

    @staticmethod
    def _check_exit(price, side, sl, tp, combined, i, n_rows) -> Optional[str]:
        if side == "BUY_YES":
            if price <= sl:                                          return "STOP_LOSS"
            if price >= tp:                                          return "TAKE_PROFIT"
        else:
            if price >= sl:                                          return "STOP_LOSS"
            if price <= tp:                                          return "TAKE_PROFIT"
        if i == n_rows - 1:                                          return "END_OF_DATA"
        if combined.direction not in ("HOLD", side) and abs(combined.edge) > 0.08:
            return "SIGNAL_REVERSAL"
        return None

    def _build_report(
        self,
        condition_id: str,
        question:     str,
        final_cap:    float,
        trades:       list[BtTrade],
        equity_curve: list[float],
    ) -> BacktestReport:

        wins   = [t for t in trades if t.is_win()]
        losses = [t for t in trades if not t.is_win()]

        gross_pnl   = sum(t.gross_pnl   for t in trades)
        total_fees  = sum(t.total_fees  for t in trades)
        total_slip  = sum(t.total_slip  for t in trades)
        net_pnl     = sum(t.net_pnl     for t in trades)
        roi_pct     = net_pnl / self.initial_capital * 100

        avg_win  = float(np.mean([t.net_pnl for t in wins]))   if wins   else 0.0
        avg_loss = float(np.mean([t.net_pnl for t in losses])) if losses else 0.0

        gross_wins   = sum(t.net_pnl for t in wins)   if wins   else 0.0
        gross_losses = abs(sum(t.net_pnl for t in losses)) if losses else 1e-9
        profit_factor = gross_wins / max(gross_losses, 1e-9)

        expectancy = net_pnl / max(len(trades), 1)

        returns    = [t.net_return for t in trades]
        sharpe     = _sharpe(returns)
        sortino    = _sortino(returns)
        mdd, mdd_usd = _max_drawdown(equity_curve)
        calmar     = (roi_pct / 100) / max(mdd, 1e-9)

        fee_drag   = total_fees  / max(abs(gross_pnl), 1e-9) * 100
        slip_drag  = total_slip  / max(abs(gross_pnl), 1e-9) * 100

        avg_edge   = float(np.mean([t.edge       for t in trades])) if trades else 0.0
        avg_conf   = float(np.mean([t.confidence for t in trades])) if trades else 0.0

        return BacktestReport(
            condition_id    = condition_id,
            question        = question,
            initial_capital = round(self.initial_capital, 4),
            final_capital   = round(final_cap, 4),
            gross_pnl       = round(gross_pnl,  6),
            total_fees      = round(total_fees,  6),
            total_slippage  = round(total_slip,  6),
            net_pnl         = round(net_pnl,     6),
            roi_pct         = round(roi_pct,     4),
            total_trades    = len(trades),
            winning_trades  = len(wins),
            losing_trades   = len(losses),
            win_rate        = len(wins) / max(len(trades), 1),
            avg_win_usd     = round(avg_win,     6),
            avg_loss_usd    = round(avg_loss,    6),
            profit_factor   = round(profit_factor, 4),
            expectancy      = round(expectancy,    6),
            max_drawdown    = round(mdd,           6),
            max_drawdown_usd = round(mdd_usd,      4),
            sharpe_ratio    = round(sharpe,        4),
            sortino_ratio   = round(sortino,       4),
            calmar_ratio    = round(calmar,        4),
            avg_edge        = round(avg_edge,      6),
            avg_confidence  = round(avg_conf,      4),
            fee_drag_pct    = round(fee_drag,      2),
            slip_drag_pct   = round(slip_drag,     2),
            equity_curve    = equity_curve,
            trades          = trades,
        )

    def _empty_report(self, condition_id: str, question: str) -> BacktestReport:
        return BacktestReport(
            condition_id=condition_id, question=question,
            initial_capital=self.initial_capital, final_capital=self.initial_capital,
            gross_pnl=0, total_fees=0, total_slippage=0, net_pnl=0, roi_pct=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win_usd=0, avg_loss_usd=0, profit_factor=0, expectancy=0,
            max_drawdown=0, max_drawdown_usd=0, sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, avg_edge=0, avg_confidence=0, fee_drag_pct=0, slip_drag_pct=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr  = np.array(returns, dtype=float)
    mean = arr.mean() - risk_free
    std  = arr.std(ddof=1)
    return float(mean / std * math.sqrt(252)) if std > 1e-9 else 0.0


def _sortino(returns: list[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr     = np.array(returns, dtype=float)
    mean    = arr.mean() - risk_free
    negrets = arr[arr < 0]
    if len(negrets) < 2:
        return float("inf") if mean > 0 else 0.0
    downside_std = float(np.std(negrets, ddof=1))
    return float(mean / downside_std * math.sqrt(252)) if downside_std > 1e-9 else 0.0


def _max_drawdown(equity: list[float]) -> tuple[float, float]:
    if not equity:
        return 0.0, 0.0
    arr  = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / np.where(peak > 0, peak, 1.0)
    mdd  = float(abs(dd.min()))
    mdd_usd = float((peak * dd).min())      # max dollar drawdown
    return mdd, abs(mdd_usd)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for engine validation)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_ohlcv(
    n_bars:    int   = 500,
    seed:      int   = 42,
    p0:        float = 0.50,
    drift:     float = 0.001,
    vol:       float = 0.015,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic Polymarket price series combining:
      - Random-walk component
      - Momentum regime (trending for ~20 bars then reverting)
      - Volume spikes on price moves
    """
    rng = np.random.default_rng(seed)
    prices  = [p0]
    regime  = 1        # 1 = uptrend, -1 = downtrend
    reg_cnt = 0

    for _ in range(n_bars - 1):
        reg_cnt += 1
        if reg_cnt > rng.integers(15, 40):
            regime  = -regime
            reg_cnt = 0
        shocks = rng.normal(drift * regime, vol)
        p = max(0.01, min(0.99, prices[-1] + shocks))
        prices.append(p)

    prices_arr = np.array(prices)
    volumes    = rng.lognormal(mean=9.5, sigma=1.5, size=n_bars) + 1_000
    # Volume spikes when price moves fast
    price_moves = np.abs(np.diff(prices_arr, prepend=prices_arr[0]))
    volumes    *= (1 + price_moves * 20)

    spreads    = np.clip(rng.uniform(0.01, 0.08, n_bars), 0.005, 0.15)
    liquidity  = volumes * rng.uniform(0.9, 2.5, n_bars)
    days       = np.linspace(90, 1, n_bars).clip(min=1)

    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="4h")

    return pd.DataFrame({
        "fetched_at":    timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mid_price":     prices_arr,
        "best_yes_bid":  (prices_arr - spreads / 2).clip(0.001, 0.999),
        "best_yes_ask":  (prices_arr + spreads / 2).clip(0.001, 0.999),
        "spread":        spreads,
        "volume":        volumes,
        "liquidity":     liquidity,
        "days_to_close": days.astype(int),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_reports(reports: list[BacktestReport]) -> dict:
    """
    Combine multiple single-market reports into portfolio-level statistics.
    """
    valid = [r for r in reports if r.total_trades > 0]
    if not valid:
        return {"error": "No trades across all markets"}

    all_trades    = [t for r in valid for t in r.trades]
    total_net_pnl = sum(r.net_pnl for r in valid)
    total_fees    = sum(r.total_fees for r in valid)
    total_slip    = sum(r.total_slippage for r in valid)
    total_gross   = sum(r.gross_pnl for r in valid)
    wins          = [t for t in all_trades if t.is_win()]
    all_returns   = [t.net_return for t in all_trades]

    return {
        "markets_tested":   len(reports),
        "markets_with_trades": len(valid),
        "total_trades":     len(all_trades),
        "total_wins":       len(wins),
        "portfolio_win_rate": f"{len(wins)/max(len(all_trades),1):.1%}",
        "total_gross_pnl":  f"${total_gross:+.4f}",
        "total_fees":       f"${total_fees:.4f}",
        "total_slippage":   f"${total_slip:.4f}",
        "total_net_pnl":    f"${total_net_pnl:+.4f}",
        "portfolio_roi":    f"{total_net_pnl / CAPITAL * 100:+.2f}%",
        "portfolio_sharpe": f"{_sharpe(all_returns):.3f}",
        "best_market":      max(valid, key=lambda r: r.net_pnl).condition_id[:20],
        "worst_market":     min(valid, key=lambda r: r.net_pnl).condition_id[:20],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

backtester = Backtester()
