"""
polybot/metrics.py
==================
Performance Metrics Module — tracks real-time trading performance and
computes key statistics:
  - Win rate
  - Total ROI %
  - Sharpe ratio (annualised)
  - Max drawdown
  - Profit factor
  - Average trade duration
  - Daily P&L summary
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from polybot.config import CAPITAL
from polybot.logger import get_logger

log = get_logger("metrics")


@dataclass
class TradeRecord:
    """Immutable record of a completed trade."""
    open_time:    str
    close_time:   str
    question:     str
    side:         str
    entry_price:  float
    exit_price:   float
    size_dollars: float
    pnl:          float
    pnl_pct:      float
    exit_reason:  str
    confidence:   float
    edge:         float


class PerformanceTracker:
    """
    Maintains a running ledger of all completed trades and computes
    cumulative performance statistics on demand.
    """

    def __init__(self, log_path: str = "data/trade_log.json", capital: float = CAPITAL):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.capital  = capital
        self.trades:  list[TradeRecord] = []
        self._equity: list[float]       = [capital]
        self._load()

    def record(
        self,
        question:     str,
        side:         str,
        entry_price:  float,
        exit_price:   float,
        size_dollars: float,
        pnl:          float,
        exit_reason:  str,
        confidence:   float = 0.0,
        edge:         float = 0.0,
        open_time:    str   = "",
    ):
        """Add a completed trade to the ledger."""
        pnl_pct = pnl / max(size_dollars, 0.001)
        self._equity.append(self._equity[-1] + pnl)

        t = TradeRecord(
            open_time    = open_time or datetime.utcnow().isoformat(),
            close_time   = datetime.utcnow().isoformat(),
            question     = question[:80],
            side         = side,
            entry_price  = round(entry_price, 4),
            exit_price   = round(exit_price, 4),
            size_dollars = round(size_dollars, 2),
            pnl          = round(pnl, 4),
            pnl_pct      = round(pnl_pct, 4),
            exit_reason  = exit_reason,
            confidence   = round(confidence, 3),
            edge         = round(edge, 4),
        )
        self.trades.append(t)
        self._save()
        log.info(
            f"Trade recorded | {side} | pnl=${pnl:+.2f} ({pnl_pct:+.1%}) | "
            f"reason={exit_reason} | total_trades={len(self.trades)}"
        )

    # ── Core Metrics ──────────────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> list[TradeRecord]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losses(self) -> list[TradeRecord]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.wins) / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def roi_pct(self) -> float:
        return self.total_pnl / self.capital * 100

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.wins)
        gross_loss   = abs(sum(t.pnl for t in self.losses))
        return round(gross_profit / max(gross_loss, 0.001), 3)

    @property
    def avg_win(self) -> float:
        return sum(t.pnl for t in self.wins) / max(len(self.wins), 1)

    @property
    def avg_loss(self) -> float:
        return sum(t.pnl for t in self.losses) / max(len(self.losses), 1)

    @property
    def expectancy(self) -> float:
        """Expected P&L per trade = (win_rate * avg_win) + (loss_rate * avg_loss)."""
        loss_rate = 1 - self.win_rate
        return self.win_rate * self.avg_win + loss_rate * self.avg_loss

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio on per-trade returns."""
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl_pct for t in self.trades])
        mean    = np.mean(returns)
        std     = np.std(returns, ddof=1)
        return float(mean / std * math.sqrt(252)) if std > 1e-9 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown as a fraction."""
        if len(self._equity) < 2:
            return 0.0
        arr  = np.array(self._equity)
        peak = np.maximum.accumulate(arr)
        dd   = (arr - peak) / np.where(peak > 0, peak, 1.0)
        return float(abs(dd.min()))

    @property
    def best_trade(self) -> Optional[TradeRecord]:
        return max(self.trades, key=lambda t: t.pnl) if self.trades else None

    @property
    def worst_trade(self) -> Optional[TradeRecord]:
        return min(self.trades, key=lambda t: t.pnl) if self.trades else None

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "total_trades":  self.total_trades,
            "win_rate":      f"{self.win_rate:.1%}",
            "total_pnl":     f"${self.total_pnl:+.2f}",
            "roi":           f"{self.roi_pct:+.2f}%",
            "sharpe_ratio":  f"{self.sharpe_ratio:.3f}",
            "max_drawdown":  f"{self.max_drawdown:.1%}",
            "profit_factor": self.profit_factor,
            "expectancy":    f"${self.expectancy:+.4f}",
            "avg_win":       f"${self.avg_win:+.4f}",
            "avg_loss":      f"${self.avg_loss:+.4f}",
        }

    def print_summary(self):
        sep = "─" * 55
        print(f"\n{sep}")
        print("  PERFORMANCE METRICS")
        print(sep)
        for k, v in self.summary().items():
            print(f"  {k:<20} {v}")
        if self.best_trade:
            print(f"\n  Best trade:  ${self.best_trade.pnl:+.2f}  {self.best_trade.question[:40]}")
        if self.worst_trade:
            print(f"  Worst trade: ${self.worst_trade.pnl:+.2f}  {self.worst_trade.question[:40]}")
        print(sep)

    def recent_trades(self, n: int = 10) -> list[dict]:
        return [
            {
                "time":    t.close_time,
                "side":    t.side,
                "pnl":     f"${t.pnl:+.3f}",
                "reason":  t.exit_reason,
                "q":       t.question[:50],
            }
            for t in self.trades[-n:]
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        data = {
            "capital": self.capital,
            "equity":  self._equity,
            "trades":  [t.__dict__ for t in self.trades],
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, encoding="utf-8") as f:
                data = json.load(f)
            self.capital  = data.get("capital", CAPITAL)
            self._equity  = data.get("equity", [self.capital])
            self.trades   = [TradeRecord(**t) for t in data.get("trades", [])]
            log.info(f"Loaded {len(self.trades)} historical trades from {self.log_path}")
        except Exception as exc:
            log.warning(f"Could not load trade log: {exc}")


# Module-level singleton
perf = PerformanceTracker()
