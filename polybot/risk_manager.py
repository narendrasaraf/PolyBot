"""
polybot/risk_manager.py
=======================
Risk Management layer — enforces all capital protection rules before
any trade reaches the executor.

Rules enforced:
  1. Max 2% capital per trade
  2. Daily loss limit = 5% of capital
  3. Daily profit limit = 15% (stop overtrading)
  4. Liquidity gate (skip thin markets)
  5. Spread gate (skip wide-spread markets)
  6. Stop-loss per position (5%)
  7. Take-profit targets (5–15%)
  8. Kelly Criterion position sizing
  9. Max concurrent positions
"""

from __future__ import annotations

import json
from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from polybot.config import (
    CAPITAL, MAX_POSITION_SIZE, DAILY_LOSS_LIMIT, DAILY_PROFIT_LIMIT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MIN_TAKE_PROFIT_PCT, MAX_TAKE_PROFIT_PCT,
    MIN_LIQUIDITY_USD, MAX_SPREAD, SLIPPAGE_TOLERANCE,
)
from polybot.logger import get_logger

log = get_logger("risk")

MAX_CONCURRENT_POSITIONS = 5


# ── Open position record ──────────────────────────────────────────────────────

@dataclass
class Position:
    """Tracks a single open trade."""
    condition_id:  str
    question:      str
    token_id:      str
    side:          str                 # "BUY_YES" | "BUY_NO"
    entry_price:   float
    size_tokens:   float               # number of tokens held
    size_dollars:  float               # USD capital risked
    stop_loss:     float               # price to auto-exit at loss
    take_profit:   float               # price to auto-exit at gain
    entry_time:    str = field(default_factory=lambda: datetime.utcnow().isoformat())
    order_id:      str = ""

    @property
    def unrealised_pnl(self) -> float:
        """Placeholder — inject current_price via pnl_at() for real P&L."""
        return 0.0

    def pnl_at(self, current_price: float) -> float:
        factor = 1 if self.side == "BUY_YES" else -1
        return (current_price - self.entry_price) * self.size_tokens * factor

    def should_stop_loss(self, current_price: float) -> bool:
        if self.side == "BUY_YES":
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss      # for NO tokens, loss = price rise

    def should_take_profit(self, current_price: float) -> bool:
        if self.side == "BUY_YES":
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def to_dict(self) -> dict:
        return self.__dict__


# ── Daily accounting state ─────────────────────────────────────────────────────

@dataclass
class DailyState:
    date:             date  = field(default_factory=date.today)
    realised_pnl:     float = 0.0
    trades:           int   = 0
    wins:             int   = 0
    losses:           int   = 0
    open_positions:   dict[str, Position] = field(default_factory=dict)
    trade_history:    list[dict]          = field(default_factory=list)

    def reset_if_new_day(self):
        if date.today() != self.date:
            log.info(
                f"New day — resetting daily state | "
                f"Yesterday P&L: ${self.realised_pnl:.2f} | "
                f"Win rate: {self.win_rate:.0%}"
            )
            self.date = date.today()
            self.realised_pnl = 0.0
            self.trades = 0
            self.wins   = 0
            self.losses = 0
            # Do NOT clear open positions — they persist across days

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return self.wins / closed if closed else 0.0

    def record_trade(self, pnl: float, question: str, side: str, entry: float, exit_: float):
        self.realised_pnl += pnl
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.trade_history.append({
            "time":  datetime.utcnow().isoformat(),
            "question": question[:60],
            "side":  side,
            "entry": entry,
            "exit":  exit_,
            "pnl":   round(pnl, 4),
        })

    def can_trade(self) -> tuple[bool, str]:
        self.reset_if_new_day()
        if self.realised_pnl <= -DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit hit (${self.realised_pnl:.2f}). Trading halted."
        if self.realised_pnl >= DAILY_PROFIT_LIMIT:
            return False, f"Daily profit target reached (+${self.realised_pnl:.2f}). Paused."
        if len(self.open_positions) >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached."
        return True, "OK"

    def save_state(self, path: str = "data/daily_state.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "date":          str(self.date),
            "realised_pnl":  self.realised_pnl,
            "trades":        self.trades,
            "wins":          self.wins,
            "losses":        self.losses,
            "open_positions": {k: v.to_dict() for k, v in self.open_positions.items()},
            "trade_history": self.trade_history,
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)

    def load_state(self, path: str = "data/daily_state.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            saved_date = date.fromisoformat(data["date"])
            if saved_date == date.today():
                self.date         = saved_date
                self.realised_pnl = data.get("realised_pnl", 0)
                self.trades       = data.get("trades", 0)
                self.wins         = data.get("wins", 0)
                self.losses       = data.get("losses", 0)
                self.trade_history = data.get("trade_history", [])
                # Restore open positions — strip non-field keys (e.g. computed properties)
                _pos_fields = {f.name for f in Position.__dataclass_fields__.values()}
                for cid, pd_ in data.get("open_positions", {}).items():
                    filtered = {k: v for k, v in pd_.items() if k in _pos_fields}
                    self.open_positions[cid] = Position(**filtered)
                log.info(f"Loaded previous state: {self.trades} trades, P&L=${self.realised_pnl:.2f}")
        except FileNotFoundError:
            log.info("No previous state found — starting fresh.")
        except Exception as exc:
            log.warning(f"Could not load state: {exc}")


# ── Singleton state ────────────────────────────────────────────────────────────
state = DailyState()


# ── Risk Manager ───────────────────────────────────────────────────────────────

class RiskManager:
    """
    Wraps DailyState and enforces all pre-trade checks and position sizing.
    """

    def __init__(self, daily_state: DailyState = state):
        self.state = daily_state

    # ── Pre-trade gates ───────────────────────────────────────────────────────

    def check_market_liquidity(self, snap) -> tuple[bool, str]:
        if snap.volume < MIN_LIQUIDITY_USD:
            return False, f"Low volume: ${snap.volume:.0f} < ${MIN_LIQUIDITY_USD:.0f}"
        if snap.liquidity < MIN_LIQUIDITY_USD:
            return False, f"Low liquidity: ${snap.liquidity:.0f} < ${MIN_LIQUIDITY_USD:.0f}"
        if snap.spread > MAX_SPREAD:
            return False, f"Wide spread: {snap.spread:.3f} > {MAX_SPREAD}"
        return True, "OK"

    def check_pre_trade(self, snap, signal) -> tuple[bool, str]:
        """Full pre-trade gate. Returns (allowed, reason)."""
        # 1. Daily limits
        ok, reason = self.state.can_trade()
        if not ok:
            return False, reason

        # 2. Already in position
        if snap.condition_id in self.state.open_positions:
            return False, "Already holding position in this market."

        # 3. Market must be tradeable
        if not snap.is_tradeable:
            ok2, reason2 = self.check_market_liquidity(snap)
            return False, f"Market not tradeable: {reason2}"

        # 4. Signal must be actionable
        if not signal.is_actionable:
            return False, (
                f"Signal not actionable: confidence={signal.confidence:.2f} "
                f"edge={signal.edge:.4f} direction={signal.direction}"
            )

        return True, "OK"

    # ── Position sizing ───────────────────────────────────────────────────────

    def kelly_size(self, edge: float, win_prob: float) -> float:
        """
        Kelly Criterion: f* = (b*p - q) / b
          b = odds of winning (edge / (1 - edge))
          p = estimated win probability
          q = 1 - p
        Cap at 2% of capital per trade (production safety).
        """
        if edge <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
        b = edge / max(1 - edge, 0.001)
        q = 1.0 - win_prob
        kelly_f = (b * win_prob - q) / b
        kelly_f = max(0.0, kelly_f)
        half_kelly = kelly_f * 0.50      # use half-Kelly for safety
        position_usd = CAPITAL * half_kelly
        return round(min(position_usd, MAX_POSITION_SIZE), 2)

    def compute_stops(self, side: str, entry_price: float) -> tuple[float, float]:
        """
        Compute stop-loss and take-profit prices.
        Returns (stop_loss_price, take_profit_price).
        """
        if side == "BUY_YES":
            sl = max(0.01, entry_price * (1 - STOP_LOSS_PCT))
            tp = min(0.99, entry_price * (1 + TAKE_PROFIT_PCT))
        else:   # BUY_NO — YES price should fall
            sl = min(0.99, entry_price * (1 + STOP_LOSS_PCT))     # stop if YES rises
            tp = max(0.01, entry_price * (1 - TAKE_PROFIT_PCT))   # profit if YES falls
        return round(sl, 4), round(tp, 4)

    def build_position(self, snap, signal, size_dollars: float) -> Position:
        """Create a Position object from a signal and sizing decision."""
        entry = snap.mid_price
        side  = signal.direction
        token_id = snap.yes_token_id if side == "BUY_YES" else snap.no_token_id
        if not token_id:
            token_id = snap.yes_token_id or snap.condition_id
        sl, tp = self.compute_stops(side, entry)
        size_tokens = round(size_dollars / max(entry, 0.01), 4)
        return Position(
            condition_id = snap.condition_id,
            question     = snap.question,
            token_id     = token_id,
            side         = side,
            entry_price  = entry,
            size_tokens  = size_tokens,
            size_dollars = size_dollars,
            stop_loss    = sl,
            take_profit  = tp,
        )

    # ── Position monitoring ───────────────────────────────────────────────────

    def get_exit_reason(self, pos: Position, current_price: float) -> Optional[str]:
        """
        Determine if a position should be exited.
        Returns None if no exit condition met.
        """
        if pos.should_stop_loss(current_price):
            pnl = pos.pnl_at(current_price)
            return f"STOP_LOSS (price={current_price:.3f}, SL={pos.stop_loss:.3f}, pnl=${pnl:.2f})"
        if pos.should_take_profit(current_price):
            pnl = pos.pnl_at(current_price)
            return f"TAKE_PROFIT (price={current_price:.3f}, TP={pos.take_profit:.3f}, pnl=${pnl:.2f})"
        return None

    def close_position(self, pos: Position, exit_price: float, reason: str):
        """Record closure and remove from open positions."""
        pnl = pos.pnl_at(exit_price)
        self.state.record_trade(
            pnl=pnl,
            question=pos.question,
            side=pos.side,
            entry=pos.entry_price,
            exit_=exit_price,
        )
        if pos.condition_id in self.state.open_positions:
            del self.state.open_positions[pos.condition_id]
        log.info(
            f"Position closed | {reason} | "
            f"P&L: ${pnl:+.2f} | daily P&L: ${self.state.realised_pnl:.2f}"
        )
        self.state.save_state()
        return pnl

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        s = self.state
        return (
            f"Trades: {s.trades} | Wins: {s.wins} | Losses: {s.losses} | "
            f"Win rate: {s.win_rate:.0%} | P&L: ${s.realised_pnl:+.2f} | "
            f"Open positions: {len(s.open_positions)}"
        )


# Module-level singleton
risk = RiskManager()
