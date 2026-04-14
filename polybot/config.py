"""
polybot/config.py
=================
Central configuration hub for PolyBot.
All environment variables and tunable constants live here.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ── API Credentials ───────────────────────────────────────────────────────────
CLOB_API_KEY      = os.getenv("CLOB_API_KEY", "")
CLOB_SECRET       = os.getenv("CLOB_SECRET", "")
CLOB_PASSPHRASE   = os.getenv("CLOB_PASSPHRASE", "")
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY", "")          # NewsAPI.org (optional)
GNEWS_KEY         = os.getenv("GNEWS_KEY", "")            # GNews.io (optional)

# ── API URLs ──────────────────────────────────────────────────────────────────
CLOB_HOST         = "https://clob.polymarket.com"
GAMMA_HOST        = "https://gamma-api.polymarket.com"
POLYMARKET_HOST   = CLOB_HOST                             # backwards compat alias

# ── Capital & Risk Parameters ─────────────────────────────────────────────────
CAPITAL           = float(os.getenv("CAPITAL", "10.0"))
MAX_POSITION_PCT  = 0.02          # 2% max per trade (production rule)
DAILY_LOSS_LIMIT_PCT  = 0.05     # 5% max daily loss
DAILY_PROFIT_LIMIT_PCT = 0.15    # 15% daily profit cap (reduce overtrading)
STOP_LOSS_PCT     = 0.05          # 5% stop-loss per position
TAKE_PROFIT_PCT   = 0.10          # 10% default take-profit
MIN_TAKE_PROFIT_PCT = 0.05        # 5% minimum take-profit trigger
MAX_TAKE_PROFIT_PCT = 0.15        # 15% maximum take-profit target

# Derived dollar amounts
MAX_POSITION_SIZE  = CAPITAL * MAX_POSITION_PCT
DAILY_LOSS_LIMIT   = CAPITAL * DAILY_LOSS_LIMIT_PCT
DAILY_PROFIT_LIMIT = CAPITAL * DAILY_PROFIT_LIMIT_PCT

# ── Signal & Filter Thresholds ────────────────────────────────────────────────
MIN_CONFIDENCE    = 0.60          # minimum model confidence to trade
MIN_EDGE          = 0.06          # minimum price edge (6 cents)
MIN_LIQUIDITY_USD = 5_000.0       # skip markets with < $5K volume
MIN_DAYS_TO_CLOSE = 2             # skip markets closing in < 2 days
MAX_SPREAD        = 0.15          # skip if bid-ask spread > 15 cents

# ── Execution ─────────────────────────────────────────────────────────────────
SLIPPAGE_TOLERANCE = 0.02         # accept 2% worse than target price
SCAN_INTERVAL     = 20            # seconds between full market scans
MAX_MARKETS_SCAN  = 25            # how many markets to pull per scan
TOP_MARKETS_TRADE = 10            # how many to analyse in depth per cycle
ORDER_TIMEOUT_SEC = 10            # seconds to wait for order fill

# ── Engine Settings ───────────────────────────────────────────────────────────
PAPER_MODE        = os.getenv("PAPER_MODE", "true").lower() == "true"
GEMINI_MODEL      = "gemini-2.0-flash"
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
DATA_DIR          = os.getenv("DATA_DIR", "data")          # historical CSV storage

# ── Strategy Weights (must sum to 1.0) ───────────────────────────────────────
SIGNAL_WEIGHTS = {
    "momentum":    0.20,
    "mean_rev":    0.15,
    "prob_gap":    0.20,
    "news":        0.20,
    "reddit":      0.10,
    "ml":          0.10,
    "base_rate":   0.05,
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, "Signal weights must sum to 1.0"


@dataclass
class BotConfig:
    """Runtime config snapshot — pass around instead of globals."""
    capital: float           = CAPITAL
    paper_mode: bool         = PAPER_MODE
    max_position_size: float = MAX_POSITION_SIZE
    daily_loss_limit: float  = DAILY_LOSS_LIMIT
    daily_profit_limit: float = DAILY_PROFIT_LIMIT
    stop_loss_pct: float     = STOP_LOSS_PCT
    take_profit_pct: float   = TAKE_PROFIT_PCT
    min_confidence: float    = MIN_CONFIDENCE
    min_edge: float          = MIN_EDGE
    min_liquidity_usd: float = MIN_LIQUIDITY_USD
    scan_interval: int       = SCAN_INTERVAL
    slippage_tolerance: float = SLIPPAGE_TOLERANCE
    signal_weights: dict     = field(default_factory=lambda: dict(SIGNAL_WEIGHTS))


# Singleton instance for convenience
cfg = BotConfig()
