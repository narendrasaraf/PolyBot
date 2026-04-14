"""
polybot/data_layer.py
=====================
Data Layer — Fetches live market data from Polymarket APIs and persists
historical snapshots to CSV files for strategy analysis and backtesting.

APIs used:
  - Gamma API  (market discovery / metadata)  — no auth required
  - CLOB API   (prices, order book, spreads)  — API key required
"""

import os
import csv
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from polybot.config import (
    CLOB_HOST, GAMMA_HOST, CLOB_API_KEY,
    MIN_LIQUIDITY_USD, MIN_DAYS_TO_CLOSE, DATA_DIR
)
from polybot.logger import get_logger

log = get_logger("data_layer")


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Immutable snapshot of a single Polymarket market at a point in time."""
    condition_id: str
    question: str
    # Gamma fields
    volume: float       = 0.0
    liquidity: float    = 0.0
    days_to_close: int  = 30
    end_date: str       = ""
    category: str       = ""
    # CLOB pricing fields
    best_yes_ask: float = 0.5    # lowest sell price for YES tokens
    best_yes_bid: float = 0.5    # highest buy price for YES tokens
    mid_price: float    = 0.5    # (ask+bid)/2
    spread: float       = 0.0
    # YES & NO token IDs (needed for order placement)
    yes_token_id: str   = ""
    no_token_id: str    = ""
    # Timestamps
    fetched_at: str     = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_liquid(self) -> bool:
        return self.liquidity >= MIN_LIQUIDITY_USD and self.volume >= MIN_LIQUIDITY_USD

    @property
    def has_enough_time(self) -> bool:
        return self.days_to_close >= MIN_DAYS_TO_CLOSE

    @property
    def is_tradeable(self) -> bool:
        return self.is_liquid and self.has_enough_time and self.spread < 0.15


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent":    "PolyBot/2.0",
    "Authorization": f"Bearer {CLOB_API_KEY}",
})

def _get(url: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
    """Safe GET with retry (up to 3 attempts, exponential back-off)."""
    for attempt in range(3):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 5 * (2 ** attempt)
                log.warning(f"Rate-limited by {url} — sleeping {wait}s")
                time.sleep(wait)
            else:
                log.debug(f"HTTP {r.status_code} from {url}: {r.text[:120]}")
                return None
        except requests.RequestException as exc:
            log.warning(f"Request error ({url}): {exc}")
            time.sleep(2 ** attempt)
    return None


# ── Gamma API — market discovery ──────────────────────────────────────────────

def fetch_active_markets(limit: int = 50, offset: int = 0) -> list[dict]:
    """
    Pull active, open markets from the Gamma API.
    Returns raw market dicts (not yet enriched with CLOB prices).
    """
    data = _get(
        f"{GAMMA_HOST}/markets",
        params={
            "active":  "true",
            "closed":  "false",
            "limit":   limit,
            "offset":  offset,
            "order":   "volume",
            "ascending": "false",
        }
    )
    if data is None:
        return []
    # Gamma returns either a plain list or {"markets": [...]}
    if isinstance(data, list):
        return data
    return data.get("markets", data.get("data", []))


def fetch_market_by_id(condition_id: str) -> Optional[dict]:
    """Fetch a single market by its conditionId from the Gamma API."""
    data = _get(f"{GAMMA_HOST}/markets/{condition_id}")
    return data


# ── CLOB API — pricing ────────────────────────────────────────────────────────

def fetch_clob_price(token_id: str) -> Optional[dict]:
    """
    Fetch best bid/ask/mid for a token from the CLOB.
    Returns dict like: {"bid": 0.45, "ask": 0.55, "mid": 0.50}
    """
    mid_data = _get(f"{CLOB_HOST}/midpoint", params={"token_id": token_id})
    spread_data = _get(f"{CLOB_HOST}/spread", params={"token_id": token_id})
    book_data = _get(f"{CLOB_HOST}/book", params={"token_id": token_id})

    mid = 0.5
    bid = 0.5
    ask = 0.5
    spread = 0.0

    if mid_data:
        mid = float(mid_data.get("mid", 0.5))
    if spread_data:
        spread = float(spread_data.get("spread", 0.0))
        bid = round(mid - spread / 2, 4)
        ask = round(mid + spread / 2, 4)
    elif book_data:
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])
        if bids:
            bid = float(bids[0].get("price", mid))
        if asks:
            ask = float(asks[0].get("price", mid))
        spread = round(ask - bid, 4)
        mid = round((bid + ask) / 2, 4)

    return {"bid": bid, "ask": ask, "mid": mid, "spread": spread}


def fetch_order_book(token_id: str) -> dict:
    """Returns full order book for a token (used for depth analysis)."""
    data = _get(f"{CLOB_HOST}/book", params={"token_id": token_id})
    return data or {"bids": [], "asks": []}


# ── Enriched market snapshot builder ─────────────────────────────────────────

def build_snapshot(raw: dict) -> Optional[MarketSnapshot]:
    """
    Convert a raw Gamma market dict into a fully enriched MarketSnapshot
    by pulling live CLOB pricing.
    """
    condition_id = raw.get("conditionId") or raw.get("condition_id", "")
    question     = raw.get("question", "")

    if not condition_id or not question:
        return None

    # Extract Gamma metadata
    volume      = float(raw.get("volumeNum", raw.get("volume", 0)) or 0)
    liquidity   = float(raw.get("liquidityNum", raw.get("liquidity", volume)) or volume)
    end_date    = raw.get("endDate", raw.get("end_date", ""))
    category    = raw.get("category", raw.get("type", ""))

    # Days to resolution
    days_to_close = 30
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            days_to_close = max(0, (end_dt.replace(tzinfo=None) - datetime.utcnow()).days)
        except Exception:
            pass

    # Token IDs for YES and NO
    tokens       = raw.get("tokens", raw.get("clobTokenIds", []))
    yes_token_id = ""
    no_token_id  = ""
    if isinstance(tokens, list) and len(tokens) >= 2:
        yes_token_id = tokens[0] if isinstance(tokens[0], str) else tokens[0].get("token_id", "")
        no_token_id  = tokens[1] if isinstance(tokens[1], str) else tokens[1].get("token_id", "")
    elif isinstance(tokens, list) and len(tokens) == 1:
        yes_token_id = tokens[0] if isinstance(tokens[0], str) else tokens[0].get("token_id", "")

    # Fallback token IDs from outcomes []
    if not yes_token_id:
        outcomes = raw.get("outcomes", [])
        if outcomes:
            yes_token_id = condition_id  # use condition_id as proxy

    # Fetch live CLOB price
    token_for_price = yes_token_id or condition_id
    price_data = fetch_clob_price(token_for_price) if token_for_price else None

    mid_price  = float(raw.get("bestAsk", 0.5))   # fallback from Gamma
    best_bid   = mid_price
    best_ask   = mid_price
    spread     = 0.0

    if price_data:
        mid_price = price_data["mid"]
        best_bid  = price_data["bid"]
        best_ask  = price_data["ask"]
        spread    = price_data["spread"]

    return MarketSnapshot(
        condition_id = condition_id,
        question     = question,
        volume       = volume,
        liquidity    = liquidity,
        days_to_close = days_to_close,
        end_date     = end_date,
        category     = category,
        best_yes_ask = best_ask,
        best_yes_bid = best_bid,
        mid_price    = mid_price,
        spread       = spread,
        yes_token_id = yes_token_id,
        no_token_id  = no_token_id,
    )


# ── Historical data persistence ───────────────────────────────────────────────

class HistoricalStore:
    """
    Appends market snapshots to per-market CSV files.
    Directory structure: data/<condition_id>.csv
    Each row = one snapshot timestamp.
    """

    COLUMNS = [
        "fetched_at", "mid_price", "best_yes_bid", "best_yes_ask",
        "spread", "volume", "liquidity", "days_to_close",
    ]

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(self, condition_id: str) -> Path:
        return self.data_dir / f"{condition_id[:20]}.csv"

    def append(self, snap: MarketSnapshot) -> None:
        """Append one snapshot row to the market's CSV."""
        path = self._csv_path(snap.condition_id)
        write_header = not path.exists()
        row = {
            "fetched_at":   snap.fetched_at,
            "mid_price":    snap.mid_price,
            "best_yes_bid": snap.best_yes_bid,
            "best_yes_ask": snap.best_yes_ask,
            "spread":       snap.spread,
            "volume":       snap.volume,
            "liquidity":    snap.liquidity,
            "days_to_close": snap.days_to_close,
        }
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load(self, condition_id: str) -> pd.DataFrame:
        """Load historical data for a market as a DataFrame."""
        path = self._csv_path(condition_id)
        if not path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.read_csv(path, parse_dates=["fetched_at"])
        df.sort_values("fetched_at", inplace=True)
        return df

    def list_markets(self) -> list[str]:
        """Return all tracked condition IDs."""
        return [p.stem for p in self.data_dir.glob("*.csv")]

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all market CSVs as {condition_id: DataFrame}."""
        return {mid: self.load(mid) for mid in self.list_markets()}


# Module-level singleton
store = HistoricalStore()
