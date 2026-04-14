"""
polybot/ml/features.py
======================
Feature Engineering Pipeline for Polymarket ML Model

Transforms raw historical price/volume/liquidity data + external signals
into a clean, normalised feature matrix ready for all three model families.

Feature Groups:
  1. Price features       — levels, returns, momentum
  2. Volume features      — log-volume, normalised volume delta
  3. Liquidity features   — spread, depth, liquidity ratio
  4. Time features        — days-to-close, log-time, urgency flag
  5. Technical indicators — RSI, Bollinger Band position, MACD
  6. Sentiment features   — news/Reddit normalised score
  7. Market efficiency    — price extremes, vol-adjusted calibration

Output: pandas DataFrame (one row = one training sample)
Target: `yes_prob_outcome` — resolved probability (0 or 1 for closed markets,
         mid_price at time T+N for open markets in labelling mode)
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Price
    "price",
    "log_price",
    "price_ret_1",
    "price_ret_5",
    "price_ret_10",
    # Momentum
    "momentum_5",
    "momentum_10",
    "momentum_20",
    # Volatility
    "rolling_std_5",
    "rolling_std_10",
    # Bollinger Band position
    "bb_position",       # (price - lower) / (upper - lower), clamped 0-1
    "bb_width",          # (upper - lower) / mean
    # RSI
    "rsi_14",
    # MACD
    "macd",
    "macd_signal",
    "macd_hist",
    # Volume
    "log_volume",
    "vol_ret_1",
    "vol_acceleration",  # recent_vol_avg / prior_vol_avg
    # Liquidity
    "spread",
    "log_liquidity",
    "liquidity_ratio",   # liquidity / volume
    # Time
    "days_to_close",
    "log_days_to_close",
    "time_urgency",      # exponential decay: e^(-days/30)
    "near_resolution",   # 1 if days_to_close < 5
    # Market efficiency proxies
    "price_extreme_low",   # 1 if price < 0.12
    "price_extreme_high",  # 1 if price > 0.88
    "distance_from_half",  # |price - 0.5|
    "vol_efficiency",      # log10(volume+1) / 7 — how well-priced the market likely is
    # Sentiment (optional — filled with 0 if unavailable)
    "news_sentiment",
    "reddit_sentiment",
]


# ── Core feature builder ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    sentiment_score: float = 0.5,
    reddit_score: float    = 0.5,
    row_idx: int           = -1,       # which row to extract features for (default: last)
) -> Optional[np.ndarray]:
    """
    Build a 1-D feature vector from a historical price DataFrame.

    Args:
        df:               Historical data with columns:
                          [mid_price, volume, liquidity, spread, days_to_close]
        sentiment_score:  News sentiment probability [0,1] (0.5 = neutral)
        reddit_score:     Reddit sentiment probability [0,1]
        row_idx:          Row index to extract features for (-1 = last row)

    Returns:
        numpy array of shape (n_features,)  or  None if insufficient data.
    """
    if df is None or len(df) < 5:
        return None

    df = df.copy().reset_index(drop=True)

    # Ensure required columns exist with sensible defaults
    if "mid_price" not in df.columns:
        return None
    for col, default in [("volume", 1000), ("liquidity", 1000), ("spread", 0.02), ("days_to_close", 30)]:
        if col not in df.columns:
            df[col] = default

    prices    = df["mid_price"].astype(float).clip(0.001, 0.999)
    volumes   = df["volume"].astype(float).clip(lower=1)
    liquidit  = df["liquidity"].astype(float).clip(lower=1)
    spreads   = df["spread"].astype(float).clip(lower=0)
    days      = df["days_to_close"].astype(float).clip(lower=0)

    row = row_idx if row_idx >= 0 else len(df) - 1
    p   = float(prices.iloc[row])
    v   = float(volumes.iloc[row])
    liq = float(liquidit.iloc[row])
    sp  = float(spreads.iloc[row])
    d   = float(days.iloc[row])

    # ── 1. Price features ─────────────────────────────────────────────────────
    price          = p
    log_price      = math.log(p / (1 - p + 1e-9))   # logit transform
    price_ret_1    = _pct_ret(prices, row, 1)
    price_ret_5    = _pct_ret(prices, row, 5)
    price_ret_10   = _pct_ret(prices, row, 10)

    # ── 2. Momentum ───────────────────────────────────────────────────────────
    momentum_5    = _momentum(prices, row, 5)
    momentum_10   = _momentum(prices, row, 10)
    momentum_20   = _momentum(prices, row, 20)

    # ── 3. Volatility / Bollinger Bands ───────────────────────────────────────
    roll5  = prices.iloc[max(0, row-4) : row+1]
    roll10 = prices.iloc[max(0, row-9) : row+1]
    roll20 = prices.iloc[max(0, row-19): row+1]

    rolling_std_5  = float(roll5.std())  if len(roll5)  > 1 else 0.0
    rolling_std_10 = float(roll10.std()) if len(roll10) > 1 else 0.0

    mean20 = float(roll20.mean()) if len(roll20) > 1 else p
    std20  = float(roll20.std())  if len(roll20) > 1 else 0.01
    upper  = mean20 + 2 * std20
    lower  = mean20 - 2 * std20
    band_range  = max(upper - lower, 1e-6)
    bb_position = float(np.clip((p - lower) / band_range, 0, 1))
    bb_width    = band_range / max(mean20, 1e-6)

    # ── 4. RSI(14) ────────────────────────────────────────────────────────────
    rsi_14 = _rsi(prices, row, period=14)

    # ── 5. MACD (12, 26, 9) ───────────────────────────────────────────────────
    macd, macd_signal, macd_hist = _macd(prices, row)

    # ── 6. Volume features ────────────────────────────────────────────────────
    log_volume     = math.log10(v + 1)
    vol_ret_1      = _pct_ret(volumes, row, 1)
    recent_vol_avg = float(volumes.iloc[max(0, row-4) : row+1].mean())
    prior_vol_avg  = float(volumes.iloc[max(0, row-9) : max(0, row-4)].mean()) if row >= 9 else recent_vol_avg
    vol_acceleration = recent_vol_avg / max(prior_vol_avg, 1)

    # ── 7. Liquidity features ─────────────────────────────────────────────────
    spread           = sp
    log_liquidity    = math.log10(liq + 1)
    liquidity_ratio  = liq / max(v, 1)

    # ── 8. Time features ──────────────────────────────────────────────────────
    days_to_close    = d
    log_days_to_close = math.log(d + 1)
    time_urgency     = math.exp(-d / 30)      # → 1.0 when near expiry
    near_resolution  = 1.0 if d < 5 else 0.0

    # ── 9. Market efficiency proxies ─────────────────────────────────────────
    price_extreme_low  = 1.0 if p < 0.12 else 0.0
    price_extreme_high = 1.0 if p > 0.88 else 0.0
    distance_from_half = abs(p - 0.5)
    vol_efficiency     = math.log10(v + 1) / 7.0        # normalised 0-1 range

    # ── 10. Sentiment ─────────────────────────────────────────────────────────
    news_sentiment   = float(np.clip(sentiment_score, 0, 1))
    reddit_sentiment = float(np.clip(reddit_score, 0, 1))

    feat = np.array([
        price, log_price, price_ret_1, price_ret_5, price_ret_10,
        momentum_5, momentum_10, momentum_20,
        rolling_std_5, rolling_std_10,
        bb_position, bb_width,
        rsi_14,
        macd, macd_signal, macd_hist,
        log_volume, vol_ret_1, vol_acceleration,
        spread, log_liquidity, liquidity_ratio,
        days_to_close, log_days_to_close, time_urgency, near_resolution,
        price_extreme_low, price_extreme_high, distance_from_half, vol_efficiency,
        news_sentiment, reddit_sentiment,
    ], dtype=np.float32)

    # Replace NaN/Inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
    return feat


def build_feature_matrix(
    df: pd.DataFrame,
    n_forward: int         = 5,        # label: price N steps ahead
    sentiment_col: str     = None,
    reddit_col: str        = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a full training dataset from a single market's historical DataFrame.

    Labels are the actual YES probability N steps ahead (future mid_price).
    For closed markets with known outcomes, set the label to 0.0 or 1.0.

    Args:
        df:           Historical price data
        n_forward:    Steps ahead to label (proxy for future price)
        sentiment_col: Column name for news sentiment if available
        reddit_col:   Column name for reddit sentiment if available

    Returns:
        X: Feature matrix  (n_samples, n_features)
        y: Label vector    (n_samples,)  — future probability [0,1]
    """
    if df is None or len(df) < 20:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0)

    X_rows, y_rows = [], []
    prices = df["mid_price"].astype(float).clip(0.001, 0.999)

    for i in range(10, len(df) - n_forward):
        # Sentiment values from columns if present, else neutral
        sent  = float(df[sentiment_col].iloc[i])   if sentiment_col  and sentiment_col  in df.columns else 0.5
        redit = float(df[reddit_col].iloc[i])       if reddit_col     and reddit_col     in df.columns else 0.5

        feat = build_features(df.iloc[:i+1], sentiment_score=sent, reddit_score=redit, row_idx=-1)
        if feat is None:
            continue

        # Label: future mid_price as proxy for YES probability
        label = float(prices.iloc[i + n_forward])
        X_rows.append(feat)
        y_rows.append(label)

    if not X_rows:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def build_multi_market_dataset(
    market_dfs: dict[str, pd.DataFrame],
    n_forward: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Combine data from multiple markets into one training dataset.

    Returns:
        X:    (total_samples, n_features)
        y:    (total_samples,)
        cids: list of condition_ids (one per sample, for grouping)
    """
    X_all, y_all, cids = [], [], []
    for cid, df in market_dfs.items():
        if df.empty or len(df) < 20:
            continue
        X, y = build_feature_matrix(df, n_forward=n_forward)
        if len(X) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
        cids.extend([cid] * len(X))

    if not X_all:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0), []

    return np.vstack(X_all), np.concatenate(y_all), cids


# ── Technical indicator helpers ───────────────────────────────────────────────

def _pct_ret(series: pd.Series, row: int, lag: int) -> float:
    """Percentage return relative to `lag` periods back."""
    if row < lag:
        return 0.0
    prev = float(series.iloc[row - lag])
    curr = float(series.iloc[row])
    return (curr - prev) / max(abs(prev), 1e-9)


def _momentum(series: pd.Series, row: int, window: int) -> float:
    """Simple momentum: curr - rolling_mean over previous window."""
    if row < window:
        return 0.0
    past_mean = float(series.iloc[max(0, row - window) : row].mean())
    return float(series.iloc[row]) - past_mean


def _rsi(series: pd.Series, row: int, period: int = 14) -> float:
    """Wilder RSI [0, 1] — normalised to 0-1 range."""
    if row < period + 1:
        return 0.5
    sub = series.iloc[max(0, row - period - 1) : row + 1].values.astype(float)
    deltas = np.diff(sub)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss < 1e-9:
        return 1.0
    rs  = avg_gain / avg_loss
    rsi = 1.0 - 1.0 / (1.0 + rs)
    return float(np.clip(rsi, 0, 1))


def _macd(series: pd.Series, row: int,
          fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram) — all normalised to price scale."""
    if row < slow + signal:
        return 0.0, 0.0, 0.0
    sub = series.iloc[max(0, row - slow - signal) : row + 1]
    ema_fast   = sub.ewm(span=fast,   adjust=False).mean()
    ema_slow   = sub.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    hist       = macd_line - sig_line
    return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(hist.iloc[-1])


# ── Feature importance display ────────────────────────────────────────────────

def feature_importance_table(importances: np.ndarray) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    df = pd.DataFrame({
        "feature":    FEATURE_COLS[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df
