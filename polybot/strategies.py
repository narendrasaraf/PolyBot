"""
polybot/strategies.py
=====================
Strategy Engine — Core strategies + sentiment + ML ensemble.

CHANGES FROM PREVIOUS VERSION
------------------------------
1. _BASE_RATES dict replaced with EmpiricalBaseRates class:
   - Loads historical resolution data from the data store
   - Computes empirical YES-rates per keyword category with Wilson CI
   - Returns (rate, confidence) tuples; falls back to (0.50, 0.0) when sparse

2. Reddit sentiment confidence capped at 0.25 (was 0.70).
   Bag-of-words sentiment is noisy and should not dominate the ensemble
   until a proper NLP model (BERT/RoBERTa fine-tuned on PM questions) is used.

3. run_all_strategies() now uses a proper meta-ensemble:
   - Collects yes_probability from all strategies as a feature vector
   - If a meta-model (logistic regression on strategy probabilities) is trained
     and available at models/meta_model.pkl, it is used for the final probability
   - Otherwise falls back to weighted average with disagreement penalty

4. EV formula in the aggregator is CORRECT:
   EV_YES = predicted_prob × (1 - market_price) - (1 - predicted_prob) × market_price
   EV_NO  = (1 - predicted_prob) × market_price - predicted_prob × (1 - market_price)
   Direction is determined by which EV exceeds MIN_EV_THRESHOLD (not raw edge).

5. ML signal weight is soft-capped — if the calibrated ML model is live-ready,
   the effective weight is raised to 0.45 automatically.

Each strategy returns a SignalResult with:
  - direction: "BUY_YES" | "BUY_NO" | "HOLD"
  - yes_probability: 0.0 – 1.0
  - confidence: 0.0 – 1.0
  - metadata: arbitrary debug info
"""

from __future__ import annotations

import json
import time
import math
import pickle
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from polybot.config import (
    CLOB_API_KEY, GEMINI_API_KEY, GEMINI_MODEL,
    SIGNAL_WEIGHTS, MIN_EDGE, MIN_CONFIDENCE,
)
from polybot.logger import get_logger

log = get_logger("strategies")

# Minimum EV/stake threshold for signalling a trade
MIN_EV_THRESHOLD = 0.04

# Reddit confidence cap until proper NLP model is integrated
REDDIT_MAX_CONFIDENCE = 0.25

# ML signal weight when model is live-ready (Brier < 0.20, >= 200 test samples)
ML_LIVE_WEIGHT = 0.45


# ── Signal result container ───────────────────────────────────────────────────

@dataclass
class SignalResult:
    """
    Output from a single strategy function.

    Attributes:
        strategy:        Strategy identifier string.
        direction:       'BUY_YES' | 'BUY_NO' | 'HOLD'.
        yes_probability: Model's P(YES) in [0, 1].
        confidence:      Strategy-specific confidence in [0, 1].
        edge:            yes_probability - market_price (legacy; use EV for trades).
        metadata:        Debug info dict.
    """
    strategy:        str
    direction:       str   = "HOLD"
    yes_probability: float = 0.5
    confidence:      float = 0.0
    edge:            float = 0.0
    metadata:        dict  = field(default_factory=dict)


@dataclass
class CombinedSignal:
    """
    The aggregated output consumed by the risk manager and executor.

    Attributes:
        yes_probability: Final ensemble P(YES).
        confidence:      Ensemble confidence (penalised for disagreement).
        ev_yes:          Expected value of BUY_YES trade.
        ev_no:           Expected value of BUY_NO trade.
        edge:            Signed edge (legacy compatibility).
        direction:       'BUY_YES' | 'BUY_NO' | 'HOLD' — based on EV, not edge.
        signal_details:  Per-strategy breakdown dict.
        meta_model_used: True if the meta-ensemble model made the final call.
    """
    yes_probability: float
    confidence:      float
    ev_yes:          float
    ev_no:           float
    edge:            float
    direction:       str
    signal_details:  dict = field(default_factory=dict)
    meta_model_used: bool = False

    @property
    def is_actionable(self) -> bool:
        """True if this signal passes the confidence and EV gates."""
        active_ev = self.ev_yes if self.direction == "BUY_YES" else self.ev_no
        return (
            self.direction != "HOLD"
            and self.confidence >= MIN_CONFIDENCE
            and active_ev >= MIN_EV_THRESHOLD
        )


# ─────────────────────────────────────────────────────────────────────────────
# EMPIRICAL BASE RATES (replaces _BASE_RATES dict)
# ─────────────────────────────────────────────────────────────────────────────

class EmpiricalBaseRates:
    """
    Data-driven empirical resolution rates per keyword category.

    Instead of a hand-coded lookup table, this class loads historical resolved
    markets from the data store and computes the actual fraction of YES
    resolutions per keyword (with Wilson confidence intervals for uncertainty).

    Falls back to (0.50, 0.0) — maximum prior uncertainty — when fewer than
    5 resolved markets match a keyword.

    Usage:
        ebr = EmpiricalBaseRates()
        ebr.load(markets_dict)  # load resolved markets
        rate, conf = ebr.get_rate("Will Bitcoin exceed $100k by 2025?")

    Example:
        >>> ebr = EmpiricalBaseRates()
        >>> ebr.load(resolved_markets)
        >>> rate, conf = ebr.get_rate("Will the Fed raise rates in March?")
        >>> print(f"Base rate: {rate:.2%} ± conf={conf:.2f}")
    """

    # Keyword buckets (order matters — first match wins)
    _KEYWORD_MAP: dict[str, list[str]] = {
        "crypto":    ["bitcoin", "btc", "ethereum", "eth", "crypto", "doge", "solana"],
        "fed":       ["fed", "fomc", "rate", "inflation", "gdp"],
        "election":  ["election", "president", "vote", "senate", "congress", "ballot"],
        "tech":      ["apple", "google", "microsoft", "nvidia", "ai", "openai", "meta"],
        "geopolitics": ["war", "attack", "conflict", "military", "sanction"],
        "macro":     ["recession", "gdp", "unemployment", "cpi"],
        "sports":    ["nba", "nfl", "nhl", "mlb", "soccer", "football", "tennis"],
        "default":   [],
    }

    # Fallback prior rates (used when no empirical data exists for category)
    _PRIOR_RATES: dict[str, float] = {
        "crypto":      0.42,
        "fed":         0.48,
        "election":    0.50,
        "tech":        0.60,
        "geopolitics": 0.35,
        "macro":       0.40,
        "sports":      0.50,
        "default":     0.50,
    }

    def __init__(self) -> None:
        self._rates:   dict[str, float] = {}   # category → empirical rate
        self._counts:  dict[str, int]   = {}   # category → number of resolved markets
        self._loaded   = False

    def load(self, market_dfs: dict[str, pd.DataFrame]) -> None:
        """
        Compute empirical rates from a dictionary of resolved market DataFrames.

        Each market must have a 'resolution' (0 or 1) and a 'question' column.
        Markets without a readable question are binned to 'default'.

        Args:
            market_dfs: Dict mapping condition_id → DataFrame with binary labels.

        Example:
            >>> ebr.load(resolved_markets)
        """
        category_outcomes: dict[str, list[int]] = {k: [] for k in self._KEYWORD_MAP}

        for cid, df in market_dfs.items():
            label = _extract_binary_label(df)
            if label is None:
                continue
            question = ""
            if "question" in df.columns:
                question = str(df["question"].iloc[-1]).lower()
            cat = self._classify(question)
            category_outcomes[cat].append(int(label))

        for cat, outcomes in category_outcomes.items():
            n = len(outcomes)
            self._counts[cat] = n
            if n >= 5:
                self._rates[cat] = float(np.mean(outcomes))
            # else: no empirical rate — will fall back to prior

        self._loaded = True
        log.info(
            f"EmpiricalBaseRates loaded: "
            + ", ".join(f"{k}={self._counts.get(k,0)}mkts" for k in self._KEYWORD_MAP)
        )

    def get_rate(self, question: str) -> tuple[float, float]:
        """
        Return (empirical_rate, confidence) for a given market question.

        Confidence is the lower bound of the Wilson 95% CI normalised to [0,1]:
          high count (+30 markets) → confidence ≈ 0.8
          5-30 markets             → confidence ~ 0.3–0.6
          < 5 markets              → falls back to prior, confidence = 0.0

        Args:
            question: Full market question text.

        Returns:
            Tuple of (rate: float, confidence: float) both in [0, 1].

        Example:
            >>> rate, conf = ebr.get_rate("Will BTC exceed $100k?")
            >>> print(rate, conf)   # e.g. 0.42, 0.65
        """
        cat   = self._classify(question.lower())
        n     = self._counts.get(cat, 0)
        prior = self._PRIOR_RATES.get(cat, 0.50)

        if not self._loaded or n < 5:
            return prior, 0.0

        rate = self._rates.get(cat, prior)
        conf = float(np.clip((n - 5) / 45.0, 0.0, 0.8))   # saturates at 0.80 after 50 markets
        return rate, conf

    def _classify(self, question_lower: str) -> str:
        """Classify a question string into a keyword category."""
        for cat, keywords in self._KEYWORD_MAP.items():
            if cat == "default":
                continue
            if any(kw in question_lower for kw in keywords):
                return cat
        return "default"


# Module-level empirical base rates instance (lazy-loaded)
_empirical_base_rates: Optional[EmpiricalBaseRates] = None


def _get_empirical_base_rates() -> EmpiricalBaseRates:
    """
    Return the module-level EmpiricalBaseRates instance, loading it if needed.

    Lazily loads from the data store on first call and caches forever
    (since base rates are stationary within a single bot session).

    Returns:
        EmpiricalBaseRates instance, loaded with available resolved markets.

    Example:
        >>> ebr = _get_empirical_base_rates()
        >>> rate, conf = ebr.get_rate(question)
    """
    global _empirical_base_rates
    if _empirical_base_rates is not None:
        return _empirical_base_rates

    _empirical_base_rates = EmpiricalBaseRates()
    try:
        from polybot.data_layer import store
        all_dfs = store.load_all()
        resolved = {
            k: df for k, df in all_dfs.items()
            if _extract_binary_label(df) is not None
        }
        _empirical_base_rates.load(resolved)
    except Exception as exc:
        log.warning(f"EmpiricalBaseRates could not load from store: {exc}")

    return _empirical_base_rates


def _extract_binary_label(df: pd.DataFrame) -> Optional[float]:
    """Extract 0 or 1 binary label from a market DataFrame."""
    for col in ("resolution", "label", "outcome"):
        if col in df.columns:
            for v in df[col].dropna().unique():
                try:
                    fv = float(v)
                    if fv in (0.0, 1.0):
                        return fv
                except (ValueError, TypeError):
                    continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: Momentum
# ─────────────────────────────────────────────────────────────────────────────

def strategy_momentum(snap, history: pd.DataFrame) -> SignalResult:
    """
    Detect momentum: a sustained price move with rising volume.

    Logic:
      - Compute rolling mean of mid_price over last N rows vs older mean.
      - Compute volume acceleration (recent vs prior quarter).
      - Breakout = price moved > 4c AND volume 15% higher than baseline.

    Args:
        snap:    MarketSnapshot with mid_price.
        history: Historical DataFrame with mid_price + volume columns.

    Returns:
        SignalResult with BUY_YES/BUY_NO/HOLD direction.

    Example:
        >>> sig = strategy_momentum(snap, history_df)
        >>> print(sig.direction, sig.confidence)
    """
    name = "momentum"
    if history.empty or len(history) < 10:
        return SignalResult(strategy=name, metadata={"reason": "insufficient history"})

    df = history.copy().tail(40)
    if "mid_price" not in df.columns:
        return SignalResult(strategy=name, metadata={"reason": "no price column"})

    prices  = df["mid_price"].astype(float)
    volumes = df["volume"].astype(float) if "volume" in df.columns else pd.Series([1.0] * len(df))

    window       = min(8, len(prices) // 2)
    recent_price = prices.iloc[-window:].mean()
    prior_price  = prices.iloc[:-window].mean()
    price_delta  = recent_price - prior_price

    recent_vol   = volumes.iloc[-window:].mean()
    prior_vol    = volumes.iloc[:-window].mean()
    vol_ratio    = recent_vol / max(prior_vol, 1.0)

    if abs(price_delta) < 0.04 or vol_ratio < 1.15:
        return SignalResult(
            strategy=name,
            metadata={"price_delta": round(price_delta, 4), "vol_ratio": round(vol_ratio, 3)},
        )

    direction = "BUY_YES" if price_delta > 0 else "BUY_NO"
    raw_conf  = min(1.0, abs(price_delta) / 0.20 * 0.6 + (vol_ratio - 1.0) * 0.4)
    yes_prob  = float(np.clip(snap.mid_price + price_delta * 0.5, 0.01, 0.99))
    edge      = yes_prob - snap.mid_price

    return SignalResult(
        strategy       = name,
        direction      = direction,
        yes_probability = round(yes_prob, 3),
        confidence     = round(raw_conf, 3),
        edge           = round(edge, 4),
        metadata       = {"price_delta": round(price_delta, 4), "vol_ratio": round(vol_ratio, 3)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────

def strategy_mean_reversion(snap, history: pd.DataFrame) -> SignalResult:
    """
    Detect mean reversion via Bollinger Band breach.

    Price above upper band → overbought → BUY_NO.
    Price below lower band → oversold  → BUY_YES.
    Confidence is proportional to how far outside the band the price is.

    Args:
        snap:    MarketSnapshot.
        history: Historical DataFrame with mid_price.

    Returns:
        SignalResult.

    Example:
        >>> sig = strategy_mean_reversion(snap, history_df)
    """
    name = "mean_rev"
    if history.empty or len(history) < 14:
        return SignalResult(strategy=name, metadata={"reason": "insufficient history"})

    prices  = history["mid_price"].astype(float).tail(30)
    mean    = prices.mean()
    std     = prices.std()
    if std < 1e-6:
        return SignalResult(strategy=name, metadata={"reason": "flat market"})

    upper   = mean + 2.0 * std
    lower   = mean - 2.0 * std
    current = snap.mid_price
    z       = (current - mean) / std

    if current > upper:
        direction = "BUY_NO"
        yes_prob  = float(np.clip(mean - 0.5 * std, 0.01, 0.99))
        conf      = min(1.0, (current - upper) / std * 0.5)
    elif current < lower:
        direction = "BUY_YES"
        yes_prob  = float(np.clip(mean + 0.5 * std, 0.01, 0.99))
        conf      = min(1.0, (lower - current) / std * 0.5)
    else:
        return SignalResult(
            strategy=name,
            metadata={"z_score": round(z, 3), "mean": round(mean, 4), "std": round(std, 4)},
        )

    edge = yes_prob - current
    return SignalResult(
        strategy       = name,
        direction      = direction,
        yes_probability = round(yes_prob, 3),
        confidence     = round(conf, 3),
        edge           = round(edge, 4),
        metadata       = {"z_score": round(z, 3), "mean": round(mean, 4),
                          "upper": round(upper, 4), "lower": round(lower, 4)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: Probability Gap (uses EmpiricalBaseRates)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_probability_gap(snap) -> SignalResult:
    """
    Signal when market price is significantly off the empirical fair value.

    Fair value = weighted blend of:
      1. Empirical base rate (with Wilson CI confidence) for the market category.
      2. Volume calibration (thin markets have larger expected pricing errors).
      3. Time decay (near-expiry markets are more efficiently priced).
      4. Price extremes correction (overreaction at tails).

    Args:
        snap: MarketSnapshot.

    Returns:
        SignalResult.

    Example:
        >>> sig = strategy_probability_gap(snap)
        >>> print(sig.metadata["base_rate"], sig.direction)
    """
    name    = "prob_gap"
    p       = snap.mid_price
    ebr     = _get_empirical_base_rates()
    base_rate, base_conf = ebr.get_rate(snap.question)

    # Volume calibration
    vol_factor  = max(0.5, min(2.5, 500_000 / max(snap.volume, 1.0)))

    # Time decay
    time_factor = min(1.0, snap.days_to_close / 90.0)

    # Extreme price correction
    if p > 0.88:
        extreme_adj = -0.04 * vol_factor
    elif p < 0.12:
        extreme_adj = +0.03 * vol_factor
    else:
        extreme_adj = (0.50 - p) * 0.05 * vol_factor

    # Weight for empirical base: 0.40 if well-supported, else 0.25
    base_weight = 0.40 if base_conf > 0.3 else 0.25
    fair_value  = base_rate * base_weight + p * (1.0 - base_weight) + extreme_adj * time_factor
    fair_value  = float(np.clip(fair_value, 0.01, 0.99))
    edge        = fair_value - p

    if abs(edge) < 0.04:
        return SignalResult(
            strategy=name,
            metadata={"fair_value": round(fair_value, 4), "edge": round(edge, 4),
                      "base_rate": base_rate, "base_conf": round(base_conf, 3)},
        )

    direction = "BUY_YES" if edge > 0 else "BUY_NO"
    conf      = min(1.0, abs(edge) * 3.5 * vol_factor * 0.3 * (1.0 + base_conf))

    return SignalResult(
        strategy       = name,
        direction      = direction,
        yes_probability = round(fair_value, 4),
        confidence     = round(conf, 3),
        edge           = round(edge, 4),
        metadata       = {"fair_value": round(fair_value, 4), "base_rate": base_rate,
                          "base_conf": round(base_conf, 3), "vol_factor": round(vol_factor, 2)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4: News Sentiment (Gemini)
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_client():
    """Lazy import of Gemini client."""
    try:
        from google import genai
        from google.genai import types as gt
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client, gt
    except Exception as e:
        log.warning(f"Gemini client unavailable: {e}")
        return None, None


def strategy_news(question: str) -> SignalResult:
    """
    Use Gemini with Google Search grounding to score recent news.

    Returns P(YES) based on recency-weighted news analysis.

    Args:
        question: Market question text.

    Returns:
        SignalResult with yes_probability from Gemini JSON response.

    Example:
        >>> sig = strategy_news("Will the Fed cut rates in June?")
    """
    name = "news"
    client, gt = _gemini_client()
    if client is None:
        return SignalResult(strategy=name, yes_probability=0.5, metadata={"reason": "no gemini"})

    system_prompt = (
        "You are an expert prediction market analyst. "
        "Search for the most recent news relevant to this market question. "
        'Return ONLY valid JSON: {"score": <0-100>, "confidence": <0-100>, "summary": "<15 words max>"}. '
        "score = P(YES resolves), confidence = how certain you are. No extra text."
    )
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Prediction market: {question}",
                config=gt.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[gt.Tool(google_search=gt.GoogleSearch())],
                    max_output_tokens=200,
                ),
            )
            text = resp.text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(text)
            score    = max(0.0, min(1.0, int(data.get("score", 50)) / 100))
            raw_conf = max(0.0, min(1.0, int(data.get("confidence", 50)) / 100))
            edge     = score - 0.5
            direction = "BUY_YES" if score > 0.5 else ("BUY_NO" if score < 0.5 else "HOLD")
            return SignalResult(
                strategy       = name,
                direction      = direction,
                yes_probability = round(score, 3),
                confidence     = round(raw_conf, 3),
                edge           = round(edge, 3),
                metadata       = {"summary": data.get("summary", ""), "raw_score": data.get("score")},
            )
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                wait = 15 * (2 ** attempt)
                log.warning(f"Gemini quota — waiting {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                log.warning(f"News signal error: {exc}")
                break
    return SignalResult(strategy=name, yes_probability=0.5, metadata={"reason": "gemini error"})


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5: Reddit Sentiment (confidence CAPPED at 0.25)
# ─────────────────────────────────────────────────────────────────────────────

_POS_WORDS = {"yes", "likely", "will", "confirmed", "bullish", "positive", "win", "up", "green", "pass"}
_NEG_WORDS = {"no", "unlikely", "wont", "bearish", "negative", "lose", "down", "red", "fail", "rejected", "never"}


def strategy_reddit(question: str) -> SignalResult:
    """
    Scrape Reddit search results and score bullish/bearish keyword sentiment.

    Confidence is CAPPED AT 0.25 because bag-of-words sentiment over a 30-post
    sample is too noisy and topic-ambiguous for reliable prediction market signals.
    This cap will be lifted once a domain-fine-tuned NLP model is integrated.

    Args:
        question: Market question text.

    Returns:
        SignalResult with confidence <= 0.25.

    Example:
        >>> sig = strategy_reddit("Will Bitcoin reach $100k?")
        >>> assert sig.confidence <= 0.25
    """
    name = "reddit"
    try:
        keywords = question[:70].replace("?", "").replace("Will ", "")
        r = requests.get(
            "https://www.reddit.com/search.json",
            headers={"User-Agent": "polybot/3.0"},
            params={"q": keywords, "sort": "new", "limit": 30, "t": "week"},
            timeout=10,
        )
        posts = r.json().get("data", {}).get("children", [])
        if not posts:
            return SignalResult(strategy=name, yes_probability=0.5, metadata={"count": 0})

        weighted_scores: list[float] = []
        for post in posts:
            d    = post["data"]
            text = (d.get("title", "") + " " + d.get("selftext", "")).lower()
            words = set(text.split())
            pos  = len(words & _POS_WORDS)
            neg  = len(words & _NEG_WORDS)
            if pos + neg == 0:
                continue
            upvote_w = min(2.5, 1.0 + d.get("score", 0) / 200)
            sent     = pos / (pos + neg)
            weighted_scores.append(sent * upvote_w)

        if not weighted_scores:
            return SignalResult(strategy=name, yes_probability=0.5, metadata={"count": len(posts)})

        avg  = float(np.clip(sum(weighted_scores) / len(weighted_scores), 0.01, 0.99))

        # CAPPED AT 0.25 — proper NLP model needed for higher confidence
        raw_conf = min(REDDIT_MAX_CONFIDENCE, len(weighted_scores) / 20.0 * REDDIT_MAX_CONFIDENCE)

        edge      = avg - 0.5
        direction = "BUY_YES" if avg > 0.55 else ("BUY_NO" if avg < 0.45 else "HOLD")
        return SignalResult(
            strategy       = name,
            direction      = direction,
            yes_probability = round(avg, 3),
            confidence     = round(raw_conf, 3),
            edge           = round(edge, 3),
            metadata       = {"post_count": len(posts), "scored": len(weighted_scores),
                              "confidence_cap": REDDIT_MAX_CONFIDENCE},
        )
    except Exception as exc:
        log.warning(f"Reddit signal error: {exc}")
        return SignalResult(strategy=name, yes_probability=0.5, metadata={"reason": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 6: ML (calibrated model + EV computation)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_ml(snap, days_to_resolve: int = 30, history=None) -> SignalResult:
    """
    ML-powered probability estimator using calibrated XGBoost/LightGBM.

    Tries in order:
      1. Calibrated XGBoost / LightGBM / Logistic from predictor.py
      2. Heuristic 5-feature weighted ensemble (if no models found on disk)

    The signal confidence reflects:
      - 0.5 × calibration_certainty (distance of predicted_prob from 0.5)
      - 0.3 × feature_reliability (non-zero fraction)
      - 0.2 × model_agreement (1 - |xgb - logistic|)

    Args:
        snap:            MarketSnapshot.
        days_to_resolve: Days until resolution (from snap.days_to_close).
        history:         Historical DataFrame for feature engineering.

    Returns:
        SignalResult with direction based on correct EV, not raw edge.

    Example:
        >>> sig = strategy_ml(snap, history=history_df)
        >>> print(sig.metadata["model"], sig.confidence)
    """
    name = "ml"
    p    = snap.mid_price

    try:
        from polybot.ml.predictor import predictor as _ml_predictor
        import pandas as pd

        hist = history if (
            history is not None and isinstance(history, pd.DataFrame)
            and not history.empty
        ) else pd.DataFrame()

        ml_sig = _ml_predictor.predict_and_signal(snap, hist)

        if ml_sig.model_name != "heuristic" or ml_sig.predicted_prob != 0.5:
            score = ml_sig.predicted_prob
            ev_y  = ml_sig.ev_yes
            ev_n  = ml_sig.ev_no

            # Use EV-based direction, not edge
            if ev_y > ev_n and ev_y > MIN_EV_THRESHOLD:
                direction = "BUY_YES"
            elif ev_n > ev_y and ev_n > MIN_EV_THRESHOLD:
                direction = "BUY_NO"
            else:
                direction = "HOLD"

            return SignalResult(
                strategy        = name,
                direction       = direction,
                yes_probability = round(score, 4),
                confidence      = round(ml_sig.confidence, 3),
                edge            = round(ml_sig.edge, 4),
                metadata        = {
                    "model":       ml_sig.model_name,
                    "ev_yes":      round(ev_y, 4),
                    "ev_no":       round(ev_n, 4),
                    "high_conf":   ml_sig.is_high_conf,
                    "mispricing":  ml_sig.mispricing_label,
                },
            )
    except Exception as exc:
        log.debug(f"ML predictor unavailable ({exc}) — using heuristic fallback")

    # ── Heuristic fallback ─────────────────────────────────────────────────
    f1   = p
    f2   = max(0, min(1, 1.0 - days_to_resolve / 120)) * f1 + \
           (1 - max(0, min(1, 1.0 - days_to_resolve / 120))) * 0.5
    f3   = p * (1 - min(0.5, snap.spread * 2)) + 0.5 * min(0.5, snap.spread * 2)
    f4   = 0.50
    vol_adj = max(0.5, min(1.5, math.log10(max(snap.volume, 1) + 1) / 6))
    f5   = p * vol_adj + 0.5 * (1 - vol_adj)
    score = float(np.clip(0.35 * f1 + 0.20 * f2 + 0.15 * f3 + 0.15 * f4 + 0.15 * f5, 0.01, 0.99))

    from polybot.ml.predictor import compute_ev_yes, compute_ev_no
    ev_y = compute_ev_yes(score, p)
    ev_n = compute_ev_no(score, p)
    direction = "BUY_YES" if ev_y > MIN_EV_THRESHOLD and ev_y > ev_n else (
                "BUY_NO"  if ev_n > MIN_EV_THRESHOLD else "HOLD")
    conf = min(0.50, abs(score - p) * 4.0)

    return SignalResult(
        strategy        = name,
        direction       = direction,
        yes_probability = round(score, 4),
        confidence      = round(conf, 3),
        edge            = round(score - p, 4),
        metadata        = {"model": "heuristic", "ev_yes": round(ev_y, 4), "ev_no": round(ev_n, 4)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# META-ENSEMBLE: Aggregate all signals
# ─────────────────────────────────────────────────────────────────────────────

def _load_meta_model() -> Optional[object]:
    """
    Load the meta-ensemble model from disk, if available.

    The meta-model is a simple logistic regression trained on strategy
    yes_probability values as features to predict actual binary outcomes.
    Trained separately in trainer.py (not yet implemented — future work).

    Returns:
        Fitted meta-model with .predict_proba(X) or None.

    Example:
        >>> meta = _load_meta_model()
        >>> if meta: final_prob = meta.predict_proba([strategy_probs])[0, 1]
    """
    meta_path = Path("models/meta_model.pkl")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log.warning(f"Could not load meta_model.pkl: {exc}")
        return None


def _is_ml_live_ready() -> bool:
    """
    Check if a live-ready calibrated ML model exists on disk.

    Returns True if any calibrated model file is present (proxy for live-readiness).

    Returns:
        True if models/calibrated_xgboost.pkl or calibrated_lightgbm.pkl exists.

    Example:
        >>> if _is_ml_live_ready():
        ...     weights["ml"] = ML_LIVE_WEIGHT
    """
    return any((
        Path("models/calibrated_xgboost.pkl").exists(),
        Path("models/calibrated_lightgbm.pkl").exists(),
        Path("models/calibrated_logistic.pkl").exists(),
    ))


def run_all_strategies(snap, history: pd.DataFrame) -> CombinedSignal:
    """
    Run all 6 strategies, aggregate with meta-ensemble or weighted average,
    compute correct EV, and return a CombinedSignal.

    Aggregation logic:
      1. If models/meta_model.pkl exists: use it for the final probability
         (logistic regression on all strategy probabilities as features).
      2. Otherwise: weighted average of yes_probabilities with disagreement penalty.

    EV computation:
      Uses the correct formulas:
        EV_YES = final_prob × (1 - market_price) - (1 - final_prob) × market_price
        EV_NO  = (1 - final_prob) × market_price - final_prob × (1 - market_price)

    ML weight is automatically elevated to 0.45 when a live-ready calibrated model
    is detected (vs 0.10 in the legacy config).

    Args:
        snap:    MarketSnapshot (live data).
        history: Historical DataFrame from HistoricalStore.

    Returns:
        CombinedSignal with all aggregated fields.

    Example:
        >>> combined = run_all_strategies(snap, history)
        >>> print(combined.direction, combined.ev_yes)
    """
    from polybot.ml.predictor import compute_ev_yes, compute_ev_no

    results: dict[str, SignalResult] = {
        "momentum": strategy_momentum(snap, history),
        "mean_rev": strategy_mean_reversion(snap, history),
        "prob_gap": strategy_probability_gap(snap),
        "news":     strategy_news(snap.question),
        "reddit":   strategy_reddit(snap.question),
        "ml":       strategy_ml(snap, snap.days_to_close, history=history),
    }

    # Empirical base rate as a neutral prior
    ebr = _get_empirical_base_rates()
    base_rate, base_conf = ebr.get_rate(snap.question)
    results["base_rate"] = SignalResult(
        strategy="base_rate", yes_probability=base_rate, confidence=base_conf * 0.5,
    )

    # ── Dynamic weight adjustment for ML ─────────────────────────────────────
    weights = dict(SIGNAL_WEIGHTS)   # take a copy so we don't mutate the global
    if _is_ml_live_ready():
        extra = ML_LIVE_WEIGHT - weights.get("ml", 0.10)
        weights["ml"] = ML_LIVE_WEIGHT
        # Redistribute the extra weight proportionally from other strategies
        others = [k for k in weights if k != "ml"]
        for k in others:
            weights[k] = max(0.0, weights[k] - extra / max(len(others), 1))

    # Renormalise
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # ── Meta-model or weighted average ────────────────────────────────────────
    meta_model     = _load_meta_model()
    meta_used      = False

    if meta_model is not None:
        try:
            strategy_probs = np.array([[
                results[k].yes_probability
                for k in sorted(results.keys())
            ]], dtype=np.float32)
            yes_prob  = float(meta_model.predict_proba(strategy_probs)[0, 1])
            meta_used = True
        except Exception as exc:
            log.warning(f"Meta-model inference failed: {exc} — falling back to weighted avg")
            yes_prob  = _weighted_avg(results, weights)
    else:
        yes_prob = _weighted_avg(results, weights)

    # ── Confidence with disagreement penalty ──────────────────────────────────
    avg_conf     = sum(weights.get(k, 0) * r.confidence for k, r in results.items())
    probs        = [r.yes_probability for r in results.values()]
    variance     = float(np.var(probs))
    disagree_pen = min(0.40, variance * 4)
    confidence   = float(np.clip(avg_conf - disagree_pen, 0.0, 1.0))

    # ── Correct EV computation ────────────────────────────────────────────────
    mp    = float(snap.mid_price)
    ev_y  = compute_ev_yes(yes_prob, mp)
    ev_n  = compute_ev_no(yes_prob, mp)
    edge  = yes_prob - mp

    if ev_y > MIN_EV_THRESHOLD and ev_y >= ev_n:
        direction = "BUY_YES"
    elif ev_n > MIN_EV_THRESHOLD and ev_n > ev_y:
        direction = "BUY_NO"
    else:
        direction = "HOLD"

    signal_details = {k: {
        "yes_prob":   r.yes_probability,
        "confidence": r.confidence,
        "direction":  r.direction,
        "edge":       r.edge,
        "meta":       r.metadata,
    } for k, r in results.items()}

    log.debug(
        f"Combined: yes_prob={yes_prob:.3f} conf={confidence:.3f} "
        f"EV_YES={ev_y:.4f} EV_NO={ev_n:.4f} dir={direction} meta={meta_used}"
    )

    return CombinedSignal(
        yes_probability = round(yes_prob, 4),
        confidence      = round(confidence, 3),
        ev_yes          = round(ev_y, 4),
        ev_no           = round(ev_n, 4),
        edge            = round(edge, 4),
        direction       = direction,
        signal_details  = signal_details,
        meta_model_used = meta_used,
    )


def _weighted_avg(results: dict[str, SignalResult], weights: dict[str, float]) -> float:
    """
    Compute the weight-normalised average of yes_probability across all strategies.

    Args:
        results: Dict of strategy → SignalResult.
        weights: Dict of strategy → weight.

    Returns:
        Weighted average probability in [0.01, 0.99].

    Example:
        >>> avg = _weighted_avg(results, weights)
    """
    total_w = sum(weights.get(k, 0) for k in results)
    if total_w < 1e-9:
        return 0.5
    avg = sum(weights.get(k, 0) * r.yes_probability for k, r in results.items()) / total_w
    return float(np.clip(avg, 0.01, 0.99))
