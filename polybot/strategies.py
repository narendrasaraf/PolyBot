"""
polybot/strategies.py
=====================
Strategy Engine — Three core strategies + sentiment + ML.

Each strategy function accepts a MarketSnapshot and optional historical
DataFrame, and returns a SignalResult with:
  - direction: "BUY_YES" | "BUY_NO" | "HOLD"
  - yes_probability: 0.0 – 1.0 (model's estimate of YES outcome)
  - confidence: 0.0 – 1.0
  - edge: model_prob - market_price (positive = BUY YES, negative = BUY NO)
  - metadata: arbitrary debug info
"""

from __future__ import annotations

import json
import time
import math
import requests
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from polybot.config import (
    CLOB_API_KEY, GEMINI_API_KEY, GEMINI_MODEL,
    SIGNAL_WEIGHTS, MIN_EDGE, MIN_CONFIDENCE,
)
from polybot.logger import get_logger

log = get_logger("strategies")


# ── Signal result container ───────────────────────────────────────────────────

@dataclass
class SignalResult:
    strategy: str
    direction: str       = "HOLD"     # "BUY_YES" | "BUY_NO" | "HOLD"
    yes_probability: float = 0.5
    confidence: float    = 0.0
    edge: float          = 0.0        # yes_prob - market_price
    metadata: dict       = field(default_factory=dict)


@dataclass
class CombinedSignal:
    """The aggregated output used by the risk manager and executor."""
    yes_probability: float
    confidence: float
    edge: float                        # signed: + means BUY YES, - means BUY NO
    direction: str                     # "BUY_YES" | "BUY_NO" | "HOLD"
    signal_details: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != "HOLD"
            and self.confidence >= MIN_CONFIDENCE
            and abs(self.edge) >= MIN_EDGE
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: Momentum — price + volume breakout
# ─────────────────────────────────────────────────────────────────────────────

def strategy_momentum(snap, history: pd.DataFrame) -> SignalResult:
    """
    Detects momentum: a sustained price move with rising volume.
    Logic:
      - Compute rolling mean of mid_price over last N rows vs older mean
      - Compute volume acceleration (recent vs prior quarter)
      - A breakout = price moved > threshold AND volume accelerating
    """
    name = "momentum"
    if history.empty or len(history) < 10:
        return SignalResult(strategy=name, metadata={"reason": "insufficient history"})

    df = history.copy().tail(40)
    if "mid_price" not in df.columns:
        return SignalResult(strategy=name, metadata={"reason": "no price column"})

    prices  = df["mid_price"].astype(float)
    volumes = df["volume"].astype(float) if "volume" in df.columns else pd.Series([1.0] * len(df))

    window  = min(8, len(prices) // 2)
    recent_price = prices.iloc[-window:].mean()
    prior_price  = prices.iloc[:-window].mean()
    price_delta  = recent_price - prior_price           # positive = uptrend

    recent_vol   = volumes.iloc[-window:].mean()
    prior_vol    = volumes.iloc[:-window].mean()
    vol_ratio    = recent_vol / max(prior_vol, 1.0)

    # Breakout conditions
    breakout_threshold = 0.04    # 4-cent move
    vol_acceleration   = 1.15    # volume 15% higher than baseline

    if abs(price_delta) < breakout_threshold or vol_ratio < vol_acceleration:
        return SignalResult(
            strategy=name,
            metadata={"price_delta": round(price_delta, 4), "vol_ratio": round(vol_ratio, 3)},
        )

    # Direction
    direction = "BUY_YES" if price_delta > 0 else "BUY_NO"
    raw_conf  = min(1.0, abs(price_delta) / 0.20 * 0.6 + (vol_ratio - 1) * 0.4)

    # Estimated probability follows the momentum
    yes_prob = snap.mid_price + price_delta * 0.5
    yes_prob = max(0.01, min(0.99, yes_prob))
    edge     = yes_prob - snap.mid_price

    return SignalResult(
        strategy=name,
        direction=direction,
        yes_probability=round(yes_prob, 3),
        confidence=round(raw_conf, 3),
        edge=round(edge, 4),
        metadata={
            "price_delta": round(price_delta, 4),
            "vol_ratio":   round(vol_ratio, 3),
            "window":      window,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: Mean Reversion — overbought/oversold detection
# ─────────────────────────────────────────────────────────────────────────────

def strategy_mean_reversion(snap, history: pd.DataFrame) -> SignalResult:
    """
    Computes Bollinger Bands on historical prices.
    - Price > upper band → overbought → BUY_NO (price will revert down)
    - Price < lower band → oversold  → BUY_YES (price will revert up)
    Confidence proportional to distance outside band.
    """
    name = "mean_rev"
    if history.empty or len(history) < 14:
        return SignalResult(strategy=name, metadata={"reason": "insufficient history"})

    prices = history["mid_price"].astype(float).tail(30)
    mean   = prices.mean()
    std    = prices.std()

    if std < 1e-6:     # flat market
        return SignalResult(strategy=name, metadata={"reason": "flat market"})

    upper = mean + 2.0 * std
    lower = mean - 2.0 * std
    current = snap.mid_price

    # Z-score
    z = (current - mean) / std

    if current > upper:
        direction = "BUY_NO"
        yes_prob  = max(0.01, mean - 0.5 * std)       # revert toward mean
        edge      = yes_prob - current
        conf      = min(1.0, (current - upper) / std * 0.5)
    elif current < lower:
        direction = "BUY_YES"
        yes_prob  = min(0.99, mean + 0.5 * std)
        edge      = yes_prob - current
        conf      = min(1.0, (lower - current) / std * 0.5)
    else:
        return SignalResult(
            strategy=name,
            metadata={"z_score": round(z, 3), "mean": round(mean, 4), "std": round(std, 4)},
        )

    return SignalResult(
        strategy=name,
        direction=direction,
        yes_probability=round(yes_prob, 3),
        confidence=round(conf, 3),
        edge=round(edge, 4),
        metadata={"z_score": round(z, 3), "mean": round(mean, 4), "upper": round(upper, 4), "lower": round(lower, 4)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: Probability Gap — market price vs fair value estimate
# ─────────────────────────────────────────────────────────────────────────────

# Historical base-rate map (category → typical YES resolution rate)
_BASE_RATES = {
    "fed":        0.48, "rate":      0.48,
    "bitcoin":    0.42, "btc":       0.42, "crypto": 0.40, "eth": 0.44,
    "election":   0.50, "president": 0.50, "vote":   0.50,
    "apple":      0.62, "iphone":    0.65, "google": 0.60,
    "microsoft":  0.60, "ai":        0.58,
    "war":        0.35, "recession": 0.30, "inflation": 0.55,
    "nba":        0.50, "nfl":       0.50, "sports": 0.50,
    "default":    0.50,
}

def _lookup_base_rate(question: str) -> tuple[float, str]:
    q = question.lower()
    for kw, rate in _BASE_RATES.items():
        if kw in q:
            return rate, kw
    return 0.50, "default"


def strategy_probability_gap(snap) -> SignalResult:
    """
    Estimates fair value using:
      1. Time decay weight (markets near resolution are well-priced)
      2. Volume calibration (thin markets have more pricing errors)
      3. Historical base rates by category
      4. Price extremes correction (markets over-react at 90% / under-react at 10%)
    Signals when market price is significantly off fair value.
    """
    name = "prob_gap"
    p    = snap.mid_price

    # --- Feature 1: base rate ---
    base_rate, matched_kw = _lookup_base_rate(snap.question)

    # --- Feature 2: volume calibration ---
    # Low-volume markets have lower efficiency → bigger expected errors
    vol_factor = max(0.5, min(2.5, 500_000 / max(snap.volume, 1)))

    # --- Feature 3: time decay ---
    # Near-expiry markets are very efficient; distant markets less so
    time_factor = min(1.0, snap.days_to_close / 90)

    # --- Feature 4: extreme price correction ---
    # Market tends to over-price favourites (>0.85) and under-price longshots (<0.12)
    if p > 0.88:
        extreme_adj = -0.04 * vol_factor
    elif p < 0.12:
        extreme_adj = +0.03 * vol_factor
    else:
        extreme_adj = (0.50 - p) * 0.05 * vol_factor       # gentle mean-pull

    fair_value = base_rate * 0.40 + p * 0.60 + extreme_adj * time_factor
    fair_value = max(0.01, min(0.99, fair_value))
    edge       = fair_value - p

    # Only signal if edge is meaningful
    if abs(edge) < 0.04:
        return SignalResult(
            strategy=name,
            metadata={"fair_value": round(fair_value, 4), "edge": round(edge, 4), "kw": matched_kw},
        )

    direction = "BUY_YES" if edge > 0 else "BUY_NO"
    # Confidence grows with edge magnitude and volume inefficiency
    conf = min(1.0, abs(edge) * 3.5 * vol_factor * 0.3)

    return SignalResult(
        strategy=name,
        direction=direction,
        yes_probability=round(fair_value, 4),
        confidence=round(conf, 3),
        edge=round(edge, 4),
        metadata={"fair_value": round(fair_value, 4), "base_rate": base_rate, "kw": matched_kw, "vol_factor": round(vol_factor, 2)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4: News Sentiment (Gemini Google Search grounding)
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_client():
    """Lazy import of Gemini client to allow rest of module to load without it."""
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
    Uses Gemini with Google Search grounding to fetch and score recent news.
    Returns probability of YES resolution based on news analysis.
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
            edge     = score - 0.5               # vs neutral prior
            direction = "BUY_YES" if score > 0.5 else ("BUY_NO" if score < 0.5 else "HOLD")
            return SignalResult(
                strategy=name,
                direction=direction,
                yes_probability=round(score, 3),
                confidence=round(raw_conf, 3),
                edge=round(edge, 3),
                metadata={"summary": data.get("summary", ""), "raw_score": data.get("score")},
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
# STRATEGY 5: Reddit Sentiment
# ─────────────────────────────────────────────────────────────────────────────

_POS_WORDS = {"yes","likely","will","confirmed","bullish","positive","win","up","green","pass"}
_NEG_WORDS = {"no","unlikely","wont","bearish","negative","lose","down","red","fail","rejected","never"}

def strategy_reddit(question: str) -> SignalResult:
    """Scrapes Reddit search for recent posts and scores bullish/bearish sentiment."""
    name = "reddit"
    try:
        keywords = question[:70].replace("?", "").replace("Will ", "")
        r = requests.get(
            "https://www.reddit.com/search.json",
            headers={"User-Agent": "polybot/2.0"},
            params={"q": keywords, "sort": "new", "limit": 30, "t": "week"},
            timeout=10,
        )
        posts = r.json().get("data", {}).get("children", [])
        if not posts:
            return SignalResult(strategy=name, yes_probability=0.5, metadata={"count": 0})

        weighted_scores = []
        for post in posts:
            d = post["data"]
            text  = (d.get("title", "") + " " + d.get("selftext", "")).lower()
            words = set(text.split())
            pos   = len(words & _POS_WORDS)
            neg   = len(words & _NEG_WORDS)
            if pos + neg == 0:
                continue
            upvote_w = min(2.5, 1.0 + d.get("score", 0) / 200)
            sent = pos / (pos + neg)
            weighted_scores.append(sent * upvote_w)

        if not weighted_scores:
            return SignalResult(strategy=name, yes_probability=0.5, metadata={"count": len(posts)})

        avg = sum(weighted_scores) / len(weighted_scores)
        avg = max(0.01, min(0.99, avg))
        conf = min(0.7, len(weighted_scores) / 20)       # max 0.70 confidence from Reddit
        edge = avg - 0.5
        direction = "BUY_YES" if avg > 0.55 else ("BUY_NO" if avg < 0.45 else "HOLD")
        return SignalResult(
            strategy=name,
            direction=direction,
            yes_probability=round(avg, 3),
            confidence=round(conf, 3),
            edge=round(edge, 3),
            metadata={"post_count": len(posts), "scored": len(weighted_scores)},
        )
    except Exception as exc:
        log.warning(f"Reddit signal error: {exc}")
        return SignalResult(strategy=name, yes_probability=0.5, metadata={"reason": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 6: ML Ensemble (heuristic feature model — replace with real ML pkl)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_ml(snap, days_to_resolve: int = 30, history=None) -> SignalResult:
    """
    ML-powered probability estimator.

    Tries the following in order:
      1. Trained XGBoost / Logistic / LSTM model (from polybot/ml/predictor.py)
         — requires running `python train_model.py` first
      2. Heuristic 5-feature weighted ensemble (fallback, no training needed)

    The trained model uses 32 engineered features:
      price returns, RSI, MACD, Bollinger Bands, volume acceleration,
      liquidity ratio, time decay, sentiment, and market efficiency proxies.
    """
    name = "ml"
    p    = snap.mid_price

    # ── Try trained ML model first ─────────────────────────────────────────
    try:
        from polybot.ml.predictor import predictor as _ml_predictor
        import pandas as pd

        hist = history if (history is not None and isinstance(history, pd.DataFrame) and not history.empty) else pd.DataFrame()
        ml_sig = _ml_predictor.predict_and_signal(snap, hist)

        if ml_sig.predicted_prob != 0.5 or ml_sig.model_name != "heuristic":
            score = ml_sig.predicted_prob
            edge  = ml_sig.edge
            direction = ml_sig.direction if ml_sig.direction != "HOLD" else (
                "BUY_YES" if edge > 0.03 else ("BUY_NO" if edge < -0.03 else "HOLD")
            )
            return SignalResult(
                strategy  = name,
                direction = direction,
                yes_probability = round(score, 4),
                confidence      = round(ml_sig.confidence, 3),
                edge            = round(edge, 4),
                metadata  = {
                    "model":       ml_sig.model_name,
                    "high_conf":   ml_sig.is_high_conf,
                    "strong_edge": ml_sig.is_strong_edge,
                    "mispricing":  ml_sig.mispricing_label,
                },
            )
    except Exception as exc:
        log.debug(f"ML predictor unavailable ({exc}) — using heuristic fallback")

    # ── Heuristic fallback ─────────────────────────────────────────────────
    # F1: price prior
    f1 = p

    # F2: time decay
    f2_raw = max(0, min(1, 1.0 - days_to_resolve / 120))
    f2 = f2_raw * f1 + (1 - f2_raw) * 0.5

    # F3: spread efficiency
    spread_penalty = min(0.5, snap.spread * 2)
    f3 = p * (1 - spread_penalty) + 0.5 * spread_penalty

    # F4: category heuristic
    q = snap.question.lower()
    if any(w in q for w in ["fed", "rate", "inflation", "gdp", "fomc"]):
        f4 = 0.52
    elif any(w in q for w in ["bitcoin", "btc", "crypto", "eth", "doge"]):
        f4 = 0.45
    elif any(w in q for w in ["election", "vote", "president", "senate"]):
        f4 = 0.50
    elif any(w in q for w in ["apple", "google", "microsoft", "nvidia", "ai"]):
        f4 = 0.58
    elif any(w in q for w in ["war", "attack", "conflict", "military"]):
        f4 = 0.35
    else:
        f4 = 0.50

    # F5: volume-weighted implied prob
    vol_adj = max(0.5, min(1.5, math.log10(max(snap.volume, 1) + 1) / 6))
    f5 = p * vol_adj + 0.5 * (1 - vol_adj)

    weights  = [0.35, 0.20, 0.15, 0.15, 0.15]
    features = [f1, f2, f3, f4, f5]
    score    = sum(w * f for w, f in zip(weights, features))
    score    = max(0.01, min(0.99, score))
    edge     = score - p

    direction = "BUY_YES" if edge > 0.03 else ("BUY_NO" if edge < -0.03 else "HOLD")
    conf      = min(0.80, abs(edge) * 4.0)

    return SignalResult(
        strategy  = name,
        direction = direction,
        yes_probability = round(score, 4),
        confidence      = round(conf, 3),
        edge            = round(edge, 4),
        metadata  = {
            "model":    "heuristic",
            "features": dict(zip(["price","time","spread","cat","vol"], [round(f,4) for f in features])),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATOR: Combine all signals with configurable weights
# ─────────────────────────────────────────────────────────────────────────────

def run_all_strategies(snap, history: pd.DataFrame) -> CombinedSignal:
    """
    Run all 6 strategies, weight their yes_probability estimates,
    compute aggregate confidence, and return a CombinedSignal.
    """
    from polybot.data_layer import MarketSnapshot   # avoid circular at import time

    results: dict[str, SignalResult] = {
        "momentum": strategy_momentum(snap, history),
        "mean_rev": strategy_mean_reversion(snap, history),
        "prob_gap": strategy_probability_gap(snap),
        "news":     strategy_news(snap.question),
        "reddit":   strategy_reddit(snap.question),
        "ml":       strategy_ml(snap, snap.days_to_close, history=history),
        "base_rate": SignalResult(
            strategy="base_rate",
            yes_probability=_lookup_base_rate(snap.question)[0],
            confidence=0.5,
        ),
    }

    weights = SIGNAL_WEIGHTS
    total_w = sum(weights.values())

    # Weighted average probability
    yes_prob = sum(weights.get(k, 0) * r.yes_probability for k, r in results.items()) / total_w

    # Confidence = weighted avg of individual confidences, penalised by disagreement
    avg_conf = sum(weights.get(k, 0) * r.confidence for k, r in results.items()) / total_w
    probs    = [r.yes_probability for r in results.values()]
    variance = sum((p - yes_prob) ** 2 for p in probs) / len(probs)
    disagreement_penalty = min(0.40, variance * 4)
    confidence = max(0.0, avg_conf - disagreement_penalty)

    edge      = yes_prob - snap.mid_price
    direction = "BUY_YES" if edge > MIN_EDGE else ("BUY_NO" if edge < -MIN_EDGE else "HOLD")

    signal_details = {k: {
        "yes_prob":  r.yes_probability,
        "confidence": r.confidence,
        "direction": r.direction,
        "edge":      r.edge,
        "meta":      r.metadata,
    } for k, r in results.items()}

    log.debug(f"Combined: yes_prob={yes_prob:.3f} conf={confidence:.3f} edge={edge:.4f} dir={direction}")
    return CombinedSignal(
        yes_probability=round(yes_prob, 4),
        confidence=round(confidence, 3),
        edge=round(edge, 4),
        direction=direction,
        signal_details=signal_details,
    )
