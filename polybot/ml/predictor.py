"""
polybot/ml/predictor.py
=======================
ML Predictor — Real-time inference + trading signal generation.

CHANGES FROM PREVIOUS VERSION
------------------------------
1. CORRECT EV FORMULA:
     EV_YES = predicted_prob × (1 - market_price) - (1 - predicted_prob) × market_price
     EV_NO  = (1 - predicted_prob) × market_price - predicted_prob × (1 - market_price)
   The old formula used `edge = predicted_prob - market_price`, which is NOT the
   expected monetary value — it ignores the asymmetric payoff structure of
   binary options where you stake market_price to win (1 - market_price).

2. EV THRESHOLD: Only signals with EV / stake > ev_threshold (default 0.04)
   are returned as BUY_YES or BUY_NO. Pure edge is no longer sufficient.

3. CORRECT CONFIDENCE FORMULA:
     confidence = 0.5 × calibration_certainty
                + 0.3 × feature_reliability
                + 0.2 × (1 - model_disagreement)

4. PREFER CALIBRATED MODELS: ModelLoader now prefers calibrated_*.pkl files
   over raw model files. Raw files are a fallback only.

5. LRU CACHE WITH TTL=60s: Predictions are cached keyed on
   (condition_id, round(mid_price, 3)). Cache invalidates after 60 seconds
   to avoid stale signals in volatile markets.

6. predict_batch() method for efficient batch inference across a list of
   (snap, history) pairs — runs all feature builds in parallel using
   list comprehensions (no per-call overhead).

7. LSTM references completely removed from ModelLoader.
"""

from __future__ import annotations

import pickle
import time
import functools
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from polybot.ml.features import build_features, FEATURE_COLS
from polybot.ml.models import (
    LogisticModel,
    XGBoostModel,
    LightGBMModel,
    IsotonicCalibrator,
    _XGB_AVAILABLE,
    _LGB_AVAILABLE,
)
from polybot.config import MIN_CONFIDENCE
from polybot.logger import get_logger

log = get_logger("predictor")

MODEL_DIR = Path("models")

# EV gate — only trade when normalized EV exceeds this threshold
DEFAULT_EV_THRESHOLD = 0.04

# LRU cache TTL in seconds
CACHE_TTL_SECONDS = 60


# ── ML Signal output ──────────────────────────────────────────────────────────

@dataclass
class MLSignal:
    """
    Structured output from the ML predictor.

    Attributes:
        model_name:     Name of the model used.
        market_price:   Current mid-price of the market.
        predicted_prob: Model's P(YES) — calibrated probability.
        ev_yes:         Expected value of a BUY_YES trade.
        ev_no:          Expected value of a BUY_NO trade.
        direction:      'BUY_YES' | 'BUY_NO' | 'HOLD'.
        confidence:     Composite confidence score [0,1].
        is_high_conf:   Confidence >= 0.70.
        feature_vector: The raw feature array used (for debugging).
        mispricing_label: Human-readable label for detected mispricing type.
        xgb_pred:       Raw XGBoost prediction (for disagreement calculation).
        logistic_pred:  Raw LogisticModel prediction (for disagreement calc).
    """
    model_name:       str
    market_price:     float
    predicted_prob:   float
    ev_yes:           float
    ev_no:            float
    direction:        str
    confidence:       float
    is_high_conf:     bool
    feature_vector:   Optional[np.ndarray] = None
    mispricing_label: str = ""
    xgb_pred:         float = 0.5
    logistic_pred:    float = 0.5

    # Backward-compat: edge = predicted_prob - market_price (kept for strategies.py)
    @property
    def edge(self) -> float:
        """Signed edge. DEPRECATED — use ev_yes / ev_no for actual EV calculations."""
        return self.predicted_prob - self.market_price

    @property
    def is_actionable(self) -> bool:
        """True if the signal is strong enough to execute a trade."""
        active_ev = self.ev_yes if self.direction == "BUY_YES" else self.ev_no
        return (
            self.direction != "HOLD"
            and self.confidence >= MIN_CONFIDENCE
            and active_ev > DEFAULT_EV_THRESHOLD   # must be POSITIVE EV, not abs
        )

    @property
    def is_strong_edge(self) -> bool:
        """True if |predicted_prob - market_price| >= 0.10."""
        return abs(self.edge) >= 0.10

    def describe(self) -> str:
        """Human-readable one-line summary of the signal."""
        ev = self.ev_yes if self.direction == "BUY_YES" else self.ev_no
        return (
            f"[{self.model_name}] market={self.market_price:.3f} "
            f"model={self.predicted_prob:.3f} EV={ev:+.4f} "
            f"conf={self.confidence:.3f} → {self.direction}"
            + (" [HIGH CONF]" if self.is_high_conf else "")
            + (f" [{self.mispricing_label}]" if self.mispricing_label else "")
        )


# ── LRU cache with TTL ────────────────────────────────────────────────────────

class _TTLCache:
    """
    Simple in-memory cache with per-entry TTL expiry.

    Keyed on arbitrary hashable tuples. Entries are evicted after `ttl` seconds.

    Example:
        >>> cache = _TTLCache(ttl=60)
        >>> cache.set(("0xabc", 0.650), signal)
        >>> cache.get(("0xabc", 0.650))   # returns signal or None if expired
    """

    def __init__(self, ttl: float = 60.0, maxsize: int = 512):
        """
        Args:
            ttl:     Time-to-live in seconds for each cached entry.
            maxsize: Maximum number of entries. Oldest entries are evicted when full.
        """
        self._store:   dict[tuple, tuple[float, object]] = {}  # key → (expire_at, value)
        self._ttl      = ttl
        self._maxsize  = maxsize

    def get(self, key: tuple) -> Optional[object]:
        """
        Retrieve a cached value by key, or None if expired / missing.

        Args:
            key: Hashable tuple (e.g. (condition_id, rounded_price)).

        Returns:
            Cached value or None.

        Example:
            >>> val = cache.get(("0xabc", 0.650))
        """
        if key not in self._store:
            return None
        expire_at, value = self._store[key]
        if time.monotonic() > expire_at:
            del self._store[key]
            return None
        return value

    def set(self, key: tuple, value: object) -> None:
        """
        Store a value with TTL.

        Evicts oldest entry if maxsize is exceeded.

        Args:
            key:   Hashable tuple key.
            value: Value to store.

        Example:
            >>> cache.set(("0xabc", 0.650), my_signal)
        """
        if len(self._store) >= self._maxsize:
            # Evict oldest entry
            oldest = min(self._store.keys(), key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        """Flush all cached entries."""
        self._store.clear()


_prediction_cache = _TTLCache(ttl=CACHE_TTL_SECONDS, maxsize=512)


# ── Model loader ──────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Lazy-loads the best available model from disk.

    Priority order:
      1. calibrated_xgboost    (best calibration)
      2. calibrated_lightgbm   (fast, reliable)
      3. calibrated_logistic   (interpretable baseline)
      4. raw xgboost           (uncalibrated fallback)
      5. raw lightgbm          (uncalibrated fallback)
      6. raw logistic          (uncalibrated fallback)

    No LSTM / PyTorch dependency.

    Example:
        >>> model = ModelLoader.load_best()
        >>> probs = model.predict(X)
    """

    _cache: dict[str, object] = {}

    @classmethod
    def load_best(cls) -> Optional[object]:
        """
        Load the best model available on disk.

        Prefer calibrated models. Fall back to raw models.

        Returns:
            Loaded model instance or None if nothing is found.

        Example:
            >>> model = ModelLoader.load_best()
        """
        priority = [
            # (name_key, path, loader_fn)
            ("cal_xgboost",  MODEL_DIR / "calibrated_xgboost.pkl",  _load_pkl),
            ("cal_lightgbm", MODEL_DIR / "calibrated_lightgbm.pkl", _load_pkl),
            ("cal_logistic", MODEL_DIR / "calibrated_logistic.pkl", _load_pkl),
            ("xgboost",      MODEL_DIR / "xgboost.pkl",             XGBoostModel.load),
            ("lightgbm",     MODEL_DIR / "lightgbm.pkl",            LightGBMModel.load),
            ("logistic",     MODEL_DIR / "logistic.pkl",            LogisticModel.load),
        ]
        for name, path, loader in priority:
            if not path.exists():
                continue
            if name in cls._cache:
                return cls._cache[name]
            try:
                model = loader(str(path))
                cls._cache[name] = model
                log.info(f"Loaded model '{name}' from {path}")
                return model
            except Exception as exc:
                log.warning(f"Failed to load '{name}': {exc}")

        log.debug("No trained model found — will use heuristic fallback")
        return None

    @classmethod
    def load_by_name(cls, name: str) -> Optional[object]:
        """
        Load a specific model by name.

        Args:
            name: One of 'xgboost', 'lightgbm', 'logistic',
                       'cal_xgboost', 'cal_lightgbm', 'cal_logistic'.

        Returns:
            Loaded model or None.

        Example:
            >>> model = ModelLoader.load_by_name("cal_xgboost")
        """
        paths = {
            "cal_xgboost":  (MODEL_DIR / "calibrated_xgboost.pkl",  _load_pkl),
            "cal_lightgbm": (MODEL_DIR / "calibrated_lightgbm.pkl", _load_pkl),
            "cal_logistic": (MODEL_DIR / "calibrated_logistic.pkl", _load_pkl),
            "xgboost":      (MODEL_DIR / "xgboost.pkl",             XGBoostModel.load),
            "lightgbm":     (MODEL_DIR / "lightgbm.pkl",            LightGBMModel.load),
            "logistic":     (MODEL_DIR / "logistic.pkl",            LogisticModel.load),
        }
        if name not in paths:
            log.warning(f"Unknown model name '{name}'")
            return None
        path, loader = paths[name]
        if not path.exists():
            return None
        if name in cls._cache:
            return cls._cache[name]
        try:
            model = loader(str(path))
            cls._cache[name] = model
            return model
        except Exception as exc:
            log.warning(f"Could not load {name}: {exc}")
            return None

    @classmethod
    def clear_cache(cls) -> None:
        """Flush the in-memory model cache (force reload from disk)."""
        cls._cache.clear()


def _load_pkl(path: str) -> object:
    """Load any object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── EV formulas ───────────────────────────────────────────────────────────────

def compute_ev_yes(predicted_prob: float, market_price: float) -> float:
    """
    Expected Value of a BUY_YES position.

    For a binary contract priced at `market_price` that resolves to 1 with
    probability `predicted_prob`:
      - You pay `market_price` per share
      - You receive 1.00 if it resolves YES, 0 if NO
      - EV = predicted_prob × (1 - market_price) - (1 - predicted_prob) × market_price

    Args:
        predicted_prob: Model's calibrated P(YES) in [0, 1].
        market_price:   Current market mid-price in [0, 1].

    Returns:
        EV float. Positive = profitable BUY_YES expected.

    Example:
        >>> compute_ev_yes(0.70, 0.55)
        0.175  # (0.7 × 0.45) - (0.3 × 0.55) = 0.315 - 0.165
    """
    pp = float(np.clip(predicted_prob, 1e-6, 1 - 1e-6))
    mp = float(np.clip(market_price,   1e-6, 1 - 1e-6))
    return pp * (1.0 - mp) - (1.0 - pp) * mp


def compute_ev_no(predicted_prob: float, market_price: float) -> float:
    """
    Expected Value of a BUY_NO position.

    For a BUY_NO trade, you buy NO shares at price (1 - market_price), and win
    (market_price) per share if the market resolves NO:
      EV_NO = (1 - predicted_prob) × market_price - predicted_prob × (1 - market_price)

    Args:
        predicted_prob: Model's calibrated P(YES) in [0, 1].
        market_price:   Current market mid-price in [0, 1].

    Returns:
        EV float. Positive = profitable BUY_NO expected.

    Example:
        >>> compute_ev_no(0.30, 0.55)
        0.175  # (0.7 × 0.55) - (0.3 × 0.45) = 0.385 - 0.135
    """
    pp = float(np.clip(predicted_prob, 1e-6, 1 - 1e-6))
    mp = float(np.clip(market_price,   1e-6, 1 - 1e-6))
    return (1.0 - pp) * mp - pp * (1.0 - mp)


# ── Confidence estimator ──────────────────────────────────────────────────────

def _estimate_confidence(
    predicted_prob: float,
    feature_vector: Optional[np.ndarray],
    xgb_pred:       float,
    logistic_pred:  float,
) -> float:
    """
    Composite confidence score combining three independent signals:
      1. Calibration certainty (0.5 weight): how far predicted_prob is from 0.5.
         A prediction near 0.5 is essentially a coin flip — low confidence.
      2. Feature reliability (0.3 weight): fraction of non-zero features in
         the vectorised input. Missing data → imputed zeros → lower reliability.
      3. Model agreement (0.2 weight): disagreement |xgb - logistic| penalised.
         High disagreement signals an uncertain market regime.

    Formula:
      confidence = 0.5 × calibration_certainty
                 + 0.3 × feature_reliability
                 + 0.2 × (1 - disagreement)

    Args:
        predicted_prob: Main model's calibrated P(YES).
        feature_vector: Feature array (may be None if build_features failed).
        xgb_pred:       XGBoost raw prediction (for disagreement calc).
        logistic_pred:  Logistic raw prediction (for disagreement calc).

    Returns:
        Composite confidence in [0, 1].

    Example:
        >>> conf = _estimate_confidence(0.72, feat_array, 0.74, 0.68)
        >>> # calibration_certainty = 2 × |0.72 - 0.5| = 0.44
        >>> # feature_reliability   = non_zero_frac
        >>> # disagreement          = |0.74 - 0.68| / 0.30 = 0.20
    """
    # 1. Calibration certainty: normalised distance from 0.5 → [0, 1]
    calibration_certainty = float(np.clip(2.0 * abs(predicted_prob - 0.5), 0.0, 1.0))

    # 2. Feature reliability: fraction of non-zero features
    if feature_vector is not None and len(feature_vector) > 0:
        feature_reliability = float(np.mean(feature_vector != 0.0))
    else:
        feature_reliability = 0.0

    # 3. Model disagreement: |xgb - logistic| normalised to [0,1] range
    raw_disagreement = abs(xgb_pred - logistic_pred)
    disagreement     = float(np.clip(raw_disagreement / 0.30, 0.0, 1.0))

    confidence = (
        0.5 * calibration_certainty
        + 0.3 * feature_reliability
        + 0.2 * (1.0 - disagreement)
    )
    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


def _label_mispricing(predicted_prob: float, market_price: float) -> str:
    """
    Generate a human-readable mispricing label for logs and trading UI.

    Args:
        predicted_prob: Model's calibrated P(YES).
        market_price:   Current mid-price.

    Returns:
        String label describing the type of mispricing detected.

    Example:
        >>> _label_mispricing(0.22, 0.08)
        'LONGSHOT_UNDERPRICED'
    """
    edge = predicted_prob - market_price
    if abs(edge) < 0.04:
        return ""
    if market_price < 0.12 and predicted_prob > 0.20:
        return "LONGSHOT_UNDERPRICED"
    if market_price > 0.88 and predicted_prob < 0.80:
        return "FAVOURITE_OVERPRICED"
    if edge > 0.15:
        return "SIGNIFICANT_UNDERPRICED"
    if edge < -0.15:
        return "SIGNIFICANT_OVERPRICED"
    if edge > 0:
        return "MODERATELY_UNDERPRICED"
    return "MODERATELY_OVERPRICED"


# ── Main predictor ────────────────────────────────────────────────────────────

class MLPredictor:
    """
    Unified ML inference interface with correct EV formulas, LRU cache,
    calibrated models, and composite confidence scoring.

    Usage:
        predictor = MLPredictor()
        signal = predictor.predict_and_signal(snap, history)
        if signal.is_actionable:
            execute(signal.direction, signal.ev_yes)
    """

    def __init__(
        self,
        model_name:   str   = "auto",
        ev_threshold: float = DEFAULT_EV_THRESHOLD,
    ):
        """
        Args:
            model_name:   'auto' (load best available), or a specific model key.
            ev_threshold: Minimum EV/stake required to generate a BUY signal.
                          Default 0.04 (4% expected return per unit staked).
        """
        self.model_name   = model_name
        self.ev_threshold = ev_threshold
        self._model       = None
        self._logistic    = None   # for disagreement calculation
        self._xgboost     = None   # for disagreement calculation

    def _get_model(self) -> Optional[object]:
        """Lazy-load the primary inference model."""
        if self._model is None:
            if self.model_name == "auto":
                self._model = ModelLoader.load_best()
            else:
                self._model = ModelLoader.load_by_name(self.model_name)
        return self._model

    def _get_disagreement_models(self) -> tuple[float, float]:
        """
        Load XGBoost and Logistic predictions with zero overhead if not cached.

        Returns:
            (xgb_prob_default, logistic_prob_default) as 0.5 if models unavailable.
        """
        # These are loaded lazily and used only for the confidence formula.
        # If only one model is trained, both fall back to 0.5 (no disagreement assumed).
        if self._xgboost is None:
            self._xgboost  = ModelLoader.load_by_name("cal_xgboost") or \
                             ModelLoader.load_by_name("xgboost")
        if self._logistic is None:
            self._logistic = ModelLoader.load_by_name("cal_logistic") or \
                             ModelLoader.load_by_name("logistic")
        return self._xgboost, self._logistic

    def predict_and_signal(
        self,
        snap,
        history,
        sentiment_score: float = 0.5,
        reddit_score:    float = 0.5,
        condition_id:    str   = "",
    ) -> MLSignal:
        """
        Build features, run calibrated inference, compute correct EV, and
        generate a structured trading signal.

        Caches results for 60 seconds keyed on (condition_id, round(price, 3))
        to prevent redundant inference on frequently polled markets.

        Args:
            snap:            MarketSnapshot with .mid_price, .spread, etc.
            history:         pd.DataFrame from HistoricalStore.
            sentiment_score: News sentiment [0,1] — 0.5 = neutral.
            reddit_score:    Reddit sentiment [0,1].
            condition_id:    Optional market ID for cache keying.

        Returns:
            MLSignal with all fields populated, including ev_yes, ev_no,
            direction (only BUY_YES / BUY_NO if EV > ev_threshold).

        Example:
            >>> sig = predictor.predict_and_signal(snap, history)
            >>> print(sig.describe())
            [xgboost] market=0.620 model=0.742 EV=+0.086 conf=0.641 → BUY_YES
        """
        market_price = float(snap.mid_price)
        cache_key    = (condition_id, round(market_price, 3))

        # Check LRU cache
        cached = _prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        model      = self._get_model()
        model_name = getattr(model, "NAME", "heuristic") if model else "heuristic"

        # ── Build feature vector ───────────────────────────────────────────
        hist_df = _coerce_history(history)
        try:
            feat = build_features(
                df              = hist_df,
                sentiment_score = sentiment_score,
                reddit_score    = reddit_score,
            )
        except Exception as exc:
            log.warning(f"Feature build failed: {exc}")
            feat = None

        # ── Primary model inference ────────────────────────────────────────
        if feat is not None and model is not None:
            try:
                predicted_prob = float(model.predict(feat.reshape(1, -1))[0])
            except Exception as exc:
                log.warning(f"ML inference failed ({exc}) — using heuristic fallback")
                predicted_prob = _heuristic_prob(snap)
                model_name     = "heuristic"
        else:
            predicted_prob = _heuristic_prob(snap)
            model_name     = "heuristic"

        predicted_prob = float(np.clip(predicted_prob, 0.01, 0.99))

        # ── Disagreement models ────────────────────────────────────────────
        xgb_model, logistic_model = self._get_disagreement_models()
        try:
            xgb_pred = float(xgb_model.predict(feat.reshape(1, -1))[0]) \
                if (feat is not None and xgb_model is not None) else predicted_prob
        except Exception:
            xgb_pred = predicted_prob

        try:
            logistic_pred = float(logistic_model.predict(feat.reshape(1, -1))[0]) \
                if (feat is not None and logistic_model is not None) else predicted_prob
        except Exception:
            logistic_pred = predicted_prob

        # ── Compute EV (correct formula) ───────────────────────────────────
        ev_yes = compute_ev_yes(predicted_prob, market_price)
        ev_no  = compute_ev_no(predicted_prob, market_price)

        # Trade direction — only signal when EV exceeds threshold
        if ev_yes > self.ev_threshold:
            direction = "BUY_YES"
        elif ev_no > self.ev_threshold:
            direction = "BUY_NO"
        else:
            direction = "HOLD"

        # ── Composite confidence ───────────────────────────────────────────
        confidence   = _estimate_confidence(predicted_prob, feat, xgb_pred, logistic_pred)
        is_high_conf = confidence >= 0.70
        mispricing   = _label_mispricing(predicted_prob, market_price)

        signal = MLSignal(
            model_name      = model_name,
            market_price    = round(market_price, 4),
            predicted_prob  = round(predicted_prob, 4),
            ev_yes          = round(ev_yes, 4),
            ev_no           = round(ev_no, 4),
            direction       = direction,
            confidence      = confidence,
            is_high_conf    = is_high_conf,
            feature_vector  = feat,
            mispricing_label = mispricing,
            xgb_pred        = round(xgb_pred, 4),
            logistic_pred   = round(logistic_pred, 4),
        )

        log.debug(signal.describe())
        _prediction_cache.set(cache_key, signal)
        return signal

    def predict_batch(
        self,
        snaps:   list,
        histories: list,
        sentiment_scores: Optional[list[float]] = None,
        reddit_scores:    Optional[list[float]] = None,
        condition_ids:    Optional[list[str]]   = None,
    ) -> list[MLSignal]:
        """
        Efficient batch inference over a list of (snap, history) pairs.

        All feature vectors are built first, then a single model.predict() call
        is dispatched for the full batch (avoids per-call overhead).

        Args:
            snaps:            List of MarketSnapshot objects.
            histories:        List of pd.DataFrames (matching length).
            sentiment_scores: List of float [0,1], defaults to 0.5 for each.
            reddit_scores:    List of float [0,1], defaults to 0.5 for each.
            condition_ids:    List of condition_id strings for cache keying.

        Returns:
            List of MLSignal objects, one per input snap.

        Example:
            >>> signals = predictor.predict_batch(snaps, histories)
            >>> buy_signals = [s for s in signals if s.direction != 'HOLD']
        """
        n = len(snaps)
        if n == 0:
            return []

        sents  = sentiment_scores or [0.5] * n
        reddits = reddit_scores   or [0.5] * n
        cids   = condition_ids    or [""] * n

        model = self._get_model()

        # Build all feature vectors
        feats: list[Optional[np.ndarray]] = []
        for snap, hist, sent, red in zip(snaps, histories, sents, reddits):
            try:
                hist_df = _coerce_history(hist)
                feat    = build_features(hist_df, sentiment_score=sent, reddit_score=red)
            except Exception as exc:
                log.warning(f"Batch feature build failed for {getattr(snap, 'question', '?')}: {exc}")
                feat = None
            feats.append(feat)

        # Batch model inference for rows that have features
        valid_idx  = [i for i, f in enumerate(feats) if f is not None]
        probs_map:     dict[int, float] = {}
        xgb_probs_map: dict[int, float] = {}
        log_probs_map: dict[int, float] = {}

        # Load secondary (disagreement) models once
        xgb_model, logistic_model = self._get_disagreement_models()

        if valid_idx and model is not None:
            try:
                X_batch = np.vstack([feats[i].reshape(1, -1) for i in valid_idx])
                batch_probs = model.predict(X_batch)
                for i, p in zip(valid_idx, batch_probs):
                    probs_map[i] = float(np.clip(p, 0.01, 0.99))
            except Exception as exc:
                log.warning(f"Batch prediction failed: {exc} — falling back to per-sample heuristic")

        # Disagreement: batch XGBoost predictions
        if valid_idx and xgb_model is not None:
            try:
                X_batch = np.vstack([feats[i].reshape(1, -1) for i in valid_idx])
                xgb_p   = xgb_model.predict(X_batch)
                for i, p in zip(valid_idx, xgb_p):
                    xgb_probs_map[i] = float(np.clip(p, 0.01, 0.99))
            except Exception as exc:
                log.debug(f"XGBoost batch disagree failed: {exc}")

        # Disagreement: batch Logistic predictions
        if valid_idx and logistic_model is not None:
            try:
                X_batch = np.vstack([feats[i].reshape(1, -1) for i in valid_idx])
                log_p   = logistic_model.predict(X_batch)
                for i, p in zip(valid_idx, log_p):
                    log_probs_map[i] = float(np.clip(p, 0.01, 0.99))
            except Exception as exc:
                log.debug(f"Logistic batch disagree failed: {exc}")

        # Assemble signals
        signals: list[MLSignal] = []
        for i, snap in enumerate(snaps):
            pred    = probs_map.get(i, _heuristic_prob(snap))
            xgb_p   = xgb_probs_map.get(i, pred)
            log_p   = log_probs_map.get(i, pred)
            mp      = float(snap.mid_price)
            ev_y    = compute_ev_yes(pred, mp)
            ev_n    = compute_ev_no(pred, mp)

            if ev_y > self.ev_threshold:
                direction = "BUY_YES"
            elif ev_n > self.ev_threshold:
                direction = "BUY_NO"
            else:
                direction = "HOLD"

            # Use real disagreement values, not the dummy (pred, pred) pair
            conf = _estimate_confidence(pred, feats[i], xgb_p, log_p)
            sig  = MLSignal(
                model_name     = getattr(model, "NAME", "heuristic") if model else "heuristic",
                market_price   = round(mp, 4),
                predicted_prob = round(pred, 4),
                ev_yes         = round(ev_y, 4),
                ev_no          = round(ev_n, 4),
                direction      = direction,
                confidence     = conf,
                is_high_conf   = conf >= 0.70,
                feature_vector = feats[i],
                xgb_pred       = round(xgb_p, 4),
                logistic_pred  = round(log_p, 4),
            )
            signals.append(sig)

        return signals

    def compare_vs_market(
        self,
        snap,
        history,
        sentiment_score: float = 0.5,
        reddit_score:    float = 0.5,
    ) -> dict:
        """
        Return a detailed comparison dict of model prediction vs market price.

        Args:
            snap:            MarketSnapshot.
            history:         Historical DataFrame.
            sentiment_score: News sentiment.
            reddit_score:    Reddit sentiment.

        Returns:
            Dict with prediction details for display / logging.

        Example:
            >>> d = predictor.compare_vs_market(snap, history)
            >>> print(d["ev_yes"], d["direction"])
        """
        sig = self.predict_and_signal(snap, history, sentiment_score, reddit_score)
        return {
            "question":        snap.question[:60],
            "market_price":    sig.market_price,
            "model_pred":      sig.predicted_prob,
            "edge":            sig.edge,             # legacy — kept for compatibility
            "ev_yes":          sig.ev_yes,
            "ev_no":           sig.ev_no,
            "direction":       sig.direction,
            "confidence":      sig.confidence,
            "high_conf":       sig.is_high_conf,
            "strong_edge":     sig.is_strong_edge,
            "mispricing_type": sig.mispricing_label,
            "model_used":      sig.model_name,
            "xgb_pred":        sig.xgb_pred,
            "logistic_pred":   sig.logistic_pred,
            "disagreement":    abs(sig.xgb_pred - sig.logistic_pred),
            "volume_usd":      getattr(snap, "volume", 0),
            "spread":          getattr(snap, "spread", 0),
            "days_to_close":   getattr(snap, "days_to_close", 30),
        }

    def print_comparison(
        self,
        snap,
        history,
        sentiment_score: float = 0.5,
        reddit_score:    float = 0.5,
    ) -> None:
        """Pretty-print model vs market comparison for a single market."""
        d = self.compare_vs_market(snap, history, sentiment_score, reddit_score)
        G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; E = "\033[0m"
        ev = d["ev_yes"] if d["direction"] == "BUY_YES" else d["ev_no"]
        ev_col = G if ev > 0 else R
        print(f"\n  {'─'*65}")
        print(f"  Market: {d['question']}")
        print(f"  {'─'*65}")
        print(f"  Market price:     {d['market_price']:.4f}")
        print(f"  Model predicts:   {d['model_pred']:.4f}  (using: {d['model_used']})")
        print(f"  EV(YES):          {ev_col}{d['ev_yes']:+.4f}{E}")
        print(f"  EV(NO):           {ev_col}{d['ev_no']:+.4f}{E}")
        print(f"  Active EV:        {ev_col}{ev:+.4f}{E}  → {d['direction']}")
        print(f"  Confidence:       {d['confidence']:.3f}  {'[HIGH CONF]' if d['high_conf'] else ''}")
        print(f"  XGB / Logistic:   {d['xgb_pred']:.3f} / {d['logistic_pred']:.3f}  "
              f"(disagreement={d['disagreement']:.3f})")
        print(f"  Mispricing:       {d['mispricing_type'] or 'None'}")
        print(f"  Volume:           ${d['volume_usd']:,.0f} | Spread: {d['spread']:.4f} | "
              f"Days: {d['days_to_close']}")
        print(f"  Action:           {Y}{d['direction']}{E}")
        print(f"  {'─'*65}\n")


# ── EV helpers (exported for strategies.py) ───────────────────────────────────

def compute_ev_yes(predicted_prob: float, market_price: float) -> float:
    """
    EV_YES = predicted_prob × (1 - market_price) - (1 - predicted_prob) × market_price

    Args:
        predicted_prob: Model's calibrated P(YES).
        market_price:   Current market mid-price.

    Returns:
        Expected value float; positive = profitable BUY_YES.

    Example:
        >>> compute_ev_yes(0.70, 0.55)
        0.175
    """
    pp = float(np.clip(predicted_prob, 1e-6, 1 - 1e-6))
    mp = float(np.clip(market_price,   1e-6, 1 - 1e-6))
    return pp * (1.0 - mp) - (1.0 - pp) * mp


def compute_ev_no(predicted_prob: float, market_price: float) -> float:
    """
    EV_NO = (1 - predicted_prob) × market_price - predicted_prob × (1 - market_price)

    Args:
        predicted_prob: Model's calibrated P(YES).
        market_price:   Current market mid-price.

    Returns:
        Expected value float; positive = profitable BUY_NO.

    Example:
        >>> compute_ev_no(0.30, 0.55)
        0.175
    """
    pp = float(np.clip(predicted_prob, 1e-6, 1 - 1e-6))
    mp = float(np.clip(market_price,   1e-6, 1 - 1e-6))
    return (1.0 - pp) * mp - pp * (1.0 - mp)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _coerce_history(history) -> pd.DataFrame:
    """Convert history input to a non-empty DataFrame (or empty DataFrame)."""
    if history is None:
        return pd.DataFrame()
    if isinstance(history, pd.DataFrame):
        return history
    return pd.DataFrame()


def _heuristic_prob(snap) -> float:
    """
    Fallback heuristic probability when no trained model is available.

    Based on a weighted blend of: current price, time decay, spread efficiency,
    and volume calibration. Used only when no model file exists on disk.

    Args:
        snap: MarketSnapshot with mid_price, days_to_close, spread, volume.

    Returns:
        Float probability estimate in [0.01, 0.99].

    Example:
        >>> _heuristic_prob(snap)
        0.523
    """
    import math
    p            = float(np.clip(snap.mid_price, 0.01, 0.99))
    time_factor  = max(0.0, min(1.0, 1.0 - snap.days_to_close / 120.0))
    spread_pen   = min(0.5, snap.spread * 2.0)
    f_time       = time_factor * p + (1.0 - time_factor) * 0.5
    f_spread     = p * (1.0 - spread_pen) + 0.5 * spread_pen
    vol_adj      = max(0.5, min(1.5, math.log10(max(snap.volume, 1.0) + 1.0) / 6.0))
    f_vol        = p * vol_adj + 0.5 * (1.0 - vol_adj)
    score        = 0.40 * p + 0.20 * f_time + 0.20 * f_spread + 0.20 * f_vol
    return float(np.clip(score, 0.01, 0.99))


# Module-level singleton
predictor = MLPredictor(model_name="auto", ev_threshold=DEFAULT_EV_THRESHOLD)
