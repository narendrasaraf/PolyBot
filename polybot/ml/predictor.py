"""
polybot/ml/predictor.py
=======================
ML Predictor — Real-time inference + trading signal generation.

Takes a live MarketSnapshot + historical DataFrame, builds the feature
vector, runs the trained model, compares predicted probability against
the market price, and generates a structured trading signal.

Key function:
  predict_and_signal(snap, history) → MLSignal

Decision logic:
  - Predicted prob > market_price + MIN_EDGE → BUY YES
  - Predicted prob < market_price - MIN_EDGE → BUY NO
  - Confidence >= 0.70 → HIGH CONFIDENCE trade signal
"""

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

from polybot.ml.features import build_features, FEATURE_COLS
from polybot.ml.models import LogisticModel, XGBoostModel, LSTMModel, _XGB_AVAILABLE, _TORCH_AVAILABLE
from polybot.config import MIN_EDGE, MIN_CONFIDENCE
from polybot.logger import get_logger

log = get_logger("predictor")

MODEL_DIR    = Path("models")
HIGH_CONF_THRESHOLD = 0.70      # signals above this are "high confidence"
STRONG_EDGE_THRESHOLD = 0.10    # edge > 10% = strong mispricing


# ── ML Signal output ──────────────────────────────────────────────────────────

@dataclass
class MLSignal:
    """Structured output from the ML predictor."""
    model_name:       str
    market_price:     float
    predicted_prob:   float         # model's P(YES)
    edge:             float         # predicted_prob - market_price
    direction:        str           # "BUY_YES" | "BUY_NO" | "HOLD"
    confidence:       float         # model confidence [0,1]
    is_high_conf:     bool          # confidence >= 0.70
    is_strong_edge:   bool          # |edge| >= 0.10
    feature_vector:   Optional[np.ndarray] = None
    mispricing_label: str = ""

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != "HOLD"
            and self.confidence >= MIN_CONFIDENCE
            and abs(self.edge) >= MIN_EDGE
        )

    def describe(self) -> str:
        return (
            f"[{self.model_name}] market={self.market_price:.3f} "
            f"model={self.predicted_prob:.3f} edge={self.edge:+.4f} "
            f"conf={self.confidence:.3f} → {self.direction}"
            + (" [HIGH CONF]" if self.is_high_conf else "")
            + (" [STRONG EDGE]" if self.is_strong_edge else "")
            + (f" [{self.mispricing_label}]" if self.mispricing_label else "")
        )


# ── Model loader ──────────────────────────────────────────────────────────────

class ModelLoader:
    """Lazy-loads the best available model from disk."""

    _cache: dict[str, object] = {}

    @classmethod
    def load_best(cls) -> Optional[object]:
        """
        Load the best model available on disk in priority order:
        XGBoost → Logistic → LSTM.
        """
        priority = [
            ("xgboost",  MODEL_DIR / "xgboost.pkl",  XGBoostModel),
            ("logistic", MODEL_DIR / "logistic.pkl",  LogisticModel),
            ("lstm",     MODEL_DIR / "lstm.pt",       LSTMModel),
        ]
        for name, path, cls_ in priority:
            if path.exists():
                if name in cls._cache:
                    return cls._cache[name]
                try:
                    if name == "lstm":
                        model = LSTMModel.load(str(path))
                    elif name == "xgboost":
                        model = XGBoostModel.load(str(path))
                    else:
                        model = LogisticModel.load(str(path))
                    cls._cache[name] = model
                    log.info(f"Loaded {name} model from {path}")
                    return model
                except Exception as exc:
                    log.warning(f"Failed to load {name} model: {exc}")

        log.debug("No trained model found — using heuristic fallback")
        return None

    @classmethod
    def load_by_name(cls, name: str) -> Optional[object]:
        paths = {
            "xgboost":  (MODEL_DIR / "xgboost.pkl",  XGBoostModel),
            "logistic": (MODEL_DIR / "logistic.pkl",  LogisticModel),
            "lstm":     (MODEL_DIR / "lstm.pt",       LSTMModel),
        }
        if name not in paths:
            return None
        path, cls_ = paths[name]
        if not path.exists():
            return None
        if name in cls._cache:
            return cls._cache[name]
        try:
            if name == "lstm":
                model = LSTMModel.load(str(path))
            elif name == "xgboost":
                model = XGBoostModel.load(str(path))
            else:
                model = LogisticModel.load(str(path))
            cls._cache[name] = model
            return model
        except Exception as exc:
            log.warning(f"Could not load {name}: {exc}")
            return None

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()


# ── Confidence estimation ─────────────────────────────────────────────────────

def _estimate_confidence(predicted: float, market_price: float, spread: float) -> float:
    """
    Confidence is a function of:
      - Edge magnitude (larger edge = more confidence the market is wrong)
      - Spread size (tight spread = more liquid = more meaningful signal)
      - Distance from 0.5 (extreme probabilities are harder to be right about)
    """
    edge        = abs(predicted - market_price)
    edge_score  = min(1.0, edge / 0.25)                  # 25c edge → full confidence from this factor
    spread_pen  = max(0.0, 1.0 - spread / 0.20)          # 20c spread → 0 confidence
    extreme_pen = 1.0 - abs(predicted - 0.5) * 0.6       # high or low prob → lower confidence
    conf = edge_score * 0.50 + spread_pen * 0.30 + extreme_pen * 0.20
    return round(max(0.0, min(1.0, conf)), 3)


def _label_mispricing(predicted: float, market_price: float, edge: float) -> str:
    """Human-readable label for the type of mispricing detected."""
    if abs(edge) < MIN_EDGE:
        return ""
    if market_price < 0.12 and predicted > 0.20:
        return "LONGSHOT_UNDERPRICED"
    if market_price > 0.88 and predicted < 0.80:
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
    Unified ML inference interface. Call predict_and_signal() at inference time.

    Priority: uses the best available trained model. Falls back to the
    heuristic strategy_ml() from strategies.py if no model is found.
    """

    def __init__(self, model_name: str = "auto"):
        self.model_name = model_name
        self._model     = None

    def _get_model(self):
        """Lazy-load underlying model."""
        if self._model is None:
            if self.model_name == "auto":
                self._model = ModelLoader.load_best()
            else:
                self._model = ModelLoader.load_by_name(self.model_name)
        return self._model

    def predict_and_signal(
        self,
        snap,
        history,
        sentiment_score: float = 0.5,
        reddit_score:    float = 0.5,
    ) -> MLSignal:
        """
        Build features, run inference, compute edge, generate trading signal.

        Args:
            snap:             MarketSnapshot (live data)
            history:          pd.DataFrame from HistoricalStore
            sentiment_score:  News sentiment [0,1]
            reddit_score:     Reddit sentiment [0,1]

        Returns:
            MLSignal with all fields populated
        """
        import pandas as pd
        market_price = snap.mid_price
        model        = self._get_model()
        model_name   = getattr(model, "NAME", "heuristic") if model else "heuristic"

        # ── Build feature vector ───────────────────────────────────────────────
        feat = build_features(
            df              = history if history is not None and not (hasattr(history, 'empty') and history.empty) else pd.DataFrame(),
            sentiment_score = sentiment_score,
            reddit_score    = reddit_score,
        )

        # ── Run inference ──────────────────────────────────────────────────────
        if feat is not None and model is not None:
            try:
                predicted_prob = float(model.predict_single(feat))
            except Exception as exc:
                log.warning(f"ML inference failed ({exc}) — using heuristic fallback")
                predicted_prob = _heuristic_prob(snap)
        else:
            predicted_prob = _heuristic_prob(snap)
            model_name     = "heuristic"

        predicted_prob = max(0.01, min(0.99, predicted_prob))

        # ── Compute edge and signal ────────────────────────────────────────────
        edge       = predicted_prob - market_price
        direction  = (
            "BUY_YES" if edge >  MIN_EDGE else
            "BUY_NO"  if edge < -MIN_EDGE else
            "HOLD"
        )
        confidence      = _estimate_confidence(predicted_prob, market_price, snap.spread)
        is_high_conf    = confidence >= HIGH_CONF_THRESHOLD
        is_strong_edge  = abs(edge) >= STRONG_EDGE_THRESHOLD
        mispricing      = _label_mispricing(predicted_prob, market_price, edge)

        signal = MLSignal(
            model_name      = model_name,
            market_price    = round(market_price, 4),
            predicted_prob  = round(predicted_prob, 4),
            edge            = round(edge, 4),
            direction       = direction,
            confidence      = confidence,
            is_high_conf    = is_high_conf,
            is_strong_edge  = is_strong_edge,
            feature_vector  = feat,
            mispricing_label = mispricing,
        )

        log.debug(signal.describe())
        return signal

    def compare_vs_market(
        self,
        snap,
        history,
        sentiment_score: float = 0.5,
        reddit_score:    float = 0.5,
    ) -> dict:
        """
        Returns a detailed comparison table of model prediction vs market price.
        Useful for understanding the source of edge.
        """
        sig = self.predict_and_signal(snap, history, sentiment_score, reddit_score)
        return {
            "question":        snap.question[:60],
            "market_price":    sig.market_price,
            "model_pred":      sig.predicted_prob,
            "edge":            sig.edge,
            "direction":       sig.direction,
            "confidence":      sig.confidence,
            "high_conf":       sig.is_high_conf,
            "strong_edge":     sig.is_strong_edge,
            "mispricing_type": sig.mispricing_label,
            "model_used":      sig.model_name,
            "volume_usd":      getattr(snap, "volume", 0),
            "spread":          getattr(snap, "spread", 0),
            "days_to_close":   getattr(snap, "days_to_close", 30),
        }

    def print_comparison(self, snap, history, sentiment_score=0.5, reddit_score=0.5):
        """Pretty-print the model vs market comparison for a single market."""
        d = self.compare_vs_market(snap, history, sentiment_score, reddit_score)
        G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; E = "\033[0m"
        edge_col = G if d["edge"] > 0 else R
        print(f"\n  {'─'*60}")
        print(f"  Market: {d['question']}")
        print(f"  {'─'*60}")
        print(f"  Market price:   {d['market_price']:.4f}")
        print(f"  Model predicts: {d['model_pred']:.4f}  (using: {d['model_used']})")
        print(f"  Edge:           {edge_col}{d['edge']:+.4f}{E}  → {d['direction']}")
        print(f"  Confidence:     {d['confidence']:.3f}  {'[HIGH CONF]' if d['high_conf'] else ''}")
        print(f"  Mispricing:     {d['mispricing_type'] or 'None'}")
        print(f"  Volume:         ${d['volume_usd']:,.0f} | Spread: {d['spread']:.4f} | Days: {d['days_to_close']}")
        print(f"  Action:         {Y}{d['direction']}{E}")
        print(f"  {'─'*60}\n")


def _heuristic_prob(snap) -> float:
    """
    Fallback heuristic probability (used when no model is trained).
    Same logic as strategy_ml() in strategies.py but self-contained.
    """
    import math
    p = snap.mid_price
    time_factor  = max(0, min(1, 1.0 - snap.days_to_close / 120))
    spread_pen   = min(0.5, snap.spread * 2)
    f_price = p
    f_time  = time_factor * p + (1 - time_factor) * 0.5
    f_spread = p * (1 - spread_pen) + 0.5 * spread_pen
    vol_adj = max(0.5, min(1.5, math.log10(max(snap.volume, 1) + 1) / 6))
    f_vol   = p * vol_adj + 0.5 * (1 - vol_adj)
    score = 0.40 * f_price + 0.20 * f_time + 0.20 * f_spread + 0.20 * f_vol
    return float(np.clip(score, 0.01, 0.99))


# Module-level singleton
predictor = MLPredictor(model_name="auto")
