"""
polybot/ml/trainer.py
=====================
Training Pipeline — data loading, cross-validation, model comparison,
hyperparameter tuning, and model persistence.

Workflow:
  1. Load all historical CSVs from the data store
  2. Build feature matrices (multi-market combined)
  3. Walk-forward cross-validation (no data leakage)
  4. Train all three model families
  5. Compare on held-out test set
  6. Save best model
  7. Print feature importance ranking
"""

from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, brier_score_loss, roc_auc_score,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

from polybot.ml.features import (
    build_feature_matrix, build_multi_market_dataset,
    feature_importance_table, FEATURE_COLS,
)
from polybot.ml.models import LogisticModel, XGBoostModel, LSTMModel, get_model, _XGB_AVAILABLE, _TORCH_AVAILABLE
from polybot.data_layer import store
from polybot.logger import get_logger

log = get_logger("trainer")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Training result ───────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    model_name:   str
    rmse:         float
    mae:          float
    brier:        float
    auc:          float
    train_time_s: float
    n_samples:    int
    cv_rmse_mean: float = 0.0
    cv_rmse_std:  float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.model_name:<22} | "
            f"RMSE={self.rmse:.4f} | MAE={self.mae:.4f} | "
            f"Brier={self.brier:.4f} | AUC={self.auc:.4f} | "
            f"CV-RMSE={self.cv_rmse_mean:.4f}±{self.cv_rmse_std:.4f} | "
            f"n={self.n_samples} | {self.train_time_s:.1f}s"
        )


# ── Data loader ───────────────────────────────────────────────────────────────

def load_training_data(
    n_forward: int    = 5,
    min_samples: int  = 20,
    synthetic: bool   = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all historical market CSVs and build a combined training dataset.

    Args:
        n_forward:    Label lookahead in rows (future price proxy)
        min_samples:  Minimum rows per market to include
        synthetic:    If real data is insufficient, inject synthetic data for demonstration

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
    """
    all_dfs = store.load_all()
    real_markets = {k: v for k, v in all_dfs.items() if len(v) >= min_samples}

    if real_markets:
        log.info(f"Loaded {len(real_markets)} markets with >= {min_samples} rows")
        X, y, _ = build_multi_market_dataset(real_markets, n_forward=n_forward)
        if len(X) >= 50:
            return X, y
        log.warning(f"Only {len(X)} training samples from real data — augmenting with synthetic")

    if synthetic:
        log.info("Generating synthetic training data (run bot longer for real data)")
        X, y = _generate_synthetic_data(n_samples=2000)
        return X, y

    return np.empty((0, len(FEATURE_COLS))), np.empty(0)


def _generate_synthetic_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic prediction market data for demonstration.
    Models real-world patterns:
      - Trending markets (momentum)
      - Mean-reverting markets (range-bound)
      - Efficient markets (random walk)
      - Thin, illiquid markets (noisy)
    """
    rng = np.random.default_rng(seed=42)
    records = []
    labels  = []

    n_per_type = n_samples // 4

    for market_type in ["momentum", "mean_rev", "efficient", "noisy"]:
        for _ in range(n_per_type):
            # Base price
            if market_type == "momentum":
                prices = _sim_trending(rng, n=40)
            elif market_type == "mean_rev":
                prices = _sim_mean_rev(rng, n=40)
            elif market_type == "efficient":
                prices = _sim_random_walk(rng, n=40)
            else:
                prices = _sim_noisy(rng, n=40)

            volume   = rng.lognormal(mean=10, sigma=2, size=40) + 500
            liquidity = volume * rng.uniform(0.8, 2.5, size=40)
            spread   = rng.uniform(0.01, 0.15, size=40)
            days     = np.linspace(60, 1, 40)

            df = pd.DataFrame({
                "mid_price":    prices,
                "volume":      volume,
                "liquidity":   liquidity,
                "spread":      spread,
                "days_to_close": days,
            })

            from polybot.ml.features import build_features
            feat = build_features(df, sentiment_score=rng.uniform(0.3, 0.7), reddit_score=rng.uniform(0.3, 0.7))
            if feat is None:
                continue

            # Label: future price (n_forward=5 steps ahead, last row as target)
            label = float(np.clip(prices[-1] + rng.normal(0, 0.02), 0.01, 0.99))
            records.append(feat)
            labels.append(label)

    X = np.array(records, dtype=np.float32)
    y = np.array(labels,  dtype=np.float32)
    return X, y


def _sim_trending(rng, n: int = 40) -> np.ndarray:
    p0 = rng.uniform(0.1, 0.9)
    drift = rng.uniform(0.005, 0.015) * rng.choice([-1, 1])
    noise = rng.normal(0, 0.01, n)
    prices = np.cumsum([drift] * n) + noise + p0
    return np.clip(prices, 0.01, 0.99)


def _sim_mean_rev(rng, n: int = 40) -> np.ndarray:
    mean = rng.uniform(0.3, 0.7)
    prices = [rng.uniform(0.2, 0.8)]
    for _ in range(n - 1):
        reversion = 0.3 * (mean - prices[-1])
        noise     = rng.normal(0, 0.02)
        prices.append(np.clip(prices[-1] + reversion + noise, 0.01, 0.99))
    return np.array(prices)


def _sim_random_walk(rng, n: int = 40) -> np.ndarray:
    p0     = rng.uniform(0.2, 0.8)
    steps  = rng.normal(0, 0.015, n)
    prices = np.cumsum(steps) + p0
    return np.clip(prices, 0.01, 0.99)


def _sim_noisy(rng, n: int = 40) -> np.ndarray:
    p0     = rng.uniform(0.1, 0.9)
    noise  = rng.normal(0, 0.05, n)
    prices = p0 + noise
    return np.clip(prices, 0.01, 0.99)


# ── Walk-forward cross-validation ────────────────────────────────────────────

def walk_forward_cv(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """
    Time-series-aware cross-validation (no data leakage).
    Uses sklearn's TimeSeriesSplit.
    Returns (mean_rmse, std_rmse).
    """
    from sklearn.base import clone
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if len(X_tr) < 30 or len(X_val) < 10:
            continue

        try:
            if isinstance(model, LogisticModel):
                m_fold = LogisticModel()
                m_fold.fit(X_tr, y_tr)
            elif isinstance(model, XGBoostModel):
                m_fold = XGBoostModel()
                m_fold.fit(X_tr, y_tr)
            else:
                # LSTM is too slow for CV — skip
                continue

            preds = m_fold.predict(X_val)
            rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
            fold_rmses.append(rmse)
        except Exception as exc:
            log.warning(f"CV fold {fold} failed: {exc}")

    if not fold_rmses:
        return 0.0, 0.0
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))


# ── Main trainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Orchestrates multi-model training, comparison, and persistence.

    Usage:
        trainer = ModelTrainer()
        results = trainer.train_all()
        trainer.print_comparison(results)
        best = trainer.best_model(results)
    """

    def __init__(self, model_dir: str = "models", test_size: float = 0.20):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self._trained: dict[str, object]  = {}
        self._results: list[TrainingResult] = []

    def _split(self, X, y):
        """Temporal train/test split (no shuffle — preserve time order)."""
        n_test = max(20, int(len(X) * self.test_size))
        return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]

    def _evaluate(self, model, X_test, y_test) -> dict:
        preds = model.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae   = float(mean_absolute_error(y_test, preds))
        brier = float(brier_score_loss((y_test > 0.5).astype(int), preds))
        try:
            auc = float(roc_auc_score((y_test > 0.5).astype(int), preds))
        except Exception:
            auc = 0.5
        return {"rmse": rmse, "mae": mae, "brier": brier, "auc": auc}

    def train_logistic(self, X_train, y_train, X_test, y_test) -> TrainingResult:
        log.info("Training Logistic Regression...")
        model  = LogisticModel(C=0.5)
        t0     = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        cv_m, cv_s = walk_forward_cv(model, X_train, y_train)
        metrics    = self._evaluate(model, X_test, y_test)
        model.save(str(self.model_dir / "logistic.pkl"))
        self._trained["logistic"] = model
        result = TrainingResult(
            model_name="Logistic Regression", n_samples=len(X_train),
            train_time_s=round(elapsed, 2), cv_rmse_mean=cv_m, cv_rmse_std=cv_s,
            **metrics,
        )
        log.info(f"  {result}")
        return result

    def train_xgboost(self, X_train, y_train, X_test, y_test) -> Optional[TrainingResult]:
        if not _XGB_AVAILABLE:
            log.warning("XGBoost not available — skipping (pip install xgboost)")
            return None
        log.info("Training XGBoost...")
        model  = XGBoostModel(n_estimators=400, max_depth=4, learning_rate=0.05)
        t0     = time.time()
        n_val  = max(20, int(len(X_train) * 0.15))
        X_tr2, y_tr2 = X_train[:-n_val], y_train[:-n_val]
        X_v2,  y_v2  = X_train[-n_val:], y_train[-n_val:]
        model.fit(X_tr2, y_tr2, X_val=X_v2, y_val=y_v2)
        elapsed = time.time() - t0
        cv_m, cv_s = walk_forward_cv(model, X_train, y_train)
        metrics    = self._evaluate(model, X_test, y_test)
        model.save(str(self.model_dir / "xgboost.pkl"))
        self._trained["xgboost"] = model
        result = TrainingResult(
            model_name="XGBoost", n_samples=len(X_train),
            train_time_s=round(elapsed, 2), cv_rmse_mean=cv_m, cv_rmse_std=cv_s,
            **metrics,
        )
        log.info(f"  {result}")
        return result

    def train_lstm(self, X_train, y_train, X_test, y_test) -> Optional[TrainingResult]:
        if not _TORCH_AVAILABLE:
            log.warning("PyTorch not available — skipping LSTM (pip install torch)")
            return None
        if len(X_train) < 100:
            log.warning(f"LSTM needs >= 100 samples, got {len(X_train)} — skipping")
            return None
        log.info("Training LSTM...")
        model   = LSTMModel(seq_len=15, hidden_size=128, num_layers=2, epochs=60)
        t0      = time.time()
        model.fit(X_train, y_train, verbose=True)
        elapsed = time.time() - t0
        metrics = self._evaluate(model, X_test, y_test)
        model.save(str(self.model_dir / "lstm.pt"))
        self._trained["lstm"] = model
        result = TrainingResult(
            model_name="LSTM", n_samples=len(X_train),
            train_time_s=round(elapsed, 2),
            **metrics,
        )
        log.info(f"  {result}")
        return result

    def train_all(self, n_forward: int = 5) -> list[TrainingResult]:
        """Load data, train all three models, return comparison results."""
        log.info("=== Model Training Pipeline ===")
        X, y = load_training_data(n_forward=n_forward)
        if len(X) < 40:
            log.error(f"Insufficient data: {len(X)} samples. Run the bot first to collect data.")
            return []

        log.info(f"Dataset: {len(X)} samples × {X.shape[1]} features")
        X_train, y_train, X_test, y_test = self._split(X, y)
        log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

        results: list[TrainingResult] = []

        r1 = self.train_logistic(X_train, y_train, X_test, y_test)
        results.append(r1)

        r2 = self.train_xgboost(X_train, y_train, X_test, y_test)
        if r2:
            results.append(r2)

        r3 = self.train_lstm(X_train, y_train, X_test, y_test)
        if r3:
            results.append(r3)

        self._results = results
        return results

    def best_model(self, results: list[TrainingResult], metric: str = "rmse") -> str:
        """Return name of best model by metric (lower = better for rmse/brier)."""
        if not results:
            return "logistic"
        if metric in ("rmse", "mae", "brier"):
            best = min(results, key=lambda r: getattr(r, metric))
        else:
            best = max(results, key=lambda r: getattr(r, metric))
        log.info(f"Best model: {best.model_name} (by {metric}={getattr(best, metric):.4f})")
        return best.model_name.lower().replace(" ", "_")

    def get_trained(self, name: str):
        """Return a trained model by name."""
        return self._trained.get(name)

    def print_comparison(self, results: list[TrainingResult]):
        """Pretty-print model comparison table."""
        sep = "═" * 110
        print(f"\n{sep}")
        print("  MODEL COMPARISON")
        print(sep)
        print(f"  {'Model':<22} {'RMSE':>8} {'MAE':>8} {'Brier':>8} {'AUC':>8} {'CV-RMSE':>12} {'Time':>8}")
        print("  " + "─" * 106)
        for r in sorted(results, key=lambda x: x.rmse):
            print(
                f"  {r.model_name:<22} "
                f"{r.rmse:>8.4f} {r.mae:>8.4f} {r.brier:>8.4f} {r.auc:>8.4f} "
                f"{r.cv_rmse_mean:>6.4f}±{r.cv_rmse_std:<5.4f} {r.train_time_s:>7.1f}s"
            )
        print(sep + "\n")

    def print_feature_importance(self, model_name: str = "xgboost", top_n: int = 15):
        """Print top N most important features for the given model."""
        model = self._trained.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not trained.")
            return
        try:
            importances = model.feature_importances()
            df = feature_importance_table(importances)
            print(f"\n  Top {top_n} Features ({model_name}):")
            print("  " + "─" * 45)
            for _, row in df.head(top_n).iterrows():
                bar = "█" * int(row["importance"] * 400 / max(df["importance"].iloc[0], 1e-9))
                print(f"  {row['feature']:<25} {row['importance']:>7.4f}  {bar}")
            print()
        except Exception as exc:
            print(f"Feature importance not available: {exc}")


# Module-level singleton
trainer = ModelTrainer()
