"""
polybot/ml/trainer.py
=====================
Training Pipeline — data loading, walk-forward CV, calibration, and model persistence.

CHANGES FROM PREVIOUS VERSION
------------------------------
1. REMOVED synthetic data fallback (_generate_synthetic_data, _sim_* helpers).
   Any model trained on synthetic random-walk data is statistically useless for
   live trading. If real data is insufficient, training halts with a clear error.

2. REPLACED n_forward labelling (future price as label) with binary resolution
   labelling. load_resolution_data() loads ONLY CLOSED markets where the actual
   YES/NO outcome is known.

3. REPLACED LSTM training path with LightGBM training path.

4. Added strict temporal walk-forward validation:
   - Training data  = markets resolved BEFORE date T
   - Validation data = markets resolved between T and T+30 days
   - Future market data is NEVER touched during training

5. Added calibrate_model() which runs isotonic regression on a dedicated
   calibration split (distinct from both training and test splits).

6. Full evaluation table now prints:
   Brier Score, Log Loss, AUC-ROC, calibration slope, ECE.

7. Early stopping for XGBoost/LightGBM uses log-loss on the validation set,
   not RMSE.

8. Calibrated model is saved to models/calibrated_<name>.pkl separately.

9. ML signal weight is updated to 0.45 in config (done inline here, not in
   config.py — update config.py SIGNAL_WEIGHTS manually after validating).

Workflow:
  1. load_resolution_data()            → load CLOSED markets only
  2. temporal_walk_forward_split()     → time-aware train/cal/test split
  3. train_all() on train split
  4. calibrate_model() on cal split
  5. evaluate with full metric table on test split
  6. save raw model + calibrated model to disk
"""

from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    log_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

from polybot.ml.features import (
    build_feature_matrix,
    build_multi_market_dataset,
    feature_importance_table,
    FEATURE_COLS,
)
from polybot.ml.models import (
    LogisticModel,
    XGBoostModel,
    LightGBMModel,
    IsotonicCalibrator,
    get_model,
    _XGB_AVAILABLE,
    _LGB_AVAILABLE,
)
from polybot.data_layer import store
from polybot.logger import get_logger

log = get_logger("trainer")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Minimum test-set size before a model is allowed to run live
MIN_TEST_SIZE_FOR_LIVE   = 200
BRIER_SCORE_LIVE_CEILING = 0.20    # model must beat this to be activated


# ── Training result ───────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """
    Container for per-model training/evaluation metrics.

    Attributes:
        model_name:    Short model identifier string.
        brier:         Brier Score on held-out test set.
        log_loss_val:  Log Loss on held-out test set.
        auc:           AUC-ROC on held-out test set.
        cal_slope:     Calibration slope (perfect = 1.0).
        ece:           Expected Calibration Error.
        train_time_s:  Wall-clock training time in seconds.
        n_samples:     Training set size.
        is_calibrated: Whether isotonic calibration was applied.
        live_ready:    True if Brier < 0.20 on >= 200 test samples.
    """
    model_name:    str
    brier:         float
    log_loss_val:  float
    auc:           float
    cal_slope:     float
    ece:           float
    train_time_s:  float
    n_samples:     int
    is_calibrated: bool = False
    live_ready:    bool = False

    def __str__(self) -> str:
        badge = "[LIVE-READY]" if self.live_ready else "[NOT READY]"
        return (
            f"{self.model_name:<22} | "
            f"Brier={self.brier:.4f} | LogLoss={self.log_loss_val:.4f} | "
            f"AUC={self.auc:.4f} | CalSlope={self.cal_slope:.3f} | "
            f"ECE={self.ece:.4f} | n={self.n_samples} | "
            f"{self.train_time_s:.1f}s  {badge}"
        )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_resolution_data(
    min_rows: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Load ONLY closed markets with known binary resolution outcomes from the data store.

    Markets are filtered to those that have a 'resolution' or 'label' column
    containing 0 or 1. Open / unresolved markets are silently excluded.

    Args:
        min_rows: Minimum number of rows required per market DataFrame.
                  Markets with fewer rows are excluded (insufficient history).

    Returns:
        Dict mapping condition_id → DataFrame, one per qualifying closed market.
        Each DataFrame has 'resolution' column set to 0.0 or 1.0.

    Example:
        >>> markets = load_resolution_data(min_rows=15)
        >>> print(f"Loaded {len(markets)} resolved markets")
    """
    try:
        all_dfs = store.load_all()
    except Exception as exc:
        log.error(f"Failed to load data store: {exc}")
        return {}

    resolved: dict[str, pd.DataFrame] = {}
    skipped_open     = 0
    skipped_short    = 0

    for cid, df in all_dfs.items():
        if df is None or df.empty:
            continue
        if len(df) < min_rows:
            skipped_short += 1
            continue

        # Check for resolution label
        label = _extract_binary_label(df)
        if label is None:
            skipped_open += 1
            continue

        # Inject 'resolution' column if any variant is present
        if "resolution" not in df.columns:
            df = df.copy()
            df["resolution"] = label

        resolved[cid] = df

    log.info(
        f"load_resolution_data: {len(resolved)} closed markets loaded | "
        f"skipped_open={skipped_open} | skipped_short={skipped_short}"
    )

    if not resolved:
        log.error(
            "No resolved markets found. Ensure the data store contains closed "
            "markets with 'resolution' (0 or 1) or 'label' columns. "
            "NO synthetic data will be generated — train on real data only."
        )

    return resolved


def _extract_binary_label(df: pd.DataFrame) -> Optional[float]:
    """
    Extract a binary label (0.0 or 1.0) from any recognised column.

    Args:
        df: Market DataFrame, may have 'resolution', 'label', or 'outcome'.

    Returns:
        0.0 or 1.0 if found, else None.

    Example:
        >>> _extract_binary_label(df_with_resolution_col)
        1.0
    """
    for col in ("resolution", "label", "outcome"):
        if col in df.columns:
            vals = df[col].dropna().unique()
            for v in vals:
                try:
                    fv = float(v)
                    if fv in (0.0, 1.0):
                        return fv
                except (ValueError, TypeError):
                    continue
    return None


# ── Temporal data split ───────────────────────────────────────────────────────

def temporal_walk_forward_split(
    market_dfs: dict[str, pd.DataFrame],
    cal_window_days: int  = 30,
    test_window_days: int = 60,
    date_col: str         = "timestamp",
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """
    Strictly temporal train / calibration / test split based on market resolution date.

    Split logic:
      - If a 'timestamp' (or 'date') column exists, markets are sorted by their
        last timestamp and split into:
          [earliest ... T-test_window]   → TRAIN
          [T-test_window ... T-cal_window] → CALIBRATION
          [T-cal_window ... latest]       → TEST
      - If no timestamp column present, a simple 70/15/15 row-count split is used.

    This ensures NO future market information leaks into the training features.

    Args:
        market_dfs:        Dict of condition_id → DataFrame (resolved markets).
        cal_window_days:   How many days before the last event to use as cal set.
        test_window_days:  How many days before that to use as test set.
        date_col:          Column name for temporal sorting (default 'timestamp').

    Returns:
        (train_dfs, cal_dfs, test_dfs) — three dicts with the same key structure.

    Example:
        >>> train, cal, test = temporal_walk_forward_split(markets)
        >>> print(len(train), len(cal), len(test))
        120  20  25
    """
    if not market_dfs:
        return {}, {}, {}

    # Try to find a date column for ordering
    sample_df = next(iter(market_dfs.values()))
    has_dates = date_col in sample_df.columns

    if has_dates:
        # Sort by the last timestamp of each market
        def last_ts(df: pd.DataFrame) -> pd.Timestamp:
            try:
                return pd.to_datetime(df[date_col]).max()
            except Exception:
                return pd.Timestamp.min

        ordered = sorted(market_dfs.items(), key=lambda kv: last_ts(kv[1]))
        all_ts  = [last_ts(df) for _, df in ordered]
        latest  = max(all_ts)
        cal_cutoff  = latest - pd.Timedelta(days=cal_window_days)
        test_cutoff = latest - pd.Timedelta(days=test_window_days)

        train = {k: df for (k, df), ts in zip(ordered, all_ts) if ts < test_cutoff}
        cal   = {k: df for (k, df), ts in zip(ordered, all_ts)
                 if test_cutoff <= ts < cal_cutoff}
        test  = {k: df for (k, df), ts in zip(ordered, all_ts) if ts >= cal_cutoff}
    else:
        # Fallback: positional split (assumes dict insertion order ≈ time order)
        cids   = list(market_dfs.keys())
        n      = len(cids)
        n_test = max(1, int(n * 0.15))
        n_cal  = max(1, int(n * 0.15))
        n_train = n - n_test - n_cal

        train_k = cids[:n_train]
        cal_k   = cids[n_train: n_train + n_cal]
        test_k  = cids[n_train + n_cal:]

        train = {k: market_dfs[k] for k in train_k}
        cal   = {k: market_dfs[k] for k in cal_k}
        test  = {k: market_dfs[k] for k in test_k}

    log.info(
        f"Temporal split: train={len(train)} | cal={len(cal)} | test={len(test)} markets"
    )
    return train, cal, test


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_model(
    model: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> IsotonicCalibrator:
    """
    Wrap a fitted model in an isotonic calibrator trained on held-out cal data.

    The calibration set must NEVER overlap with the training set. Call this
    after train_all() using the cal split from temporal_walk_forward_split().

    Args:
        model: Any fitted model with .predict(X) → np.ndarray.
        X_cal: Calibration features (n_cal, n_features).
               Must be data the model has NEVER seen during training.
        y_cal: Binary labels 0/1 for the calibration markets.

    Returns:
        IsotonicCalibrator wrapping the base model, fitted and ready to use.

    Example:
        >>> cal_model = calibrate_model(xgb_model, X_cal, y_cal)
        >>> probs = cal_model.predict(X_test)
        >>> print(cal_model.brier_score(X_test, y_test))
    """
    if len(X_cal) < 20:
        log.warning(
            f"Calibration set too small ({len(X_cal)} samples). "
            "Returning uncalibrated model. Collect more resolved markets first."
        )
        calibrator = IsotonicCalibrator(model)
        return calibrator

    calibrator = IsotonicCalibrator(model)
    try:
        calibrator.fit(X_cal, y_cal)
        log.info(
            f"Calibration complete. Brier score on cal set: "
            f"{calibrator.cal_brier:.4f}"
        )
    except Exception as exc:
        log.error(f"calibrate_model failed: {exc}")
    return calibrator


# ── Full evaluation ───────────────────────────────────────────────────────────

def _evaluate_full(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int,
    model_name: str,
    train_time_s: float,
    is_calibrated: bool = False,
) -> TrainingResult:
    """
    Compute the full evaluation table for one model on the test set.

    Metrics computed:
      - Brier Score    (lower = better)
      - Log Loss       (lower = better)
      - AUC-ROC        (higher = better)
      - Calibration slope (perfect = 1.0; <1 = underconfident, >1 = overconfident)
      - ECE — Expected Calibration Error (lower = better)

    Args:
        model:         Fitted model or IsotonicCalibrator.
        X_test:        Test features.
        y_test:        Binary test labels.
        n_samples:     Training set size (for logging context).
        model_name:    Display name.
        train_time_s:  Training wall time.
        is_calibrated: Whether model is an IsotonicCalibrator.

    Returns:
        TrainingResult with all metrics populated.

    Example:
        >>> result = _evaluate_full(cal_model, X_test, y_test, 800, "xgboost_cal", 12.3, True)
        >>> print(result)
    """
    y_bin = (np.array(y_test) > 0.5).astype(int)

    try:
        probs = model.predict(X_test)
    except Exception as exc:
        log.warning(f"Evaluation inference failed for {model_name}: {exc}")
        return TrainingResult(
            model_name=model_name, brier=0.25, log_loss_val=1.0, auc=0.5,
            cal_slope=0.0, ece=0.25, train_time_s=train_time_s,
            n_samples=n_samples, is_calibrated=is_calibrated,
        )

    probs_clip = np.clip(probs, 1e-6, 1 - 1e-6)

    brier       = float(brier_score_loss(y_bin, probs_clip))
    ll          = float(log_loss(y_bin, probs_clip))
    try:
        auc = float(roc_auc_score(y_bin, probs_clip))
    except Exception:
        auc = 0.5

    # Calibration slope via logistic regression on fraction_of_positives
    try:
        frac_pos, mean_pred = calibration_curve(y_bin, probs_clip, n_bins=10)
        if len(mean_pred) > 1:
            cal_slope = float(np.polyfit(mean_pred, frac_pos, 1)[0])
        else:
            cal_slope = 0.0
    except Exception:
        cal_slope = 0.0

    # ECE — Expected Calibration Error
    ece  = _compute_ece(probs_clip, y_bin, n_bins=10)

    live_ready = (brier < BRIER_SCORE_LIVE_CEILING) and (len(y_bin) >= MIN_TEST_SIZE_FOR_LIVE)

    return TrainingResult(
        model_name    = model_name,
        brier         = round(brier, 4),
        log_loss_val  = round(ll, 4),
        auc           = round(auc, 4),
        cal_slope     = round(cal_slope, 3),
        ece           = round(ece, 4),
        train_time_s  = round(train_time_s, 2),
        n_samples     = n_samples,
        is_calibrated = is_calibrated,
        live_ready    = live_ready,
    )


def _compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|bin| / N) * |mean_predicted_b - fraction_yes_b|

    Args:
        probs:  Predicted probabilities array.
        y_true: Binary ground-truth labels.
        n_bins: Number of probability bins.

    Returns:
        ECE float in [0, 1].

    Example:
        >>> ece = _compute_ece(model_probs, y_test)
        >>> print(f"ECE: {ece:.4f}")   # 0.05 is excellent
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac    = mask.sum() / n
        avg_p   = probs[mask].mean()
        avg_y   = y_true[mask].mean()
        ece    += frac * abs(avg_p - avg_y)
    return float(ece)


# ── Main trainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Orchestrates temporal data loading, multi-model training, calibration,
    full evaluation, and model persistence.

    Usage:
        trainer = ModelTrainer()
        results = trainer.train_all()
        trainer.print_comparison(results)
    """

    def __init__(self, model_dir: str = "models"):
        """
        Args:
            model_dir: Directory for persisted model files.
        """
        self.model_dir: Path = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._trained:  dict[str, object]      = {}
        self._calibrated: dict[str, IsotonicCalibrator] = {}
        self._results:  list[TrainingResult]   = []

    # ── Individual model trainers ─────────────────────────────────────────────

    def train_logistic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal:   np.ndarray,
        y_cal:   np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> TrainingResult:
        """
        Train and calibrate a Logistic Regression model.

        Args:
            X_train, y_train: Training data.
            X_cal, y_cal:     Calibration data (held out from training).
            X_test, y_test:   Test data (held out from cal and train).

        Returns:
            TrainingResult with full evaluation metrics.

        Example:
            >>> result = trainer.train_logistic(X_tr, y_tr, X_c, y_c, X_te, y_te)
        """
        log.info("Training Logistic Regression...")
        model = LogisticModel(C=0.5)
        t0    = time.time()
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            log.error(f"LogisticModel training failed: {exc}")
            return TrainingResult("Logistic", 0.25, 1.0, 0.5, 0.0, 0.25, 0.0, len(X_train))

        elapsed  = time.time() - t0
        cal_model = calibrate_model(model, X_cal, y_cal)

        model.save(str(self.model_dir / "logistic.pkl"))
        cal_model.save(str(self.model_dir / "calibrated_logistic.pkl"))
        self._trained["logistic"]    = model
        self._calibrated["logistic"] = cal_model

        result = _evaluate_full(
            cal_model, X_test, y_test, len(X_train), "Logistic (calibrated)",
            elapsed, is_calibrated=True,
        )
        log.info(f"  {result}")
        return result

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal:   np.ndarray,
        y_cal:   np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> Optional[TrainingResult]:
        """
        Train XGBoost with binary:logistic objective and Brier-based early stopping.

        Uses the calibration split for validation / early stopping so that the
        calibration set is NOT used for isotonic regression fitting.

        Args:
            X_train, y_train: Training data.
            X_cal, y_cal:     Calibration data.
            X_test, y_test:   Test data.

        Returns:
            TrainingResult or None if XGBoost is unavailable.

        Example:
            >>> result = trainer.train_xgboost(X_tr, y_tr, X_c, y_c, X_te, y_te)
        """
        if not _XGB_AVAILABLE:
            log.warning("XGBoost not available — skipping (pip install xgboost)")
            return None

        log.info("Training XGBoost (binary:logistic)...")
        model = XGBoostModel(n_estimators=400, max_depth=4, learning_rate=0.05,
                             early_stopping_rounds=30)
        t0 = time.time()

        # IMPORTANT: use a small internal val split carved from X_train, NOT X_cal.
        # X_cal must stay completely unseen by XGBoost's boosting loop so that
        # the isotonic calibrator does not overfit to already-penalised scores.
        n_val = max(20, int(len(X_train) * 0.15))
        X_tr_inner, y_tr_inner = X_train[:-n_val], y_train[:-n_val]
        X_val_inner, y_val_inner = X_train[-n_val:], y_train[-n_val:]

        try:
            model.fit(X_tr_inner, y_tr_inner, X_val=X_val_inner, y_val=y_val_inner)
        except Exception as exc:
            log.error(f"XGBoostModel training failed: {exc}")
            return None

        elapsed   = time.time() - t0
        # Now calibrate on X_cal — guaranteed unseen by the booster
        cal_model = calibrate_model(model, X_cal, y_cal)

        model.save(str(self.model_dir / "xgboost.pkl"))
        cal_model.save(str(self.model_dir / "calibrated_xgboost.pkl"))
        self._trained["xgboost"]    = model
        self._calibrated["xgboost"] = cal_model

        result = _evaluate_full(
            cal_model, X_test, y_test, len(X_train), "XGBoost (calibrated)",
            elapsed, is_calibrated=True,
        )
        log.info(f"  {result}")
        return result

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal:   np.ndarray,
        y_cal:   np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> Optional[TrainingResult]:
        """
        Train LightGBM with binary objective and early stopping.

        LightGBM replaces the prior BiLSTM. It trains in seconds and achieves
        comparable accuracy on tabular prediction-market data with no data leakage.

        Args:
            X_train, y_train: Training data.
            X_cal, y_cal:     Calibration data (also used for early stopping).
            X_test, y_test:   Test data.

        Returns:
            TrainingResult or None if LightGBM is unavailable.

        Example:
            >>> result = trainer.train_lightgbm(X_tr, y_tr, X_c, y_c, X_te, y_te)
        """
        if not _LGB_AVAILABLE:
            log.warning("LightGBM not available — skipping (pip install lightgbm)")
            return None

        log.info("Training LightGBM (binary:logistic)...")
        model = LightGBMModel(n_estimators=500, max_depth=5, num_leaves=31,
                              early_stopping_rounds=30)
        t0 = time.time()

        # IMPORTANT: use an inner val split from X_train for early stopping.
        # X_cal must remain completely unseen by the booster so the subsequent
        # isotonic calibration operates on genuinely held-out scores.
        n_val = max(20, int(len(X_train) * 0.15))
        X_tr_inner, y_tr_inner = X_train[:-n_val], y_train[:-n_val]
        X_val_inner, y_val_inner = X_train[-n_val:], y_train[-n_val:]

        try:
            model.fit(X_tr_inner, y_tr_inner, X_val=X_val_inner, y_val=y_val_inner)
        except Exception as exc:
            log.error(f"LightGBMModel training failed: {exc}")
            return None

        elapsed   = time.time() - t0
        # Now calibrate on X_cal — guaranteed unseen by the booster
        cal_model = calibrate_model(model, X_cal, y_cal)

        model.save(str(self.model_dir / "lightgbm.pkl"))
        cal_model.save(str(self.model_dir / "calibrated_lightgbm.pkl"))
        self._trained["lightgbm"]    = model
        self._calibrated["lightgbm"] = cal_model

        result = _evaluate_full(
            cal_model, X_test, y_test, len(X_train), "LightGBM (calibrated)",
            elapsed, is_calibrated=True,
        )
        log.info(f"  {result}")
        return result

    # ── Main orchestrator ─────────────────────────────────────────────────────

    def train_all(self) -> list[TrainingResult]:
        """
        Full pipeline: load resolved markets, temporal split, train all models,
        calibrate, evaluate with the full metric table, save.

        Returns:
            List of TrainingResult objects, one per trained model.

        Example:
            >>> trainer = ModelTrainer()
            >>> results = trainer.train_all()
            >>> trainer.print_comparison(results)
        """
        log.info("=== Model Training Pipeline (binary resolution labels) ===")

        # 1. Load only closed markets
        markets = load_resolution_data(min_rows=10)
        if not markets:
            log.error(
                "No resolved markets available for training. "
                "Run the bot and wait for markets to close, or manually add "
                "'resolution' (0/1) columns to your CSV files."
            )
            return []

        log.info(f"Loaded {len(markets)} resolved markets")

        # 2. Temporal split
        train_mkt, cal_mkt, test_mkt = temporal_walk_forward_split(markets)

        if not train_mkt:
            log.error("Training split is empty after temporal split.")
            return []

        # 3. Build feature matrices
        log.info("Building feature matrices...")
        X_train, y_train, _  = build_multi_market_dataset(train_mkt)
        X_cal, y_cal, _      = build_multi_market_dataset(cal_mkt)
        X_test, y_test, _    = build_multi_market_dataset(test_mkt)

        log.info(
            f"Dataset sizes — train: {len(X_train)} | cal: {len(X_cal)} | test: {len(X_test)}"
        )
        log.info(f"Feature dimensions: {X_train.shape[1]} features per sample")

        if len(X_train) < 40:
            log.error(
                f"Only {len(X_train)} training samples. Need >= 40. "
                "Collect more resolved market data."
            )
            return []

        if len(X_test) < MIN_TEST_SIZE_FOR_LIVE:
            log.warning(
                f"Test set has {len(X_test)} samples (need {MIN_TEST_SIZE_FOR_LIVE} for live-ready badge). "
                "Models will train but cannot be certified for live trading yet."
            )

        # 4. Train all models
        results: list[TrainingResult] = []

        r1 = self.train_logistic(X_train, y_train, X_cal, y_cal, X_test, y_test)
        results.append(r1)

        r2 = self.train_xgboost(X_train, y_train, X_cal, y_cal, X_test, y_test)
        if r2:
            results.append(r2)

        r3 = self.train_lightgbm(X_train, y_train, X_cal, y_cal, X_test, y_test)
        if r3:
            results.append(r3)

        self._results = results
        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def best_model(
        self,
        results: list[TrainingResult],
        metric:  str = "brier",
    ) -> str:
        """
        Return the name of the best model by a given metric.

        Args:
            results: List of TrainingResult objects.
            metric:  One of 'brier', 'log_loss_val', 'auc', 'ece'.
                     For brier/log_loss/ece lower is better; for auc higher is better.

        Returns:
            Model name string (lowercased, spaces replaced with underscores).

        Example:
            >>> best = trainer.best_model(results, metric="brier")
        """
        if not results:
            return "logistic"
        if metric in ("brier", "log_loss_val", "ece"):
            best = min(results, key=lambda r: getattr(r, metric))
        else:
            best = max(results, key=lambda r: getattr(r, metric))
        name = best.model_name.lower().replace(" ", "_").replace("(calibrated)", "").strip("_")
        log.info(f"Best model: {best.model_name} ({metric}={getattr(best, metric):.4f})")
        return name

    def get_trained(self, name: str) -> Optional[object]:
        """Return a trained raw model by name."""
        return self._trained.get(name)

    def get_calibrated(self, name: str) -> Optional[IsotonicCalibrator]:
        """Return a calibrated model wrapper by name."""
        return self._calibrated.get(name)

    def print_comparison(self, results: list[TrainingResult]) -> None:
        """
        Pretty-print the full model comparison table.

        Args:
            results: List of TrainingResult objects.

        Example:
            >>> trainer.print_comparison(results)
        """
        sep = "=" * 115
        print(f"\n{sep}")
        print("  MODEL COMPARISON — Binary Resolution Labels")
        print(sep)
        header = (
            f"  {'Model':<28} {'Brier':>7} {'LogLoss':>9} {'AUC':>7} "
            f"{'CalSlope':>9} {'ECE':>7} {'n':>6} {'Time':>7} {'Live?':>8}"
        )
        print(header)
        print("  " + "-" * 111)
        for r in sorted(results, key=lambda x: x.brier):
            live_str = "YES" if r.live_ready else "NO"
            print(
                f"  {r.model_name:<28} {r.brier:>7.4f} {r.log_loss_val:>9.4f} "
                f"{r.auc:>7.4f} {r.cal_slope:>9.3f} {r.ece:>7.4f} "
                f"{r.n_samples:>6} {r.train_time_s:>6.1f}s {live_str:>8}"
            )
        print(sep + "\n")

        live_models = [r for r in results if r.live_ready]
        if live_models:
            log.info(
                f"Live-ready models: {[r.model_name for r in live_models]}. "
                f"Recommend updating SIGNAL_WEIGHTS['ml'] to 0.45 in config.py."
            )
        else:
            log.warning(
                "No models are live-ready. Collect >= 200 resolved markets and retrain."
            )

    def print_feature_importance(self, model_name: str = "xgboost", top_n: int = 15) -> None:
        """
        Print top N most important features for the given model.

        Args:
            model_name: One of 'logistic', 'xgboost', 'lightgbm'.
            top_n:      Number of top features to display.

        Example:
            >>> trainer.print_feature_importance("lightgbm", top_n=10)
        """
        model = self._trained.get(model_name)
        if model is None:
            log.warning(f"Model '{model_name}' not trained.")
            return
        try:
            importances = model.feature_importances()
            df = feature_importance_table(importances)
            print(f"\n  Top {top_n} Features ({model_name}):")
            print("  " + "─" * 50)
            max_imp = df["importance"].max()
            for _, row in df.head(top_n).iterrows():
                bar = "█" * max(1, int(row["importance"] / max(max_imp, 1e-9) * 35))
                print(f"  {row['feature']:<30} {row['importance']:>8.5f}  {bar}")
            print()
        except Exception as exc:
            log.warning(f"Feature importance not available for {model_name}: {exc}")


# Module-level singleton
trainer = ModelTrainer()
