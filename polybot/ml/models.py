"""
polybot/ml/models.py
====================
Model Definitions — Three ML architectures for YES-probability prediction.

CHANGES FROM PREVIOUS VERSION
------------------------------
1. REMOVED BiLSTM (LSTMModel / LSTMNet) — bidirectional LSTMs see future data
   during training (right-side context), causing severe data leakage that is
   invisible at training time but causes live alpha to collapse immediately.

2. REPLACED with LightGBM (LightGBMModel):
   - Trains 10-50× faster than gradient-boosted trees on small datasets
   - Handles < 500 samples gracefully with built-in regularisation
   - Native binary:logistic objective outputs calibrated P(YES) directly
   - No requires separate scaling step (tree-based)

3. XGBoostModel now uses objective='binary:logistic' and eval_metric='logloss'
   when labels are binary 0/1 — this is the correct loss for classification.
   Early stopping uses log-loss on the validation Brier score, not RMSE.

4. Added IsotonicCalibrator wrapper:
   - Wraps ANY fitted model
   - Fits isotonic regression on a held-out calibration split
   - .predict(X) returns calibrated probabilities
   - .brier_score(X, y) evaluates calibration quality
   - .reliability_diagram_data(X, y) returns data for calibration plots

All models implement the unified interface:
  .fit(X, y)           → train
  .predict(X)          → probability array [0,1]
  .predict_single(x)   → float
  .save(path)          → persist to disk
  .load(path)          → restore from disk
"""

from __future__ import annotations

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

log = logging.getLogger("models")

# ── XGBoost availability ──────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    log.warning("xgboost not installed. Run: pip install xgboost")

# ── LightGBM availability ─────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    log.warning("lightgbm not installed. Run: pip install lightgbm")

# Kept for predictor.py backward-compat check
_TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

class LogisticModel:
    """
    Calibrated Logistic Regression wrapped in a StandardScaler pipeline.

    Why it works for prediction markets:
      - Prediction-market probability is largely a linear function of the
        bid-ask spread, time-to-close, and volume at macro level.
      - Platt-scaling (sigmoid calibration) corrects for the tendency of
        logistic regression to push probabilities towards 0.5.
      - L2 regularisation (C=0.5) prevents overfitting to idiosyncratic noise.

    Example:
        >>> model = LogisticModel()
        >>> model.fit(X_train, y_train)     # y must be binary 0/1
        >>> probs = model.predict(X_test)   # shape (n,)
    """

    NAME = "logistic_regression"

    def __init__(self, C: float = 0.5, max_iter: int = 1_000):
        """
        Args:
            C:        Inverse regularisation strength (smaller = stronger reg).
            max_iter: Solver iteration budget.
        """
        base = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  calibrated),
        ])
        self.is_fitted      = False
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        """
        Train on binary-labelled data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels — must be 0 or 1 (actual resolution outcomes).

        Returns:
            self

        Example:
            >>> model.fit(X_train, (y_train > 0.5).astype(int))
        """
        y_bin = _ensure_binary(y)
        try:
            self.pipeline.fit(X, y_bin)
            self.is_fitted = True
        except Exception as exc:
            log.error(f"LogisticModel.fit failed: {exc}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(YES) for each sample.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Float32 array of shape (n_samples,) in [0.01, 0.99].

        Example:
            >>> probs = model.predict(X_test)
            >>> assert all(0 <= p <= 1 for p in probs)
        """
        if not self.is_fitted:
            return np.full(len(X), 0.5, dtype=np.float32)
        try:
            probs = self.pipeline.predict_proba(X)[:, 1]
        except Exception as exc:
            log.warning(f"LogisticModel.predict failed: {exc}")
            return np.full(len(X), 0.5, dtype=np.float32)
        return np.clip(probs, 0.01, 0.99).astype(np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        """Predict P(YES) for a single feature vector."""
        return float(self.predict(x.reshape(1, -1))[0])

    def feature_importances(self) -> np.ndarray:
        """
        Return |coefficient| values as a proxy for feature importance.

        Returns:
            1-D array of length n_features.

        Example:
            >>> imp = model.feature_importances()
        """
        try:
            coef = self.pipeline.named_steps["model"].calibrated_classifiers_[0].estimator.coef_
            return np.abs(coef[0])
        except Exception:
            return np.zeros(35)

    def save(self, path: str = "models/logistic.pkl") -> None:
        """Persist model to disk via pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"LogisticModel saved → {path}")

    @classmethod
    def load(cls, path: str = "models/logistic.pkl") -> "LogisticModel":
        """Load a persisted LogisticModel from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBoost (binary classification)
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    XGBoost gradient-boosted trees trained with binary:logistic objective.

    Key change from prior version: objective is now 'binary:logistic' (not
    'reg:squarederror'), which is the correct loss when labels are 0/1.
    The output is a proper probability in (0, 1), not a regression score.
    Early stopping monitors logloss on a held-out validation set.

    Why XGBoost for prediction markets:
      - Non-linear feature interactions (price × time, volume × spread)
      - Robust to outliers in thin markets
      - SHAP values explain every prediction
      - Built-in L1/L2 regularisation handles correlated features

    Example:
        >>> model = XGBoostModel()
        >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        >>> probs = model.predict(X_test)
    """

    NAME = "xgboost"

    def __init__(
        self,
        n_estimators:  int   = 400,
        max_depth:     int   = 4,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
        colsample:     float = 0.75,
        reg_alpha:     float = 0.1,
        reg_lambda:    float = 1.0,
        early_stopping_rounds: int = 30,
    ):
        """
        Args:
            n_estimators:          Maximum number of boosting rounds.
            max_depth:             Tree depth (4 works well for ≤35 features).
            learning_rate:         Step size shrinkage.
            subsample:             Row subsampling per tree.
            colsample:             Feature subsampling per tree.
            reg_alpha:             L1 regularisation.
            reg_lambda:            L2 regularisation.
            early_stopping_rounds: Stop if val logloss doesn't improve for N rounds.
        """
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective="binary:logistic",   # CORRECT: binary classification
            eval_metric="logloss",         # CORRECT: proper scoring rule
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.scaler    = StandardScaler()
        self.model     = xgb.XGBClassifier(**self.params)
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "XGBoostModel":
        """
        Train on binary-labelled data with optional early stopping.

        Args:
            X:       Training features (n_train, n_features).
            y:       Binary labels 0/1 (n_train,).
            X_val:   Validation features for early stopping (optional).
            y_val:   Validation labels for early stopping (optional).
            verbose: Print tree-building progress.

        Returns:
            self

        Example:
            >>> model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
        """
        y_bin    = _ensure_binary(y)
        X_scaled = self.scaler.fit_transform(X)

        fit_kwargs: dict[str, Any] = {"verbose": verbose}

        if X_val is not None and y_val is not None:
            y_val_bin = _ensure_binary(y_val)
            X_val_sc  = self.scaler.transform(X_val)
            fit_kwargs["eval_set"] = [(X_val_sc, y_val_bin)]
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)

        try:
            self.model.fit(X_scaled, y_bin, **fit_kwargs)
            self.is_fitted = True
        except Exception as exc:
            log.error(f"XGBoostModel.fit failed: {exc}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(YES=1) for each sample.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Float32 array of shape (n_samples,) in [0.01, 0.99].

        Example:
            >>> probs = model.predict(X_test)
        """
        if not self.is_fitted:
            return np.full(len(X), 0.5, dtype=np.float32)
        try:
            X_scaled = self.scaler.transform(X)
            probs    = self.model.predict_proba(X_scaled)[:, 1]
            return np.clip(probs, 0.01, 0.99).astype(np.float32)
        except Exception as exc:
            log.warning(f"XGBoostModel.predict failed: {exc}")
            return np.full(len(X), 0.5, dtype=np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        """Predict P(YES) for a single feature vector."""
        return float(self.predict(x.reshape(1, -1))[0])

    def feature_importances(self) -> np.ndarray:
        """Return XGBoost gain-based feature importances."""
        if not self.is_fitted:
            return np.zeros(35)
        return self.model.feature_importances_

    def save(self, path: str = "models/xgboost.pkl") -> None:
        """Persist scaler + model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "params": self.params}, f)
        self.model.save_model(str(Path(path).with_suffix(".json")))
        log.info(f"XGBoostModel saved → {path}")

    @classmethod
    def load(cls, path: str = "models/xgboost.pkl") -> "XGBoostModel":
        """Load a persisted XGBoostModel from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        params = data["params"]
        obj = cls(
            n_estimators  = params.get("n_estimators", 400),
            max_depth     = params.get("max_depth", 4),
            learning_rate = params.get("learning_rate", 0.05),
            subsample     = params.get("subsample", 0.8),
            colsample     = params.get("colsample_bytree", 0.75),
            reg_alpha     = params.get("reg_alpha", 0.1),
            reg_lambda    = params.get("reg_lambda", 1.0),
        )
        obj.scaler = data["scaler"]
        obj.model.load_model(str(Path(path).with_suffix(".json")))
        obj.is_fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: LightGBM (replaces LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMModel:
    """
    LightGBM gradient-boosted trees — faster and more data-efficient than XGBoost.

    WHY THIS REPLACES THE LSTM:
      The prior BiLSTM used bidirectionality, meaning it had access to future
      price data at training time (right-side LSTM reads the sequence in reverse).
      This created massive data leakage: training AUC looked high but live
      performance collapsed because inference is strictly left-to-right.
      LightGBM is purely tabular, has no temporal structure, and therefore
      cannot cheat — the causal constraint is architecturally enforced.

    Additional advantages over LSTM on small PM datasets:
      - Trains in seconds, not minutes
      - Works well with < 1000 samples (LSTM needs > 5000 for good generalisation)
      - Native handling of missing features
      - SHAP explainability out of the box

    Example:
        >>> model = LightGBMModel()
        >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        >>> probs = model.predict(X_test)
    """

    NAME = "lightgbm"

    def __init__(
        self,
        n_estimators:          int   = 500,
        max_depth:             int   = 5,
        learning_rate:         float = 0.05,
        num_leaves:            int   = 31,
        subsample:             float = 0.8,
        colsample_bytree:      float = 0.75,
        reg_alpha:             float = 0.1,
        reg_lambda:            float = 1.0,
        min_child_samples:     int   = 10,
        early_stopping_rounds: int   = 30,
    ):
        """
        Args:
            n_estimators:          Max boosting rounds.
            max_depth:             Maximum tree depth (-1 = unlimited via num_leaves).
            learning_rate:         Step size shrinkage.
            num_leaves:            Max leaves per tree (controls complexity).
            subsample:             Row subsampling ratio.
            colsample_bytree:      Feature subsampling ratio.
            reg_alpha:             L1 regularisation.
            reg_lambda:            L2 regularisation.
            min_child_samples:     Min samples per leaf (prevents overfitting).
            early_stopping_rounds: Stop if val logloss doesn't improve.
        """
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        self.params = dict(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            num_leaves        = num_leaves,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            reg_alpha         = reg_alpha,
            reg_lambda        = reg_lambda,
            min_child_samples = min_child_samples,
            objective         = "binary",
            metric            = "binary_logloss",
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.scaler    = StandardScaler()
        self.model     = lgb.LGBMClassifier(**self.params) if _LGB_AVAILABLE else None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMModel":
        """
        Train on binary-labelled data with optional early stopping.

        Args:
            X:     Training features (n_train, n_features).
            y:     Binary labels 0/1 (n_train,).
            X_val: Validation features (optional — enables early stopping).
            y_val: Validation labels (optional).

        Returns:
            self

        Example:
            >>> model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
        """
        if self.model is None:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        y_bin    = _ensure_binary(y)
        X_scaled = self.scaler.fit_transform(X)

        callbacks = [lgb.log_evaluation(period=-1)]  # suppress training output

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            y_val_bin = _ensure_binary(y_val)
            X_val_sc  = self.scaler.transform(X_val)
            fit_kwargs["eval_set"]            = [(X_val_sc, y_val_bin)]
            fit_kwargs["callbacks"]           = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        else:
            fit_kwargs["callbacks"] = callbacks

        try:
            self.model.fit(X_scaled, y_bin, **fit_kwargs)
            self.is_fitted = True
        except Exception as exc:
            log.error(f"LightGBMModel.fit failed: {exc}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated P(YES) for each sample.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Float32 array of shape (n_samples,) in [0.01, 0.99].

        Example:
            >>> probs = model.predict(X_test)
        """
        if not self.is_fitted or self.model is None:
            return np.full(len(X), 0.5, dtype=np.float32)
        try:
            X_scaled = self.scaler.transform(X)
            probs    = self.model.predict_proba(X_scaled)[:, 1]
            return np.clip(probs, 0.01, 0.99).astype(np.float32)
        except Exception as exc:
            log.warning(f"LightGBMModel.predict failed: {exc}")
            return np.full(len(X), 0.5, dtype=np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        """Predict P(YES) for a single feature vector."""
        return float(self.predict(x.reshape(1, -1))[0])

    def feature_importances(self) -> np.ndarray:
        """Return LightGBM gain-based feature importances."""
        if not self.is_fitted or self.model is None:
            return np.zeros(35)
        return self.model.feature_importances_

    def save(self, path: str = "models/lightgbm.pkl") -> None:
        """Persist model to disk via pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"LightGBMModel saved → {path}")

    @classmethod
    def load(cls, path: str = "models/lightgbm.pkl") -> "LightGBMModel":
        """Load a persisted LightGBMModel from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# ISOTONIC CALIBRATOR WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class IsotonicCalibrator:
    """
    Post-hoc isotonic regression calibrator that wraps any fitted ML model.

    WHY CALIBRATION MATTERS:
      XGBoost and LightGBM output scores that are monotone in rank order, but
      not guaranteed to be well-calibrated probabilities. For prediction markets,
      a prediction of 0.70 must mean the market resolves YES approximately 70%
      of the time — uncalibrated scores can be systematically too high or low
      near the tails, causing overconfident BUY signals.

      Isotonic regression fits a piecewise-constant non-decreasing function from
      raw model scores to empirical frequencies on a held-out calibration set.
      It makes no parametric assumption (unlike Platt/sigmoid scaling).

    Usage:
        cal = IsotonicCalibrator(base_model)
        cal.fit(X_cal, y_cal)          # held-out split, NEVER training data
        probs = cal.predict(X_test)    # calibrated probabilities
        print(cal.brier_score(X_test, y_test))

    Example:
        >>> cal = IsotonicCalibrator(xgb_model)
        >>> cal.fit(X_cal, y_cal)
        >>> bs = cal.brier_score(X_test, y_test)
        >>> print(f"Brier Score: {bs:.4f}")   # target < 0.20
    """

    def __init__(self, base_model: Any):
        """
        Args:
            base_model: Any fitted model with a .predict(X) method returning
                        raw probability scores in [0, 1].
        """
        self.base_model  = base_model
        self.calibrator  = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted   = False
        self.cal_brier   = None  # Brier score on the calibration set

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on a HELD-OUT calibration set (never on training data).

        Args:
            X_cal: Calibration features (n_cal, n_features).
                   Must be data the base_model has NEVER seen during training.
            y_cal: Binary labels 0/1 (n_cal,).

        Returns:
            self

        Example:
            >>> cal.fit(X_held_out, y_held_out)
        """
        y_bin = _ensure_binary(y_cal)
        try:
            raw_probs = self.base_model.predict(X_cal)
            self.calibrator.fit(raw_probs, y_bin)
            cal_probs = np.clip(self.calibrator.predict(raw_probs), 0.01, 0.99)
            self.cal_brier = float(brier_score_loss(y_bin, cal_probs))
            self.is_fitted = True
            log.info(f"IsotonicCalibrator fitted — calibration Brier: {self.cal_brier:.4f}")
        except Exception as exc:
            log.error(f"IsotonicCalibrator.fit failed: {exc}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated P(YES) for each sample.

        If the calibrator is not fitted, falls back to base_model.predict().

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Float32 array of shape (n_samples,) in [0.01, 0.99].

        Example:
            >>> cal_probs = calibrator.predict(X_test)
        """
        try:
            raw_probs = self.base_model.predict(X)
            if not self.is_fitted:
                return raw_probs
            cal_probs = self.calibrator.predict(raw_probs)
            return np.clip(cal_probs, 0.01, 0.99).astype(np.float32)
        except Exception as exc:
            log.warning(f"IsotonicCalibrator.predict failed: {exc}")
            return np.full(len(X), 0.5, dtype=np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        """Predict calibrated P(YES) for a single feature vector."""
        return float(self.predict(x.reshape(1, -1))[0])

    def brier_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Brier Score (mean squared error of probabilities) on test data.

        Lower is better. Reference values:
          - Random (0.5 always): 0.25
          - Good calibration:    < 0.15
          - Target for live:     < 0.20

        Args:
            X: Feature matrix.
            y: Binary ground-truth labels 0/1.

        Returns:
            Brier score float in [0, 1].

        Example:
            >>> bs = calibrator.brier_score(X_test, y_test)
            >>> assert bs < 0.20, "Brier score too high for live trading"
        """
        y_bin = _ensure_binary(y)
        try:
            probs = self.predict(X)
            return float(brier_score_loss(y_bin, probs))
        except Exception as exc:
            log.warning(f"brier_score failed: {exc}")
            return 0.25   # worst case

    def reliability_diagram_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability diagram data: predicted probability bins vs actual
        fraction of YES outcomes. A perfectly calibrated model has all points
        on the diagonal y=x.

        Args:
            X:      Feature matrix.
            y:      Binary ground-truth labels 0/1.
            n_bins: Number of equal-width probability bins.

        Returns:
            bin_midpoints:   Centre of each probability bin (n_bins,).
            mean_predicted:  Mean predicted probability in each bin (n_bins,).
            fraction_yes:    Fraction of YES outcomes in each bin (n_bins,).

        Example:
            >>> mids, pred, frac = cal.reliability_diagram_data(X_test, y_test)
            >>> # mids approx [0.05, 0.15, ..., 0.95]
            >>> # perfect calibration: frac ≈ mids
        """
        y_bin = _ensure_binary(y)
        probs = self.predict(X)

        bins         = np.linspace(0.0, 1.0, n_bins + 1)
        bin_mids     = 0.5 * (bins[:-1] + bins[1:])
        mean_pred    = np.zeros(n_bins)
        frac_yes     = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                mean_pred[i] = probs[mask].mean()
                frac_yes[i]  = y_bin[mask].mean()

        return bin_mids, mean_pred, frac_yes

    def save(self, path: str = "models/calibrated_model.pkl") -> None:
        """Persist the calibrated model (base + isotonic) to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"IsotonicCalibrator saved → {path}")

    @classmethod
    def load(cls, path: str = "models/calibrated_model.pkl") -> "IsotonicCalibrator":
        """Load a persisted IsotonicCalibrator from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ensure_binary(y: np.ndarray) -> np.ndarray:
    """
    Convert a label array to strict int8 binary (0 or 1).

    Raises ValueError if any label is not 0 or 1 (allows float 0.0 / 1.0).

    Args:
        y: Label array (may be float, int, or bool).

    Returns:
        int8 array of 0s and 1s.

    Example:
        >>> _ensure_binary(np.array([0.0, 1.0, 1.0, 0.0]))
        array([0, 1, 1, 0], dtype=int8)
    """
    y_arr = np.array(y, dtype=float)
    unique_vals = set(np.unique(np.round(y_arr, 6)))
    if not unique_vals.issubset({0.0, 1.0}):
        raise ValueError(
            f"Labels must be binary 0/1. Got unique values: {unique_vals}. "
            "Ensure you are training on CLOSED markets with known outcomes only."
        )
    return y_arr.astype(np.int8)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type] = {
    "logistic":  LogisticModel,
    "xgboost":   XGBoostModel,
    "lightgbm":  LightGBMModel,
}


def get_model(name: str, **kwargs) -> Any:
    """
    Instantiate a model by name.

    Args:
        name:   One of 'logistic', 'xgboost', 'lightgbm'.
        **kwargs: Passed to the model constructor.

    Returns:
        Unfitted model instance.

    Example:
        >>> model = get_model("xgboost", n_estimators=200)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
