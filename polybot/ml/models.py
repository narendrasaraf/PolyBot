"""
polybot/ml/models.py
====================
Model Definitions — Three ML architectures for YES-probability prediction.

Model 1: Logistic Regression (sklearn)
  - Fast, interpretable, good baseline
  - L2 regularised with probability calibration via Platt scaling
  - Best for: initial validation, feature importance ranking

Model 2: XGBoost Gradient Boosting
  - State-of-the-art for tabular data
  - Built-in feature importance + SHAP values
  - Handles non-linearity, interactions, and missing data natively
  - Best for: production inference

Model 3: LSTM Neural Network (PyTorch)
  - Captures temporal dependencies across price/volume sequences
  - Best for: learning market regime patterns over time
  - Uses bidirectional LSTM + attention gate + dropout

All models implement a unified interface:
  .fit(X, y)        → train
  .predict(X)       → probability array [0,1]
  .save(path)       → persist to disk
  .load(path)       → restore from disk
"""

from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# ── Sklearn models ────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# ── XGBoost ───────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

class LogisticModel:
    """
    Calibrated Logistic Regression wrapped in a StandardScaler pipeline.

    Why it works for prediction markets:
      - Markets are semi-efficient → linear relationships dominate at macro level
      - Calibration ensures predicted probs map to actual resolution rates
      - Regularisation (C=0.1) prevents overfitting to noise
    """

    NAME = "logistic_regression"

    def __init__(self, C: float = 0.5, max_iter: int = 1000):
        base = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )
        # Platt scaling calibration for better probability estimates
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  calibrated),
        ])
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        """
        Train on binary-labelled data.
        y must be binarised: 1 if future_price > current_price + threshold, else 0
        """
        y_bin = (y > 0.5).astype(int)
        self.pipeline.fit(X, y_bin)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probability of YES outcome for each sample."""
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        probs = self.pipeline.predict_proba(X)[:, 1]
        return probs.astype(np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])

    def feature_importances(self) -> np.ndarray:
        """Return |coefficient| values as proxy for feature importance."""
        coef = self.pipeline.named_steps["model"].calibrated_classifiers_[0].estimator.coef_
        return np.abs(coef[0])

    def save(self, path: str = "models/logistic.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "models/logistic.pkl") -> "LogisticModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    """
    XGBoost gradient-boosted trees for probability regression.

    Why it works best for prediction markets:
      - Non-linear feature interactions (price × volume, time × spread)
      - Robust to outliers and noisy market data
      - SHAP values explain every prediction transparently
      - Objective='reg:squarederror' avoids binary-label waste with continuous probs
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
    ):
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
            objective="reg:squarederror",      # continuous probability output
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler    = StandardScaler()
        self.model     = xgb.XGBRegressor(**self.params)
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
        Train on continuous probability labels (float in [0,1]).
        Optionally pass validation set for early stopping.
        """
        X_scaled = self.scaler.fit_transform(X)
        y_clipped = np.clip(y, 0.001, 0.999)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            X_val_sc = self.scaler.transform(X_val)
            fit_kwargs["eval_set"]          = [(X_val_sc, y_val)]
            fit_kwargs["verbose"]           = verbose
            # Early stopping via callback
            self.model.set_params(early_stopping_rounds=30)

        self.model.fit(X_scaled, y_clipped, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.clip(preds, 0.001, 0.999).astype(np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str = "models/xgboost.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "params": self.params}, f)
        self.model.save_model(str(Path(path).with_suffix(".json")))

    @classmethod
    def load(cls, path: str = "models/xgboost.pkl") -> "XGBoostModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        params = data["params"]
        # Map colsample_bytree back to constructor arg 'colsample'
        init_kwargs = {
            "n_estimators":  params.get("n_estimators", 400),
            "max_depth":     params.get("max_depth", 4),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample":     params.get("subsample", 0.8),
            "colsample":     params.get("colsample_bytree", 0.75),
            "reg_alpha":     params.get("reg_alpha", 0.1),
            "reg_lambda":    params.get("reg_lambda", 1.0),
        }
        obj = cls(**init_kwargs)
        obj.scaler = data["scaler"]
        obj.model.load_model(str(Path(path).with_suffix(".json")))
        obj.is_fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: LSTM Neural Network
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class LSTMNet(nn.Module):
        """
        Bidirectional LSTM with self-attention and residual connection.

        Architecture:
          Input → [BiLSTM × 2 layers] → Attention Gate → FC → Sigmoid

        Why bidirectional? Markets have look-back AND look-ahead context
        in training data (price series is not strictly causal in labelled format).
        """

        def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers  = num_layers

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers  = num_layers,
                dropout     = dropout if num_layers > 1 else 0,
                batch_first = True,
                bidirectional = True,
            )
            lstm_out_size = hidden_size * 2  # bidirectional doubles output

            # Self-attention gate
            self.attention = nn.Sequential(
                nn.Linear(lstm_out_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1),
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(lstm_out_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
            attn_w = self.attention(lstm_out)   # (batch, seq_len, 1)
            context = (attn_w * lstm_out).sum(dim=1)   # (batch, hidden*2)
            out = self.fc(self.dropout(context))        # (batch, 1)
            return out.squeeze(-1)

else:
    # Placeholder if PyTorch is not installed
    class LSTMNet:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch not installed. Run: pip install torch")


class LSTMModel:
    """
    Wrapper around LSTMNet with training loop, sequence preparation, and persistence.

    Why LSTM for prediction markets:
      - Price momentum is temporal: yesterday's move predicts tomorrow at some lag
      - LSTM can learn multi-timescale patterns (hourly, daily, weekly)
      - Attention shows which historical windows matter most for each prediction
    """

    NAME = "lstm"

    def __init__(
        self,
        seq_len:     int   = 20,       # look-back window length
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        lr:          float = 1e-3,
        epochs:      int   = 50,
        batch_size:  int   = 64,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.scaler      = StandardScaler()
        self.net: Optional[LSTMNet] = None
        self.is_fitted   = False
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Convert flat feature matrix → overlapping sequences (seq_len, features)."""
        sequences, targets = [], []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len: i])
            targets.append(y[i])
        if not sequences:
            return np.empty((0, self.seq_len, X.shape[1])), np.empty(0)
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "LSTMModel":
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        y_clipped = np.clip(y, 0.001, 0.999).astype(np.float32)

        Xs, ys = self._make_sequences(X_scaled, y_clipped)
        if len(Xs) < self.batch_size:
            print(f"[LSTM] Warning: only {len(Xs)} sequences — need more data for reliable training")
            if len(Xs) == 0:
                return self

        input_size = X.shape[1]
        self.net   = LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        optimiser  = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)
        criterion  = nn.MSELoss()

        X_t = torch.tensor(Xs).to(self.device)
        y_t = torch.tensor(ys).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        best_loss = float("inf")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [LSTM] Epoch {epoch+1:3d}/{self.epochs} | Loss: {avg_loss:.5f}")
            if avg_loss < best_loss:
                best_loss = avg_loss

        self.net.eval()
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            return np.full(len(X), 0.5, dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        # For single-row or short sequences, pad with zeros
        if len(X_scaled) < self.seq_len:
            pad = np.zeros((self.seq_len - len(X_scaled), X_scaled.shape[1]), dtype=np.float32)
            X_scaled = np.vstack([pad, X_scaled])
        # Build sliding window sequences
        seqs = np.stack([X_scaled[i - self.seq_len: i] for i in range(self.seq_len, len(X_scaled) + 1)])
        with torch.no_grad():
            t     = torch.tensor(seqs).to(self.device)
            preds = self.net(t).cpu().numpy()
        # Pad output if we got fewer predictions than input rows
        gap    = len(X) - len(preds)
        result = np.concatenate([np.full(gap, preds[0] if len(preds) else 0.5), preds])
        return np.clip(result, 0.001, 0.999).astype(np.float32)

    def predict_single(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])

    def save(self, path: str = "models/lstm.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":  self.net.state_dict() if self.net else None,
            "scaler":      self.scaler,
            "config": {
                "seq_len": self.seq_len, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "dropout": self.dropout,
                "input_size": self.net.lstm.input_size if self.net else None,
            },
        }, path)

    @classmethod
    def load(cls, path: str = "models/lstm.pt") -> "LSTMModel":
        data   = torch.load(path, map_location="cpu")
        cfg    = data["config"]
        obj    = cls(
            seq_len=cfg["seq_len"], hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        )
        obj.scaler = data["scaler"]
        if data["state_dict"] is not None and cfg["input_size"]:
            obj.net = LSTMNet(cfg["input_size"], cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
            obj.net.load_state_dict(data["state_dict"])
            obj.net.eval()
            obj.is_fitted = True
        return obj


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "logistic": LogisticModel,
    "xgboost":  XGBoostModel,
    "lstm":     LSTMModel,
}

def get_model(name: str, **kwargs):
    """Instantiate a model by name. name ∈ {'logistic', 'xgboost', 'lstm'}."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
