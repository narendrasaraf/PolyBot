"""
polybot/ml/evaluator.py
=======================
ML Model Evaluator + Backtester with Trading Signal Simulation

Runs the ML model's predictions through a walk-forward backtest,
comparing predicted probability vs market price at each timestep
to generate trade signals and measure:

Probability Calibration Metrics:
  - Brier Score  (lower = better calibrated)
  - ROC-AUC      (higher = better discrimination)
  - Log-Loss     (penalises confident wrong predictions)
  - Expected Calibration Error (ECE)
  - Reliability diagram data (calibration curve)

Trading Performance Metrics:
  - Win rate, ROI, Sharpe ratio
  - Alpha vs. random baseline
  - Edge decay (how fast edge erodes as confidence cuts change)
  - Signal count per confidence tier (>70%, >80%, >90%)
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    brier_score_loss, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error,
)
from sklearn.calibration import calibration_curve

from polybot.ml.features import build_feature_matrix, FEATURE_COLS
from polybot.ml.predictor import MLPredictor, ModelLoader
from polybot.config import CAPITAL, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from polybot.logger import get_logger

log = get_logger("evaluator")

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


# ── Evaluation report ─────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    model_name:       str
    n_samples:        int
    # Probability metrics
    brier_score:      float
    roc_auc:          float
    log_loss_val:     float
    ece:              float          # Expected Calibration Error
    rmse:             float
    mae:              float
    # Trading simulation metrics
    n_signals:        int = 0
    n_high_conf:      int = 0        # signals >= 70% confidence
    trade_win_rate:   float = 0.0
    trade_roi:        float = 0.0
    sharpe:           float = 0.0
    max_drawdown:     float = 0.0
    # Signal distribution
    tier_70:          int = 0        # signals with conf >= 70%
    tier_80:          int = 0
    tier_90:          int = 0

    def print(self):
        sep = "═" * 65
        G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; E = "\033[0m"
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT — {self.model_name}")
        print(sep)
        print(f"\n  Probability Calibration:")
        bc = G if self.brier_score < 0.20 else (Y if self.brier_score < 0.25 else R)
        print(f"    Brier Score:    {bc}{self.brier_score:.4f}{E}  (lower=better, random=0.25)")
        ac = G if self.roc_auc > 0.60 else (Y if self.roc_auc > 0.55 else R)
        print(f"    ROC-AUC:        {ac}{self.roc_auc:.4f}{E}  (higher=better, random=0.50)")
        print(f"    Log-Loss:       {self.log_loss_val:.4f}  (lower=better)")
        ec = G if self.ece < 0.05  else (Y if self.ece < 0.10  else R)
        print(f"    ECE:            {ec}{self.ece:.4f}{E}  (lower=better, <5%=well-calibrated)")
        print(f"    RMSE:           {self.rmse:.4f}")
        print(f"    MAE:            {self.mae:.4f}")
        print(f"\n  Trading Simulation:")
        print(f"    Total signals:  {self.n_signals}")
        print(f"    High-conf (70%+): {self.n_high_conf}  ({self.n_high_conf/max(self.n_signals,1):.0%})")
        print(f"    By tier:        70%+={self.tier_70}  80%+={self.tier_80}  90%+={self.tier_90}")
        wc = G if self.trade_win_rate > 0.55 else (Y if self.trade_win_rate > 0.50 else R)
        print(f"    Win rate:       {wc}{self.trade_win_rate:.1%}{E}")
        rc = G if self.trade_roi > 0 else R
        print(f"    ROI:            {rc}{self.trade_roi:+.1f}%{E}")
        sc = G if self.sharpe > 0.5 else (Y if self.sharpe > 0 else R)
        print(f"    Sharpe ratio:   {sc}{self.sharpe:.3f}{E}")
        print(f"    Max drawdown:   {self.max_drawdown:.1%}")
        print(sep + "\n")


# ── Calibration metrics ───────────────────────────────────────────────────────

def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well the model's confidence aligns with actual accuracy.
    ECE < 0.05 = well-calibrated.
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    n_total  = len(y_true)

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = float(y_pred[mask].mean())
        avg_acc  = float(y_true[mask].mean())
        ece += (mask.sum() / n_total) * abs(avg_conf - avg_acc)

    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute the full suite of calibration metrics."""
    y_bin = (y_true > 0.5).astype(int)
    return {
        "brier":   float(brier_score_loss(y_bin, y_pred)),
        "auc":     float(roc_auc_score(y_bin, y_pred)) if len(np.unique(y_bin)) > 1 else 0.5,
        "logloss": float(log_loss(y_bin, np.clip(y_pred, 1e-7, 1 - 1e-7))),
        "ece":     expected_calibration_error(y_bin.astype(float), y_pred),
        "rmse":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":     float(mean_absolute_error(y_true, y_pred)),
    }


# ── Trading signal simulation ─────────────────────────────────────────────────

def simulate_trades(
    y_true:         np.ndarray,
    y_pred:         np.ndarray,
    market_prices:  np.ndarray,
    min_edge:       float = 0.06,
    min_conf:       float = 0.60,
    position_pct:   float = 0.02,
    capital:        float = CAPITAL,
) -> dict:
    """
    Walk-forward trading simulation using ML predictions.

    At each step:
      - If |pred - price| > min_edge → generate signal
      - Simulate BUY at price, EXIT at price+edge (simplified)
      - Track equity curve

    Returns dict with win_rate, roi, sharpe, max_drawdown, n_signals, etc.
    """
    equity       = [capital]
    wins, losses = 0, 0
    n_signals    = 0
    n_high       = 0
    returns_list = []

    tier_70 = tier_80 = tier_90 = 0

    for i in range(len(y_pred)):
        pred    = float(y_pred[i])
        price   = float(market_prices[i]) if i < len(market_prices) else 0.5
        actual  = float(y_true[i])
        edge    = pred - price

        # Confidence estimate (simplified — uses edge magnitude)
        conf = min(1.0, abs(edge) / 0.25)

        if conf >= 0.70:
            tier_70 += 1
        if conf >= 0.80:
            tier_80 += 1
        if conf >= 0.90:
            tier_90 += 1

        # Filter: must meet edge and confidence thresholds
        if abs(edge) < min_edge or conf < min_conf:
            continue

        n_signals += 1
        if conf >= 0.70:
            n_high += 1

        position_usd = capital * position_pct
        direction    = "BUY_YES" if edge > 0 else "BUY_NO"

        # Simplified P&L: if direction is right, earn edge × position
        right = (direction == "BUY_YES" and actual > price) or (direction == "BUY_NO" and actual < price)

        sl   = price * (1 - STOP_LOSS_PCT)
        tp   = price * (1 + TAKE_PROFIT_PCT)

        if right:
            # Take profit scenario
            pnl = position_usd * TAKE_PROFIT_PCT
            wins += 1
        else:
            # Stop loss scenario
            pnl = -position_usd * STOP_LOSS_PCT
            losses += 1

        capital += pnl
        equity.append(capital)
        returns_list.append(pnl / position_usd)

    # Metrics
    total_trades = wins + losses
    win_rate     = wins / max(total_trades, 1)
    total_roi    = (capital - CAPITAL) / CAPITAL * 100
    sharpe       = _sharpe(returns_list)
    mdd          = _max_drawdown(equity)

    return {
        "win_rate":   win_rate,
        "roi":        total_roi,
        "sharpe":     sharpe,
        "max_drawdown": mdd,
        "n_signals":  n_signals,
        "n_high_conf": n_high,
        "tier_70":    tier_70,
        "tier_80":    tier_80,
        "tier_90":    tier_90,
        "equity":     equity,
        "n_wins":     wins,
        "n_losses":   losses,
    }


def _sharpe(returns: list, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr  = np.array(returns)
    mean = arr.mean() - rf
    std  = arr.std(ddof=1)
    return float(mean / std * math.sqrt(252)) if std > 1e-9 else 0.0


def _max_drawdown(equity: list) -> float:
    arr  = np.array(equity)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / np.where(peak > 0, peak, 1.0)
    return float(abs(dd.min()))


# ── Full evaluator orchestrator ───────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates trained models on historical data and generates:
      - Calibration metrics
      - Trading simulation metrics
      - Matplotlib visualisation plots
    """

    def __init__(self, model_name: str = "auto"):
        self.model_name = model_name
        self._predictor = MLPredictor(model_name=model_name)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        market_prices: Optional[np.ndarray] = None,
    ) -> EvalReport:
        """Run full evaluation on a pre-built test set."""
        model  = self._predictor._get_model()
        m_name = getattr(model, "NAME", "heuristic") if model else "heuristic"

        if model is None or len(X_test) == 0:
            return EvalReport(
                model_name=m_name, n_samples=0,
                brier_score=0.25, roc_auc=0.5, log_loss_val=0.693,
                ece=0.1, rmse=0.5, mae=0.5,
            )

        y_pred = model.predict(X_test)
        prices = market_prices if market_prices is not None else np.full(len(y_test), 0.5)

        metrics = compute_all_metrics(y_test, y_pred)
        trades  = simulate_trades(y_test, y_pred, prices)

        return EvalReport(
            model_name    = m_name,
            n_samples     = len(X_test),
            brier_score   = metrics["brier"],
            roc_auc       = metrics["auc"],
            log_loss_val  = metrics["logloss"],
            ece           = metrics["ece"],
            rmse          = metrics["rmse"],
            mae           = metrics["mae"],
            n_signals     = trades["n_signals"],
            n_high_conf   = trades["n_high_conf"],
            trade_win_rate= trades["win_rate"],
            trade_roi     = trades["roi"],
            sharpe        = trades["sharpe"],
            max_drawdown  = trades["max_drawdown"],
            tier_70       = trades["tier_70"],
            tier_80       = trades["tier_80"],
            tier_90       = trades["tier_90"],
        )

    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> list[EvalReport]:
        """Evaluate all three model families and return comparison."""
        reports = []
        for name in ("xgboost", "logistic", "lightgbm"):  # lstm removed in ML rewrite
            m = ModelLoader.load_by_name(name)
            if m is None:
                continue
            evl = ModelEvaluator(model_name=name)
            evl._predictor._model = m
            rep = evl.evaluate(X_test, y_test)
            reports.append(rep)
            rep.print()
        return reports

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_all(
        self,
        X_test:         np.ndarray,
        y_test:         np.ndarray,
        market_prices:  Optional[np.ndarray] = None,
        save_path:      str = "plots/ml_evaluation.png",
    ):
        """
        Generate a 2×3 grid of diagnostic plots:
          [1] Calibration curve (reliability diagram)
          [2] Predicted prob vs actual outcome scatter
          [3] Edge distribution histogram
          [4] Equity curve from trading simulation
          [5] Confidence tier bar chart
          [6] Feature importance (XGBoost)
        """
        model = self._predictor._get_model()
        if model is None or len(X_test) == 0:
            log.warning("No model / data to plot")
            return

        y_pred = model.predict(X_test)
        prices = market_prices if market_prices is not None else np.full(len(y_test), 0.5)
        y_bin  = (y_test > 0.5).astype(int)
        edges  = y_pred - prices
        trades = simulate_trades(y_test, y_pred, prices)

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor("#0f1117")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
        _GRAY = "#1e2130"
        _BLUE = "#4f8ef7"
        _GREEN = "#22c55e"
        _RED   = "#ef4444"
        _GOLD  = "#f59e0b"
        _TEXT  = "#e2e8f0"

        ax_style = dict(facecolor=_GRAY, labelcolor=_TEXT, titlecolor=_TEXT)
        def _ax(pos):
            ax = fig.add_subplot(gs[pos])
            ax.set_facecolor(_GRAY)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            ax.tick_params(colors=_TEXT)
            ax.xaxis.label.set_color(_TEXT)
            ax.yaxis.label.set_color(_TEXT)
            ax.title.set_color(_TEXT)
            return ax

        # ── Plot 1: Calibration curve ─────────────────────────────────────────
        ax1 = _ax((0, 0))
        prob_true, prob_pred = calibration_curve(y_bin, y_pred, n_bins=10, strategy="uniform")
        ax1.plot([0, 1], [0, 1], ":", color="#475569", label="Perfect calibration")
        ax1.plot(prob_pred, prob_true, "o-", color=_BLUE, linewidth=2, markersize=6, label="Model")
        ax1.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color=_BLUE)
        ax1.set_title("Reliability Diagram (Calibration Curve)")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.legend(facecolor="#0f1117", labelcolor=_TEXT)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

        # ── Plot 2: Predicted vs Actual scatter ───────────────────────────────
        ax2 = _ax((0, 1))
        colours = [_GREEN if b == 1 else _RED for b in y_bin]
        ax2.scatter(y_pred, y_test, c=colours, alpha=0.4, s=15, edgecolors="none")
        ax2.plot([0, 1], [0, 1], ":", color="#475569")
        ax2.set_title("Predicted Prob vs Actual Outcome")
        ax2.set_xlabel("Model Predicted Probability")
        ax2.set_ylabel("Actual Outcome (future price)")
        from matplotlib.lines import Line2D
        ax2.legend(
            handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor=_GREEN, label='YES (outcome>0.5)'),
                     Line2D([0],[0], marker='o', color='w', markerfacecolor=_RED,   label='NO  (outcome<=0.5)')],
            facecolor="#0f1117", labelcolor=_TEXT,
        )

        # ── Plot 3: Edge distribution ─────────────────────────────────────────
        ax3 = _ax((0, 2))
        n_bins = 30
        ax3.hist(edges, bins=n_bins, color=_BLUE, alpha=0.8, edgecolor="none")
        ax3.axvline(0, color=_TEXT, linestyle="--", linewidth=1)
        ax3.axvline(0.06,  color=_GREEN, linestyle=":", linewidth=1.5, label="Min edge (6c)")
        ax3.axvline(-0.06, color=_RED,   linestyle=":", linewidth=1.5)
        ax3.set_title("Edge Distribution (Predicted − Market Price)")
        ax3.set_xlabel("Edge (predicted prob − market price)")
        ax3.set_ylabel("Frequency")
        ax3.legend(facecolor="#0f1117", labelcolor=_TEXT)

        # ── Plot 4: Equity curve ──────────────────────────────────────────────
        ax4 = _ax((1, 0))
        eq = trades["equity"]
        x  = range(len(eq))
        ax4.plot(x, eq, color=_BLUE, linewidth=1.8)
        ax4.fill_between(x, CAPITAL, eq,
                         where=[e >= CAPITAL for e in eq], alpha=0.2, color=_GREEN)
        ax4.fill_between(x, CAPITAL, eq,
                         where=[e < CAPITAL for e in eq],  alpha=0.2, color=_RED)
        ax4.axhline(CAPITAL, color="#475569", linestyle="--", linewidth=1)
        ax4.set_title("Simulated Equity Curve")
        ax4.set_xlabel("Trade Number")
        ax4.set_ylabel("Portfolio Value (USD)")
        roi_text = f"ROI: {trades['roi']:+.1f}%  |  Sharpe: {trades['sharpe']:.2f}"
        ax4.text(0.97, 0.04, roi_text, transform=ax4.transAxes,
                 ha='right', va='bottom', color=_TEXT, fontsize=9)

        # ── Plot 5: Confidence tier bar chart ─────────────────────────────────
        ax5 = _ax((1, 1))
        tiers   = ["≥60%\n(actionable)", "≥70%\n(high conf)", "≥80%\n(strong)", "≥90%\n(very strong)"]
        counts  = [trades["n_signals"], trades["tier_70"], trades["tier_80"], trades["tier_90"]]
        colours2 = [_BLUE, _GOLD, _GREEN, "#a855f7"]
        bars = ax5.bar(tiers, counts, color=colours2, width=0.55, edgecolor="none")
        for bar, cnt in zip(bars, counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(cnt), ha="center", va="bottom", color=_TEXT, fontsize=9)
        ax5.set_title("Signals by Confidence Tier")
        ax5.set_ylabel("Number of Signals")

        # ── Plot 6: Feature importance (XGBoost) ─────────────────────────────
        ax6 = _ax((1, 2))
        try:
            from polybot.ml.features import feature_importance_table
            importances = model.feature_importances()
            df_fi = feature_importance_table(importances).head(12)
            ax6.barh(df_fi["feature"][::-1], df_fi["importance"][::-1],
                     color=_BLUE, edgecolor="none")
            ax6.set_title(f"Top Features ({getattr(model,'NAME','model')})")
            ax6.set_xlabel("Importance")
        except Exception:
            ax6.text(0.5, 0.5, "Feature importance\nnot available\nfor this model",
                     ha="center", va="center", color=_TEXT, transform=ax6.transAxes)
            ax6.set_title("Feature Importance")

        # ── Title and save ────────────────────────────────────────────────────
        m_name = getattr(model, "NAME", "model").upper()
        brier  = brier_score_loss(y_bin, y_pred)
        auc    = roc_auc_score(y_bin, y_pred) if len(np.unique(y_bin)) > 1 else 0.5
        fig.suptitle(
            f"PolyBot ML Evaluation  ·  {m_name}  ·  "
            f"n={len(X_test)} samples  ·  "
            f"Brier={brier:.3f}  ·  AUC={auc:.3f}",
            color=_TEXT, fontsize=13, fontweight="bold",
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Evaluation plots saved to {save_path}")
        print(f"\n  Plots saved: {save_path}\n")

    def plot_edge_vs_confidence(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        market_prices: Optional[np.ndarray] = None,
        save_path: str = "plots/edge_confidence.png",
    ):
        """
        Scatter of |edge| vs confidence, coloured by whether the trade was right.
        Quadrant top-right = high confidence + high edge = ideal trades.
        """
        model = self._predictor._get_model()
        if model is None or len(X_test) == 0:
            return

        y_pred = model.predict(X_test)
        prices = market_prices if market_prices is not None else np.full(len(y_test), 0.5)
        edges  = np.abs(y_pred - prices)
        confs  = np.minimum(1.0, edges / 0.25)
        correct = ((y_pred > prices) == (y_test > prices)).astype(int)

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1117")
        ax.set_facecolor("#1e2130")

        sc = ax.scatter(
            edges, confs,
            c=["#22c55e" if c else "#ef4444" for c in correct],
            alpha=0.5, s=18, edgecolors="none",
        )
        ax.axvline(0.06, color="#f59e0b", linestyle="--", linewidth=1.5, label="Min edge (6c)")
        ax.axhline(0.60, color="#4f8ef7", linestyle="--", linewidth=1.5, label="Min confidence (60%)")
        ax.fill_between([0.06, 1.0], [0.60, 0.60], [1, 1], alpha=0.08, color="#4f8ef7")
        ax.text(0.55, 0.85, "High-confidence zone", color="#4f8ef7", fontsize=9, transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.tick_params(colors="#e2e8f0")
        ax.set_xlabel("Edge Magnitude |predicted − market|", color="#e2e8f0")
        ax.set_ylabel("Confidence",                          color="#e2e8f0")
        ax.set_title("Edge vs Confidence (Green=Correct, Red=Incorrect)",
                     color="#e2e8f0", fontweight="bold")
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor='#22c55e', label='Correct signal'),
                     Line2D([0],[0], marker='o', color='w', markerfacecolor='#ef4444', label='Incorrect signal'),
                     Line2D([0],[0], linestyle='--', color='#f59e0b', label='Min edge'),
                     Line2D([0],[0], linestyle='--', color='#4f8ef7', label='Min confidence')],
            facecolor="#0f1117", labelcolor="#e2e8f0",
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Edge vs Confidence plot saved: {save_path}\n")


# Module-level singleton
evaluator = ModelEvaluator(model_name="auto")
