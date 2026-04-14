"""
train_model.py
==============
Standalone training script for PolyBot ML models.

Usage:
    python train_model.py              # train all models, compare, plot
    python train_model.py --model xgboost   # train only XGBoost
    python train_model.py --eval             # eval only (no retraining)
    python train_model.py --plot             # regenerate plots from saved models
    python train_model.py --signal           # run live signal demo on current markets

Examples:
    python train_model.py
    python train_model.py --model logistic --eval
    python train_model.py --signal
"""

import sys
import argparse
import numpy as np

# Windows UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from polybot.ml.trainer   import ModelTrainer, load_training_data
from polybot.ml.evaluator import ModelEvaluator
from polybot.ml.predictor import MLPredictor, ModelLoader
from polybot.ml.features  import FEATURE_COLS
from polybot.logger import get_logger

log = get_logger("train_model")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PolyBot ML Training & Evaluation")
    p.add_argument("--model",   default="all",  choices=["all", "logistic", "xgboost", "lstm"])
    p.add_argument("--eval",    action="store_true", help="Evaluate saved models (no retraining)")
    p.add_argument("--plot",    action="store_true", help="Generate or re-generate diagnostic plots")
    p.add_argument("--signal",  action="store_true", help="Run live signal demo on current markets")
    p.add_argument("--forward", type=int, default=5,  help="Label lookahead steps (default: 5)")
    p.add_argument("--no-synthetic", action="store_true", help="Skip synthetic data (real data only)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    trainer = ModelTrainer(model_dir="models", test_size=0.20)
    results = trainer.train_all(n_forward=args.forward)

    if not results:
        print("\n  No training data available.")
        print("  Run the bot first:  python main.py")
        print("  Or use:             python train_model.py  (uses synthetic data)")
        return

    trainer.print_comparison(results)
    best_name = trainer.best_model(results, metric="rmse")
    print(f"\n  Best model: {best_name}")

    # Feature importance of the best tabular model
    for m_name in ("xgboost", "logistic"):
        if trainer.get_trained(m_name):
            trainer.print_feature_importance(m_name, top_n=15)
            break

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_saved(args, trainer=None):
    """Evaluate all saved models on held-out test data."""
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    X, y = load_training_data(n_forward=args.forward)
    if len(X) < 40:
        print("  Not enough data for evaluation.")
        return

    n_test        = max(20, int(len(X) * 0.20))
    X_test, y_test = X[-n_test:], y[-n_test:]
    prices_test   = X_test[:, 0]    # feature[0] = price (first feature column)

    evaluator = ModelEvaluator(model_name="auto")
    reports   = evaluator.compare_models(X_test, y_test)

    if reports:
        best = min(reports, key=lambda r: r.brier_score)
        print(f"\n  Best model by Brier Score: {best.model_name} ({best.brier_score:.4f})")

    return X_test, y_test, prices_test, evaluator


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(X_test, y_test, prices_test):
    print("\n  Generating diagnostic plots...")
    for model_name in ("xgboost", "logistic", "lstm"):
        from pathlib import Path
        pred_path = f"models/{model_name}.pkl" if model_name != "lstm" else "models/lstm.pt"
        if not Path(pred_path).exists():
            continue
        evl = ModelEvaluator(model_name=model_name)
        evl.plot_all(
            X_test, y_test,
            market_prices = prices_test,
            save_path     = f"plots/eval_{model_name}.png",
        )
        evl.plot_edge_vs_confidence(
            X_test, y_test,
            market_prices = prices_test,
            save_path     = f"plots/edge_conf_{model_name}.png",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Live signal demo
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_demo():
    """
    Fetch live markets and show the ML model's signal for each one.
    Demonstrates: predicted_prob vs market_price, edge, direction, confidence.
    """
    print("\n" + "=" * 65)
    print("  LIVE SIGNAL DEMO")
    print("=" * 65)

    from polybot.data_layer import fetch_active_markets, build_snapshot, store
    from polybot.ml.predictor import MLPredictor

    pred     = MLPredictor(model_name="auto")
    markets  = fetch_active_markets(limit=15)
    shown    = 0

    for raw in markets[:15]:
        snap = build_snapshot(raw)
        if snap is None or not snap.is_tradeable:
            continue
        history = store.load(snap.condition_id)
        signal  = pred.predict_and_signal(snap, history)
        pred.print_comparison(snap, history)
        shown += 1
        if shown >= 8:
            break

    if shown == 0:
        print("  No tradeable markets found. API may be unreachable.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 65)
    print("  PolyBot ML Training Pipeline")
    print(f"  Mode: {'eval-only' if args.eval else 'train'} | Model: {args.model} | Forward: {args.forward}")
    print("=" * 65 + "\n")

    trainer_obj = None

    if not args.eval and not args.signal:
        trainer_obj = train(args)

    eval_data = None
    if args.eval or args.plot or (trainer_obj is not None):
        eval_data = evaluate_saved(args, trainer_obj)

    if args.plot and eval_data:
        X_test, y_test, prices_test, _ = eval_data
        generate_plots(X_test, y_test, prices_test)

    if args.signal:
        run_signal_demo()

    print("\nDone.")
