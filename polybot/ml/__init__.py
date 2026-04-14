"""
polybot/ml/ — Machine Learning Sub-package
==========================================
Provides three model families + a unified predictor interface:

  features.py   — Feature engineering pipeline
  models.py     — LogisticRegression, XGBoost, LSTM definitions
  trainer.py    — Training, cross-validation, model selection
  predictor.py  — Inference + trading signal generation
  evaluator.py  — Backtesting + performance metrics (Brier, AUC, Sharpe)
"""
__all__ = ["features", "models", "trainer", "predictor", "evaluator"]
