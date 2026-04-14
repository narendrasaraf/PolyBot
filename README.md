# PolyBot v2 — Production Polymarket Trading Bot

A modular, production-grade Python trading bot for [Polymarket](https://polymarket.com) with multi-signal strategy engine, full risk management, automated execution, backtesting, and performance analytics.

---

## Architecture

```
polybot/
├── main.py               ← Entry point + orchestration loop
├── polybot/
│   ├── config.py         ← All constants & env vars
│   ├── logger.py         ← Coloured rotating logger
│   ├── data_layer.py     ← Gamma + CLOB API, CSV persistence
│   ├── strategies.py     ← 6-signal strategy engine
│   ├── risk_manager.py   ← Capital rules + position sizing (Kelly)
│   ├── executor.py       ← Order placement + stop/TP monitoring
│   ├── backtester.py     ← Walk-forward backtesting engine
│   ├── metrics.py        ← Performance tracker (Sharpe, MDD, etc.)
│   └── sentiment.py      ← Multi-source news sentiment (no Twitter)
├── data/                 ← Historical CSVs + state JSON (auto-created)
├── .env                  ← API credentials (never commit this!)
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
# Edit .env with your CLOB_API_KEY, WALLET_PRIVATE_KEY, GEMINI_API_KEY

# 3. Test all connections
python main.py test

# 4. Run in paper mode (default — no real money)
python main.py

# 5. Check performance metrics
python main.py metrics

# 6. Run backtests on collected data
python main.py backtest
```

---

## Configuration (`.env`)

| Variable | Description |
|---|---|
| `CLOB_API_KEY` | Polymarket CLOB API key (from `get_keys.py`) |
| `CLOB_SECRET` | CLOB API secret |
| `CLOB_PASSPHRASE` | CLOB API passphrase |
| `WALLET_PRIVATE_KEY` | MetaMask wallet private key |
| `GEMINI_API_KEY` | Google Gemini for news signal |
| `NEWSAPI_KEY` | Optional: NewsAPI.org (100 req/day free) |
| `GNEWS_KEY` | Optional: GNews.io (100 req/day free) |
| `CAPITAL` | Starting capital in USD (default: 10.0) |
| `PAPER_MODE` | `true` = simulate, `false` = live money |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` |

---

## Risk Rules

| Rule | Value |
|---|---|
| Max capital per trade | **2%** (Kelly-sized) |
| Daily loss limit | **5%** of capital |
| Daily profit cap | **15%** (stop overtrading) |
| Stop-loss per position | **5%** from entry |
| Take-profit target | **10%** from entry |
| Min market liquidity | **$5,000** volume |
| Max concurrent positions | **5** |
| Min signal confidence | **60%** |
| Min price edge | **6 cents** |

---

## Strategy Signals

| Signal | Weight | Description |
|---|---|---|
| `momentum` | 20% | Price + volume breakout via rolling window |
| `mean_rev` | 15% | Bollinger Band overbought/oversold |
| `prob_gap` | 20% | Fair value vs market price (base rates + volume efficiency) |
| `news` | 20% | Gemini Google Search grounding for real-time news |
| `reddit` | 10% | Reddit post sentiment (upvote-weighted) |
| `ml` | 10% | Feature-based ensemble (price, time, spread, category, volume) |
| `base_rate` | 5% | Historical YES resolution rates by category |

Signals are combined with **variance-penalized confidence** — the more signals agree, the higher the confidence score.

---

## Performance Metrics

- Win rate
- Total ROI %
- Sharpe ratio (annualised)
- Max drawdown
- Profit factor
- Expectancy per trade
- Best/worst trade

---

## Go Live

1. Set `PAPER_MODE=false` in `.env`
2. Deposit USDC.e on Polygon to your wallet
3. Run `python main.py test` to confirm all connections
4. Start: `python main.py`

> ⚠️ **Warning**: This bot trades real money when `PAPER_MODE=false`. Always test thoroughly in paper mode first. Never risk more than you can afford to lose.

---

## Extending

- **Real ML model**: Replace `strategy_ml()` in `strategies.py` with `joblib.load("models/rf_model.pkl").predict(features)`
- **More signals**: Add a new function to `strategies.py` returning `SignalResult`, register it in `run_all_strategies()`, add its weight to `SIGNAL_WEIGHTS` in `config.py`
- **Database storage**: Swap `HistoricalStore` in `data_layer.py` for a SQLite/PostgreSQL backend
