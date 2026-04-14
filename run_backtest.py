"""
run_backtest.py
===============
CLI entry point for the PolyBot backtesting engine.

Modes:
  python run_backtest.py                    # run on all stored markets
  python run_backtest.py --market <id>      # run on one specific market
  python run_backtest.py --synthetic        # run on generated synthetic data
  python run_backtest.py --synthetic --fee 0.005 --slip 0.001   # custom costs
  python run_backtest.py --compare          # portfolio comparison chart

Options:
  --fee FLOAT        Taker fee rate (default 0.01 = 1.0%)
  --slip FLOAT       Slippage coefficient (default 0.002 = 0.2% at 100% depth)
  --liq FLOAT        Max position as fraction of 24h volume (default 0.05)
  --pos FLOAT        Capital fraction per trade (default 0.02 = 2%)
  --capital FLOAT    Starting capital in USD (default from .env)
  --warmup INT       Warmup bars before signals trusted (default 15)
  --no-plot          Skip chart generation
  --no-news          Disable Gemini news signal in BT (default: disabled)
  --outdir DIR       Output directory for plots/CSV (default: results/)
  --bars INT         Bars for synthetic data (default: 500)
"""

import sys
import argparse
from pathlib import Path

# Windows UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from polybot.backtester import (
    Backtester, BacktestReport,
    aggregate_reports, _generate_synthetic_ohlcv,
    TAKER_FEE_RATE, SLIPPAGE_COEFF, MAX_LIQ_FRACTION,
)
from polybot.bt_visualiser import BacktestVisualiser, plot_portfolio_comparison
from polybot.data_layer import store
from polybot.config import CAPITAL
from polybot.logger import get_logger

log = get_logger("run_backtest")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PolyBot Backtesting Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--market",    default=None,   help="Single condition_id to backtest")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic price data")
    p.add_argument("--compare",   action="store_true", help="Generate portfolio comparison chart")
    p.add_argument("--fee",       type=float, default=TAKER_FEE_RATE, help="Taker fee rate")
    p.add_argument("--slip",      type=float, default=SLIPPAGE_COEFF,  help="Slippage coefficient")
    p.add_argument("--liq",       type=float, default=MAX_LIQ_FRACTION, help="Max liquidity fraction")
    p.add_argument("--pos",       type=float, default=0.02,  help="Position size as fraction of capital")
    p.add_argument("--capital",   type=float, default=CAPITAL, help="Starting capital (USD)")
    p.add_argument("--warmup",    type=int,   default=15,   help="Warmup bars")
    p.add_argument("--bars",      type=int,   default=500,  help="Bars for synthetic simulation")
    p.add_argument("--no-plot",   action="store_true", help="Skip chart generation")
    p.add_argument("--no-news",   action="store_true", help="Disable news signal (faster)")
    p.add_argument("--outdir",    default="results", help="Output directory")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(args):
    sep = "═" * 68
    print(f"\n{sep}")
    print("  PolyBot Backtesting Engine")
    print(f"{sep}")
    print(f"  Fee rate:        {args.fee:.2%}  (taker, per side)")
    print(f"  Slippage coeff:  {args.slip:.4f}  (sqrt market-impact model)")
    print(f"  Liquidity cap:   {args.liq:.0%}  of 24h volume per position")
    print(f"  Position size:   {args.pos:.0%}  of capital per trade")
    print(f"  Capital:         ${args.capital:.2f}")
    print(f"  Output dir:      {args.outdir}/")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results(report: BacktestReport, outdir: Path, suffix: str = ""):
    """Save trade log CSV and summary JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    safe_id = report.condition_id[:20].replace("/", "_") + suffix
    report.save_trade_log(str(outdir / f"trades_{safe_id}.csv"))


def run_and_plot(
    bt:        Backtester,
    report:    BacktestReport,
    vis:       BacktestVisualiser,
    args,
    outdir:    Path,
    suffix:    str = "",
) -> BacktestReport:
    report.print()
    save_results(report, outdir, suffix)

    if not args.no_plot and report.total_trades > 0:
        safe_id = report.condition_id[:15].replace("/", "_") + suffix
        vis.plot(report, save_path=str(outdir / f"bt_{safe_id}.png"))
        vis.save_summary(report, str(outdir / f"summary_{safe_id}.json"))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print_banner(args)

    bt = Backtester(
        fee_rate        = args.fee,
        slippage_coeff  = args.slip,
        liquidity_cap   = args.liq,
        position_pct    = args.pos,
        warmup_rows     = args.warmup,
        use_news        = not args.no_news,
        initial_capital = args.capital,
    )
    vis = BacktestVisualiser(output_dir=str(outdir / "plots"))

    # ── Synthetic mode ────────────────────────────────────────────────────────
    if args.synthetic:
        from polybot import config as bot_config
        # Lower thresholds so synthetic noise generates signals for demonstration
        bot_config.MIN_CONFIDENCE = 0.30
        bot_config.MIN_EDGE = 0.02

        print(f"  Generating {args.bars}-bar synthetic OHLCV data...\n")
        # Increase volatility to trigger momentum signals
        df  = _generate_synthetic_ohlcv(n_bars=args.bars, drift=0.003, vol=0.03)
        rep = bt.run_on_dataframe(df, "synthetic", "Synthetic market: will price cross 0.60?")
        run_and_plot(bt, rep, vis, args, outdir, suffix="_synthetic")
        return

    # ── Single market mode ────────────────────────────────────────────────────
    if args.market:
        markets = store.list_markets()
        if args.market not in markets:
            print(f"  Market '{args.market}' not found in data store.")
            print(f"  Available: {markets[:10]}")
            return
        rep = bt.run(args.market, question=args.market)
        run_and_plot(bt, rep, vis, args, outdir)
        return

    # ── All markets mode ─────────────────────────────────────────────────────
    markets = store.list_markets()
    if not markets:
        print("  No historical market data found.")
        print("  Run the bot first:  python main.py")
        print("  Or use:             python run_backtest.py --synthetic\n")
        return

    print(f"  Found {len(markets)} markets in data store.\n")
    reports = []
    for cid in markets:
        rep = bt.run(cid, question=cid)
        run_and_plot(bt, rep, vis, args, outdir)
        reports.append(rep)

    # ── Portfolio aggregation ─────────────────────────────────────────────────
    if reports:
        print("\n" + "─" * 65)
        print("  PORTFOLIO SUMMARY")
        print("─" * 65)
        summary = aggregate_reports(reports)
        for k, v in summary.items():
            print(f"  {k:<30} {v}")
        print()

        if args.compare and not args.no_plot:
            plot_portfolio_comparison(
                reports,
                save_path=str(outdir / "plots" / "portfolio_comparison.png"),
            )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
