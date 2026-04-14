"""
polybot/bt_visualiser.py
========================
Backtesting Visualisation Engine

Generates a comprehensive 8-panel dark-theme diagnostic dashboard:

  Panel 1: Equity Curve + Capital at Risk overlay
  Panel 2: Drawdown waterfall chart
  Panel 3: Trade P&L scatter (entry price vs net PnL, sized by confidence)
  Panel 4: Monthly returns heatmap
  Panel 5: Cost breakdown waterfall (gross → fees → slippage → net)
  Panel 6: Win/Loss distribution histogram
  Panel 7: Exit reason breakdown (pie + counts)
  Panel 8: Rolling Sharpe ratio (20-trade window)

Also exports:
  - Trade log CSV
  - Summary stats JSON
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = "#0b0f1a"          # page background
PANEL    = "#111827"          # panel background
BORDER   = "#1e293b"          # panel borders
TEXT     = "#e2e8f0"          # primary text
MUTED    = "#64748b"          # secondary text
BLUE     = "#3b82f6"          # primary accent
CYAN     = "#22d3ee"          # secondary accent
GREEN    = "#22c55e"          # profit colour
RED      = "#ef4444"          # loss colour
AMBER    = "#f59e0b"          # warning colour
PURPLE   = "#a855f7"          # third accent
GRAD_G   = ["#14532d", "#22c55e"]   # green gradient
GRAD_R   = ["#7f1d1d", "#ef4444"]   # red gradient

FONT_FAMILY = "DejaVu Sans"   # always available


def _ax_style(ax):
    """Apply standard dark theme to an axes."""
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, length=3, width=0.6)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("bold")
    return ax


def _fmt_usd(v: float) -> str:
    return f"${v:+.4f}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main visualiser class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BacktestVisualiser:
    """
    Full 8-panel backtest dashboard.

    Usage:
        from polybot.bt_visualiser import BacktestVisualiser
        vis = BacktestVisualiser()
        vis.plot(report, save_path="plots/backtest.png")
        vis.save_summary(report, "results/summary.json")
    """

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def plot(
        self,
        report,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        Generate and save the full 8-panel dashboard.
        Returns the path string of the saved file.
        """
        if save_path is None:
            safe_id  = report.condition_id[:15].replace("/", "_")
            save_path = str(self.output_dir / f"backtest_{safe_id}.png")

        fig = plt.figure(figsize=(22, 16), dpi=120)
        fig.patch.set_facecolor(BG)

        gs = gridspec.GridSpec(
            4, 4,
            figure   = fig,
            hspace   = 0.52,
            wspace   = 0.38,
            left     = 0.055,
            right    = 0.97,
            top      = 0.91,
            bottom   = 0.055,
        )

        # Layout: row 0 spans full width (equity curve)
        ax_equity  = fig.add_subplot(gs[0, :3])
        ax_kpi     = fig.add_subplot(gs[0, 3])
        ax_dd      = fig.add_subplot(gs[1, :2])
        ax_scatter = fig.add_subplot(gs[1, 2:])
        ax_monthly = fig.add_subplot(gs[2, :2])
        ax_hist    = fig.add_subplot(gs[2, 2:])
        ax_costs   = fig.add_subplot(gs[3, :2])
        ax_exits   = fig.add_subplot(gs[3, 2])
        ax_rolling = fig.add_subplot(gs[3, 3])

        for ax in [ax_equity, ax_kpi, ax_dd, ax_scatter,
                   ax_monthly, ax_hist, ax_costs, ax_exits, ax_rolling]:
            _ax_style(ax)

        self._draw_equity(ax_equity, report)
        self._draw_kpi_card(ax_kpi, report)
        self._draw_drawdown(ax_dd, report)
        self._draw_scatter(ax_scatter, report)
        self._draw_monthly_heatmap(ax_monthly, report)
        self._draw_pnl_histogram(ax_hist, report)
        self._draw_cost_waterfall(ax_costs, report)
        self._draw_exit_pie(ax_exits, report)
        self._draw_rolling_sharpe(ax_rolling, report)

        # ── Title banner ──────────────────────────────────────────────────────
        roi_col = GREEN if report.roi_pct >= 0 else RED
        sign    = "+" if report.roi_pct >= 0 else ""
        fig.suptitle(
            f"PolyBot Backtest Report  ·  {report.question[:55]}  ·  "
            f"ROI {sign}{report.roi_pct:.2f}%  ·  "
            f"Sharpe {report.sharpe_ratio:.2f}  ·  "
            f"Win rate {report.win_rate:.1%}  ·  "
            f"{report.total_trades} trades",
            color=TEXT, fontsize=12, fontweight="bold", y=0.96,
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"\n  Dashboard saved: {save_path}\n")
        return save_path

    def save_summary(self, report, path: str = "results/summary.json"):
        """Export the report's key metrics to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "condition_id":   report.condition_id,
            "question":       report.question,
            "initial_capital": report.initial_capital,
            "final_capital":  report.final_capital,
            "net_pnl":        report.net_pnl,
            "roi_pct":        report.roi_pct,
            "total_trades":   report.total_trades,
            "win_rate":       round(report.win_rate, 4),
            "sharpe_ratio":   report.sharpe_ratio,
            "sortino_ratio":  report.sortino_ratio,
            "calmar_ratio":   report.calmar_ratio,
            "max_drawdown":   report.max_drawdown,
            "max_drawdown_usd": report.max_drawdown_usd,
            "profit_factor":  report.profit_factor,
            "expectancy":     report.expectancy,
            "total_fees":     report.total_fees,
            "total_slippage": report.total_slippage,
            "fee_drag_pct":   report.fee_drag_pct,
            "slip_drag_pct":  report.slip_drag_pct,
            "avg_edge":       report.avg_edge,
            "avg_confidence": report.avg_confidence,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Summary JSON: {path}")
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Individual panel renderers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_equity(self, ax, report):
        """Panel 1: Equity curve with underwater shading."""
        eq   = np.array(report.equity_curve, dtype=float)
        init = report.initial_capital
        x    = np.arange(len(eq))

        ax.plot(x, eq, color=BLUE, linewidth=1.8, zorder=3, label="Portfolio equity")
        ax.axhline(init, color=MUTED, linestyle="--", linewidth=0.8, zorder=2, label="Starting capital")

        # Green fill above starting capital, red below
        ax.fill_between(x, init, eq, where=(eq >= init), alpha=0.18, color=GREEN, zorder=1)
        ax.fill_between(x, init, eq, where=(eq <  init), alpha=0.22, color=RED,   zorder=1)

        # Mark individual trades
        if report.trades:
            entry_bars = []
            exit_bars  = []
            for t in report.trades:
                try:
                    entry_bars.append(int(t.entry_time) if t.entry_time.isdigit() else None)
                    exit_bars.append(int(t.exit_time)   if t.exit_time.isdigit()  else None)
                except Exception:
                    pass

        ax.set_title("Equity Curve  ·  Capital at Risk Overlay")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))
        ax.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, fontsize=8)

        # Annotate final value
        final = eq[-1] if len(eq) else init
        color = GREEN if final >= init else RED
        ax.annotate(
            f"Final: ${final:.4f}",
            xy=(len(eq) - 1, final),
            xytext=(-60, 12 if final >= init else -16),
            textcoords="offset points",
            color=color, fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    def _draw_kpi_card(self, ax, report):
        """Panel 2: KPI scorecard."""
        ax.axis("off")
        ax.set_title("Key Metrics", pad=6)

        def kpi_color(val, good_threshold, reverse=False):
            ok = val >= good_threshold if not reverse else val <= good_threshold
            return GREEN if ok else RED

        metrics = [
            ("Net PnL",       f"${report.net_pnl:+.4f}",       kpi_color(report.net_pnl, 0)),
            ("ROI",           f"{report.roi_pct:+.2f}%",        kpi_color(report.roi_pct, 0)),
            ("Win Rate",      f"{report.win_rate:.1%}",          kpi_color(report.win_rate, 0.55)),
            ("Sharpe",        f"{report.sharpe_ratio:.3f}",      kpi_color(report.sharpe_ratio, 0.5)),
            ("Sortino",       f"{report.sortino_ratio:.3f}",     kpi_color(report.sortino_ratio, 0.5)),
            ("Calmar",        f"{report.calmar_ratio:.3f}",      kpi_color(report.calmar_ratio, 0.3)),
            ("Max DD",        f"{report.max_drawdown:.2%}",      kpi_color(report.max_drawdown, 0.10, reverse=True)),
            ("Max DD ($)",    f"${report.max_drawdown_usd:.4f}", kpi_color(report.max_drawdown_usd, report.initial_capital * 0.10, reverse=True)),
            ("Profit Factor", f"{report.profit_factor:.3f}",     kpi_color(report.profit_factor, 1.2)),
            ("Expectancy",    f"${report.expectancy:+.4f}",      kpi_color(report.expectancy, 0)),
            ("Avg Edge",      f"{report.avg_edge:.4f}",          kpi_color(report.avg_edge, 0.06)),
            ("Fee Drag",      f"{report.fee_drag_pct:.1f}%",     kpi_color(report.fee_drag_pct, 30, reverse=True)),
            ("Slip Drag",     f"{report.slip_drag_pct:.1f}%",    kpi_color(report.slip_drag_pct, 20, reverse=True)),
            ("Total Trades",  str(report.total_trades),          TEXT),
        ]

        y_start = 1.0
        spacing = 1.0 / (len(metrics) + 1)
        for i, (label, value, color) in enumerate(metrics):
            y = y_start - (i + 1) * spacing
            ax.text(0.02, y, label, transform=ax.transAxes,
                    color=MUTED, fontsize=8, va="center")
            ax.text(0.98, y, value, transform=ax.transAxes,
                    color=color, fontsize=8.5, va="center", ha="right", fontweight="bold")

    def _draw_drawdown(self, ax, report):
        """Panel 3: Drawdown waterfall."""
        eq   = np.array(report.equity_curve, dtype=float)
        if len(eq) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("Drawdown")
            return

        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / np.where(peak > 0, peak, 1.0) * 100
        x    = np.arange(len(dd))

        ax.fill_between(x, 0, dd, color=RED, alpha=0.7, zorder=2)
        ax.plot(x, dd, color=RED, linewidth=0.8, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)

        # Max drawdown annotation
        mdd_idx = int(np.argmin(dd))
        ax.annotate(
            f"Max DD: {dd[mdd_idx]:.2f}%",
            xy=(mdd_idx, dd[mdd_idx]),
            xytext=(10, -20), textcoords="offset points",
            color=AMBER, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.7),
        )

        ax.set_title("Drawdown (%)")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.set_ylim(top=2)

    def _draw_scatter(self, ax, report):
        """Panel 4: Trade scatter — entry price vs net PnL."""
        if not report.trades:
            ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("Trade P&L Scatter")
            return

        entries = [t.entry_price for t in report.trades]
        pnls    = [t.net_pnl     for t in report.trades]
        confs   = [max(0.3, t.confidence) for t in report.trades]
        colors  = [GREEN if p > 0 else RED for p in pnls]
        sizes   = [80 * c for c in confs]

        sc = ax.scatter(entries, pnls, c=colors, s=sizes, alpha=0.75,
                        edgecolors="none", zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.7, linestyle="--", zorder=2)

        # Regression line
        if len(entries) >= 3:
            try:
                z = np.polyfit(entries, pnls, 1)
                p = np.poly1d(z)
                xs = np.linspace(min(entries), max(entries), 50)
                ax.plot(xs, p(xs), color=CYAN, linewidth=1.2, alpha=0.6, zorder=2)
            except Exception:
                pass

        ax.set_title("Trade P&L Scatter\n(size = confidence, colour = win/loss)")
        ax.set_xlabel("Entry Price")
        ax.set_ylabel("Net PnL (USD)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))

        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                mpatches.Patch(color=GREEN, label=f"Win ({sum(1 for p in pnls if p>0)})"),
                mpatches.Patch(color=RED,   label=f"Loss ({sum(1 for p in pnls if p<=0)})"),
            ],
            facecolor=PANEL, labelcolor=TEXT, fontsize=8,
        )

    def _draw_monthly_heatmap(self, ax, report):
        """Panel 5: Monthly returns heatmap."""
        if not report.trades:
            ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("Monthly Returns")
            return

        # Build monthly returns from trades (use exit time if parseable)
        monthly: dict[tuple, float] = {}
        for t in report.trades:
            try:
                dt = pd.to_datetime(t.exit_time)
                key = (dt.year, dt.month)
                monthly[key] = monthly.get(key, 0.0) + t.net_pnl
            except Exception:
                pass

        if not monthly:
            ax.text(0.5, 0.5, "Cannot parse trade timestamps",
                    transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_title("Monthly Returns")
            return

        keys   = sorted(monthly.keys())
        years  = sorted(set(k[0] for k in keys))
        months = list(range(1, 13))
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]

        matrix = np.full((len(years), 12), np.nan)
        for (y, m), pnl in monthly.items():
            if y in years:
                matrix[years.index(y), m - 1] = pnl

        # Custom diverging colormap
        cmap = LinearSegmentedColormap.from_list("rg", [RED, PANEL, GREEN])
        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1

        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=7.5)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=7.5)

        for yi in range(len(years)):
            for mi in range(12):
                val = matrix[yi, mi]
                if not np.isnan(val):
                    color = TEXT if abs(val) < vmax * 0.6 else BG
                    ax.text(mi, yi, f"${val:+.3f}", ha="center", va="center",
                            fontsize=5.5, color=color)

        plt.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label="Net PnL (USD)")
        ax.set_title("Monthly Returns Heatmap")

    def _draw_pnl_histogram(self, ax, report):
        """Panel 6: Win/Loss PnL distribution."""
        if not report.trades:
            ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("PnL Distribution")
            return

        pnls  = [t.net_pnl for t in report.trades]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        bins = max(10, min(30, len(pnls) // 3))
        ax.hist(wins,   bins=bins, color=GREEN, alpha=0.75, label=f"Wins ({len(wins)})",  edgecolor="none")
        ax.hist(losses, bins=bins, color=RED,   alpha=0.75, label=f"Losses ({len(losses)})", edgecolor="none")
        ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--")

        if pnls:
            mean_pnl = float(np.mean(pnls))
            ax.axvline(mean_pnl, color=AMBER, linewidth=1.2, linestyle=":",
                       label=f"Mean: ${mean_pnl:+.4f}")

        ax.set_title("PnL Distribution (Net, after fees & slippage)")
        ax.set_xlabel("Net PnL per trade (USD)")
        ax.set_ylabel("Count")
        ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))

    def _draw_cost_waterfall(self, ax, report):
        """Panel 7: Waterfall — gross PnL → fees → slippage → net PnL."""
        labels = ["Gross PnL", "- Fees", "- Slippage", "Net PnL"]
        values = [
            report.gross_pnl,
            -abs(report.total_fees),
            -abs(report.total_slippage),
            report.net_pnl,
        ]
        colors = [
            GREEN if report.gross_pnl >= 0 else RED,
            RED,
            AMBER,
            GREEN if report.net_pnl >= 0 else RED,
        ]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.55, edgecolor=BORDER, linewidth=0.8)

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + max(abs(v) for v in values) * 0.03
            y_pos = bar.get_y() + bar.get_height()
            offset = 4 if val >= 0 else -14
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"${val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
                    color=TEXT, fontsize=8, fontweight="bold")

        # Draw connector lines
        cumsum = 0.0
        for i, v in enumerate(values[:-1]):
            cumsum += v
            ax.plot([i + 0.275, i + 0.725], [cumsum, cumsum],
                    color=MUTED, linewidth=0.8, linestyle=":", zorder=4)

        ax.axhline(0, color=MUTED, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title("Cost Breakdown Waterfall")
        ax.set_ylabel("USD")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))

    def _draw_exit_pie(self, ax, report):
        """Panel 8: Exit reason breakdown."""
        if not report.trades:
            ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("Exit Reasons")
            return

        from collections import Counter
        counts = Counter(t.exit_reason for t in report.trades)
        labels = list(counts.keys())
        sizes  = list(counts.values())

        palette = [GREEN, RED, AMBER, BLUE, PURPLE, CYAN]
        colors  = [palette[i % len(palette)] for i in range(len(labels))]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": TEXT, "fontsize": 7.5},
            pctdistance=0.75,
            wedgeprops={"edgecolor": BG, "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color(BG)
            at.set_fontweight("bold")

        # Legend with counts
        legend_labels = [f"{l}: {c}" for l, c in counts.items()]
        ax.legend(
            handles=[mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels)],
            loc="lower center", bbox_to_anchor=(0.5, -0.18),
            facecolor=PANEL, labelcolor=TEXT, fontsize=6.5, ncol=2,
        )
        ax.set_title("Exit Reason Breakdown")

    def _draw_rolling_sharpe(self, ax, report, window: int = 15):
        """Panel 9: Rolling Sharpe ratio (trade-window)."""
        if len(report.trades) < window + 2:
            ax.text(0.5, 0.5, f"Need >{window} trades\nfor rolling Sharpe",
                    transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_title(f"Rolling Sharpe ({window}-trade)")
            return

        rets = [t.net_return for t in report.trades]
        rolling_s = []
        for i in range(window, len(rets) + 1):
            sub = rets[max(0, i - window): i]
            arr = np.array(sub)
            std = arr.std(ddof=1)
            rs  = (arr.mean() / std * math.sqrt(252)) if std > 1e-9 else 0.0
            rolling_s.append(rs)

        x = np.arange(window, len(rets) + 1)
        rs_arr = np.array(rolling_s)

        ax.plot(x, rs_arr, color=PURPLE, linewidth=1.5, zorder=3)
        ax.fill_between(x, 0, rs_arr, where=(rs_arr >= 0), alpha=0.18, color=GREEN)
        ax.fill_between(x, 0, rs_arr, where=(rs_arr < 0),  alpha=0.18, color=RED)
        ax.axhline(0,   color=MUTED, linewidth=0.6, linestyle="--")
        ax.axhline(0.5, color=GREEN, linewidth=0.6, linestyle=":", alpha=0.5,  label="0.5 threshold")
        ax.axhline(1.0, color=GREEN, linewidth=0.6, linestyle=":", alpha=0.35, label="1.0 target")

        ax.set_title(f"Rolling Sharpe Ratio ({window}-trade window)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Sharpe (annualised)")
        ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-market comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_comparison(
    reports: list,
    save_path: str = "plots/portfolio_comparison.png",
):
    """
    Generate a 2×2 portfolio-level comparison across all backtested markets.
    Shows: ROI bar, Sharpe bar, Win Rate scatter, Net PnL vs Max Drawdown scatter.
    """
    valid = [r for r in reports if r.total_trades > 0]
    if not valid:
        print("  No valid reports to compare.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax in axes.flat:
        _ax_style(ax)

    labels  = [r.condition_id[:12] for r in valid]
    rois    = [r.roi_pct          for r in valid]
    sharpes = [r.sharpe_ratio     for r in valid]
    wrs     = [r.win_rate         for r in valid]
    pnls    = [r.net_pnl          for r in valid]
    mdds    = [r.max_drawdown     for r in valid]

    # ── ROI bar ────────────────────────────────────────────────────────────
    ax0 = axes[0, 0]
    cols = [GREEN if r >= 0 else RED for r in rois]
    bars = ax0.bar(labels, rois, color=cols, width=0.6, edgecolor=BORDER)
    ax0.axhline(0, color=MUTED, linewidth=0.7)
    for bar, val in zip(bars, rois):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3 * (1 if val >= 0 else -1),
                 f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 color=TEXT, fontsize=7)
    ax0.set_title("ROI by Market (%)")
    ax0.tick_params(axis="x", rotation=35, labelsize=7)
    ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # ── Sharpe bar ─────────────────────────────────────────────────────────
    ax1 = axes[0, 1]
    s_cols = [BLUE if s >= 0.5 else (AMBER if s >= 0 else RED) for s in sharpes]
    ax1.bar(labels, sharpes, color=s_cols, width=0.6, edgecolor=BORDER)
    ax1.axhline(0.5, color=GREEN, linewidth=0.8, linestyle=":", label="0.5 threshold")
    ax1.axhline(0,   color=MUTED, linewidth=0.7)
    ax1.set_title("Sharpe Ratio by Market")
    ax1.tick_params(axis="x", rotation=35, labelsize=7)
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)

    # ── Win rate dots ──────────────────────────────────────────────────────
    ax2 = axes[1, 0]
    wr_colors = [GREEN if w >= 0.55 else (AMBER if w >= 0.50 else RED) for w in wrs]
    ax2.scatter(labels, [w * 100 for w in wrs], c=wr_colors, s=60,
                edgecolors="none", zorder=3)
    ax2.axhline(55, color=GREEN, linewidth=0.8, linestyle=":", label="55% target")
    ax2.axhline(50, color=MUTED, linewidth=0.7, linestyle="--")
    for i, (l, w) in enumerate(zip(labels, wrs)):
        ax2.text(i, w * 100 + 1.2, f"{w:.0%}", ha="center", va="bottom",
                 color=TEXT, fontsize=6.5)
    ax2.set_title("Win Rate by Market (%)")
    ax2.tick_params(axis="x", rotation=35, labelsize=7)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)

    # ── Net PnL vs Max Drawdown ─────────────────────────────────────────────
    ax3 = axes[1, 1]
    mdd_pct = [m * 100 for m in mdds]
    dot_colors = [GREEN if p >= 0 else RED for p in pnls]
    sc = ax3.scatter(mdd_pct, pnls, c=dot_colors, s=80, edgecolors=BORDER, linewidth=0.5, zorder=3)
    for i, l in enumerate(labels):
        ax3.annotate(l, (mdd_pct[i], pnls[i]), fontsize=6, color=MUTED,
                     xytext=(4, 4), textcoords="offset points")
    ax3.axhline(0, color=MUTED, linewidth=0.6, linestyle="--")
    ax3.set_title("Net PnL vs Max Drawdown")
    ax3.set_xlabel("Max Drawdown (%)")
    ax3.set_ylabel("Net PnL (USD)")
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))

    fig.suptitle("Portfolio Backtest Comparison", color=TEXT, fontsize=13, fontweight="bold", y=0.98)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Portfolio comparison saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

visualiser = BacktestVisualiser()
