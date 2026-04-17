"""
main.py — PolyBot v2 Entry Point
=================================
Orchestrates the full trading loop:
  1. Fetch live markets (Data Layer)
  2. Store historical snapshots
  3. Run strategy engine (all 6 signals)
  4. Risk gate (liquidity, confidence, capital, daily limits)
  5. Execute trades (paper or live)
  6. Monitor open positions (stop-loss / take-profit / reversal)
  7. Track performance metrics

Usage:
    python main.py           # run the live (or paper) trading bot
    python main.py test      # connection test only
    python main.py backtest  # run backtests on all stored data
    python main.py metrics   # print current performance summary
"""

import sys
import time
import os

# Windows UTF-8 safety (must be before any import that may print)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from polybot.logger      import get_logger
from polybot.config      import (
    CAPITAL, PAPER_MODE, SCAN_INTERVAL, MAX_MARKETS_SCAN, TOP_MARKETS_TRADE,
    CLOB_HOST, CLOB_API_KEY, GEMINI_API_KEY, GAMMA_HOST,
)
from polybot.data_layer  import fetch_active_markets, build_snapshot, store
from polybot.strategies  import run_all_strategies
from polybot.risk_manager import risk, state
from polybot.executor    import executor
from polybot.backtester  import backtester, Backtester
from polybot.metrics     import perf

log = get_logger("main")


# ── Connection Test ───────────────────────────────────────────────────────────

def test_connections():
    import requests
    W = "\033[33m"; G = "\033[32m"; R = "\033[31m"; E = "\033[0m"
    sep = "=" * 60

    print(f"\n{sep}")
    print("  PolyBot v2 — Connection Test")
    print(f"  Capital: ${CAPITAL:.2f} | Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
    print(sep)

    # 1. Gamma API (public)
    print("\n[1] Polymarket Gamma API (market data)...")
    try:
        r = requests.get(f"{GAMMA_HOST}/markets", params={"limit": 1}, timeout=8)
        if r.status_code == 200:
            print(f"    {G}[OK]{E}   Gamma API reachable")
        else:
            print(f"    {W}[WARN]{E} HTTP {r.status_code}")
    except Exception as e:
        print(f"    {R}[FAIL]{E} {e}")

    # 2. CLOB API
    print("\n[2] Polymarket CLOB API (trading)...")
    try:
        r = requests.get(
            f"{CLOB_HOST}/markets", params={"limit": 1},
            headers={"Authorization": f"Bearer {CLOB_API_KEY}"}, timeout=8,
        )
        if r.status_code == 200:
            print(f"    {G}[OK]{E}   CLOB API reachable")
        elif r.status_code == 401:
            print(f"    {W}[WARN]{E} CLOB connected but API key invalid (401)")
        else:
            print(f"    {W}[WARN]{E} HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"    {R}[FAIL]{E} {e}")

    # 3. Gemini
    print("\n[3] Google Gemini API (news/ML)...")
    if not GEMINI_API_KEY:
        print(f"    {W}[SKIP]{E} No GEMINI_API_KEY set")
    else:
        try:
            from google import genai
            from google.genai import types as gt
            client = genai.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Say OK",
                config=gt.GenerateContentConfig(max_output_tokens=5),
            )
            if resp.text:
                print(f"    {G}[OK]{E}   Gemini API responding")
            else:
                print(f"    {W}[WARN]{E} Empty response")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"    {G}[OK]{E}   Gemini reachable (quota exceeded)")
            else:
                print(f"    {R}[FAIL]{E} {e}")

    # 4. Reddit
    print("\n[4] Reddit API (public, no key)...")
    try:
        r = requests.get(
            "https://www.reddit.com/r/news/new.json",
            headers={"User-Agent": "polybot/2.0"}, timeout=8,
        )
        if r.status_code == 200:
            print(f"    {G}[OK]{E}   Reddit accessible")
        else:
            print(f"    {W}[WARN]{E} HTTP {r.status_code}")
    except Exception as e:
        print(f"    {R}[FAIL]{E} {e}")

    print(f"\n{sep}\n")


# ── Main Trading Loop ─────────────────────────────────────────────────────────

def trading_cycle():
    """Execute one full scan-analyse-trade-monitor cycle."""
    # Daily limit check
    ok, reason = state.can_trade()
    if not ok:
        log.info(f"Trading paused: {reason}")
        return

    log.info(
        f"=== Scan cycle start | P&L: ${state.realised_pnl:+.2f} | "
        f"Open: {len(state.open_positions)} | Trades: {state.trades} ==="
    )

    # ── Step 1: Monitor all open positions ───────────────────────────────────
    exits = executor.monitor_positions()
    for e in exits:
        # Use richer position data when available (entry/exit/size from executor dict)
        perf.record(
            question     = e.get("question", e.get("reason", "?")),
            side         = e.get("side", "EXIT"),
            entry_price  = e.get("entry_price", 0.0),
            exit_price   = e.get("exit_price",  0.0),
            size_dollars = e.get("size_dollars", max(abs(e["pnl"]), 0.01)),
            pnl          = e["pnl"],
            exit_reason  = e["reason"],
        )

    # ── Step 2: Fetch fresh markets ───────────────────────────────────────────
    raw_markets = fetch_active_markets(limit=MAX_MARKETS_SCAN)
    if not raw_markets:
        log.warning("No markets returned from API — check connectivity.")
        return

    log.info(f"Fetched {len(raw_markets)} markets from Gamma API")

    # ── Step 3: Build snapshots + save history ────────────────────────────────
    snapshots = []
    for raw in raw_markets[:TOP_MARKETS_TRADE]:
        snap = build_snapshot(raw)
        if snap is None:
            continue
        store.append(snap)           # persist to CSV
        snapshots.append(snap)

    log.info(f"Built {len(snapshots)} enriched snapshots")

    # ── Step 4: Analyse each market ───────────────────────────────────────────
    for snap in snapshots:
        # Skip if we already hold this market
        if snap.condition_id in state.open_positions:
            pos = state.open_positions[snap.condition_id]
            # Check for signal reversal
            history  = store.load(snap.condition_id)
            combined = run_all_strategies(snap, history)
            executor.check_signal_reversal(pos, combined)
            continue

        # Skip illiquid / untradeable markets
        if not snap.is_tradeable:
            log.debug(f"SKIP (not tradeable): {snap.question[:50]}")
            continue

        # Load historical data for this market
        history  = store.load(snap.condition_id)

        # Run all signals
        combined = run_all_strategies(snap, history)

        log.info(
            f"  [{snap.question[:45]}]\n"
            f"    price={snap.mid_price:.3f} | model={combined.yes_probability:.3f} | "
            f"edge={combined.edge:+.4f} | conf={combined.confidence:.3f} | "
            f"dir={combined.direction}"
        )

        # ── Step 5: Pre-trade gates ───────────────────────────────────────────
        ok, reason = risk.check_pre_trade(snap, combined)
        if not ok:
            log.info(f"    SKIP: {reason}")
            continue

        # ── Step 6: Size using Kelly ──────────────────────────────────────────
        win_prob     = combined.yes_probability if combined.direction == "BUY_YES" else 1 - combined.yes_probability
        size_dollars = risk.kelly_size(abs(combined.edge), win_prob)
        if size_dollars < 0.10:
            log.info(f"    SKIP: position too small (${size_dollars:.2f})")
            continue

        # ── Step 7: Execute entry ─────────────────────────────────────────────
        position = executor.enter_position(snap, combined, size_dollars)
        if position:
            log.info(
                f"    TRADE ENTERED | {combined.direction} | "
                f"${size_dollars:.2f} @ {position.entry_price:.4f} | "
                f"SL={position.stop_loss:.4f} TP={position.take_profit:.4f}"
            )

        # Throttle between market analyses (Gemini rate limit awareness)
        time.sleep(6)

    state.save_state()
    log.info(f"=== Cycle complete | {risk.summary()} ===")


# ── Backtest runner ────────────────────────────────────────────────────────────

def run_backtest():
    """Run backtests on all stored market data and print reports."""
    print("\n" + "=" * 60)
    print("  PolyBot v2 — Backtest Mode")
    print("=" * 60)
    markets = store.list_markets()
    if not markets:
        print("  No historical data found. Run the bot first to collect data.")
        print("  Tip: data is stored in the 'data/' directory as CSV files.")
        return
    print(f"  Found {len(markets)} tracked market(s)\n")
    agg_pnl, agg_trades = 0.0, 0
    for cid in markets:
        df = store.load(cid)
        question = df.attrs.get("question", cid)       # stored only if we had it
        report = backtester.run(cid, question=f"Market {cid[:20]}")
        backtester.print_report(report)
        agg_pnl    += report.total_pnl
        agg_trades += report.total_trades
    print(f"\nAggregate: {agg_trades} trades | Total P&L: ${agg_pnl:+.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "run"

    if cmd == "test":
        test_connections()

    elif cmd == "backtest":
        run_backtest()

    elif cmd == "metrics":
        perf.print_summary()
        print("\nRecent trades:")
        for t in perf.recent_trades(10):
            print(f"  {t['time'][:19]}  {t['side']:<12} {t['pnl']:<10} {t['q']}")

    else:
        # Normal trading mode
        log.info(
            f"\n{'='*60}\n"
            f"  PolyBot v2 — Starting\n"
            f"  Capital: ${CAPITAL:.2f} | Mode: {'PAPER (no real money)' if PAPER_MODE else '*** LIVE — REAL MONEY ***'}\n"
            f"  Scan interval: {SCAN_INTERVAL}s | Max markets: {TOP_MARKETS_TRADE}\n"
            f"{'='*60}"
        )

        # Restore previous session state
        state.load_state()
        test_connections()

        try:
            while True:
                try:
                    trading_cycle()
                    log.info(f"Sleeping {SCAN_INTERVAL}s until next cycle...")
                    time.sleep(SCAN_INTERVAL)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error(f"Cycle error: {exc}", exc_info=True)
                    log.info("Sleeping 30s before retry...")
                    time.sleep(30)
        except KeyboardInterrupt:
            log.info("\nBot stopped by user.")
            log.info(f"Session summary: {risk.summary()}")
            perf.print_summary()
            state.save_state()
