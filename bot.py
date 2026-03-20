"""
PolyBot — Polymarket Auto-Trading Bot
Capital: $10 | Daily profit limit: 50% (+$5) | Daily loss limit: 15% (-$1.50)
Stop loss: 8% per trade | Multi-signal: News + Twitter + Reddit + ML + Statistical + Base rate
"""

import os, time, json, asyncio, logging, hashlib, hmac, requests, sys
from google import genai
from google.genai import types as genai_types
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Force UTF-8 output on Windows consoles (avoids UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("polybot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("polybot")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAPITAL            = float(os.getenv("CAPITAL", "10.0"))
DAILY_PROFIT_LIMIT = CAPITAL * 0.50   # +$5.00
DAILY_LOSS_LIMIT   = CAPITAL * 0.15   # -$1.50
STOP_LOSS_PCT      = 0.08             # 8% per trade
MAX_POSITION_SIZE  = CAPITAL * 0.20   # $2.00 max per trade
MIN_CONFIDENCE     = 0.75             # 75%
MIN_EDGE           = 0.08             # 8% edge over market price
SCAN_INTERVAL      = 15               # seconds between scans

POLYMARKET_HOST    = "https://clob.polymarket.com"
GEMINI_MODEL       = "gemini-1.5-flash-latest"   # confirmed working with google-genai v1 SDK

CLOB_API_KEY       = os.getenv("CLOB_API_KEY", "")
CLOB_SECRET        = os.getenv("CLOB_SECRET", "")
CLOB_PASSPHRASE    = os.getenv("CLOB_PASSPHRASE", "")
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
TWITTER_BEARER     = os.getenv("TWITTER_BEARER", "")
PAPER_MODE         = os.getenv("PAPER_MODE", "true").lower() == "true"

# Initialise Gemini client (new google-genai SDK)
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── DAILY RISK STATE ──────────────────────────────────────────────────────────
@dataclass
class DailyState:
    date: date = field(default_factory=date.today)
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    open_positions: dict = field(default_factory=dict)

    def reset_if_new_day(self):
        if date.today() != self.date:
            log.info(f"New day — resetting daily state. Yesterday P&L: ${self.pnl:.2f}")
            self.date = date.today()
            self.pnl = 0.0
            self.trades = 0
            self.wins = 0

    def can_trade(self) -> tuple[bool, str]:
        self.reset_if_new_day()
        if self.pnl >= DAILY_PROFIT_LIMIT:
            return False, f"Daily profit limit hit (+${self.pnl:.2f}). Paused for today."
        if self.pnl <= -DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit hit (${self.pnl:.2f}). Stopped for today."
        return True, "OK"

    def kelly_size(self, edge: float, win_prob: float) -> float:
        """Kelly criterion: f = (bp - q) / b  where b=edge odds, p=win prob, q=1-p"""
        if edge <= 0 or win_prob <= 0:
            return 0.0
        b = edge / (1 - edge)
        q = 1 - win_prob
        kelly = (b * win_prob - q) / b
        kelly = max(0, min(kelly, 0.25))   # cap at 25% of capital
        size = round(CAPITAL * kelly, 2)
        return min(size, MAX_POSITION_SIZE)

state = DailyState()

# ── SIGNAL 1: NEWS ────────────────────────────────────────────────────────────
def signal_news(question: str) -> dict:
    """Fetch recent news via Gemini Google Search grounding and score YES probability."""
    system_prompt = (
        "You are a news analyst for prediction markets. "
        "Search for recent news about the topic and return ONLY a JSON object: "
        '{"score": <0-100 int>, "summary": "<10 words max>"}. '
        "Score = probability YES resolves based on news. No extra text, just JSON."
    )
    for attempt in range(3):   # retry up to 3× for quota / transient errors
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Analyze news for this prediction market question and return JSON only: {question}",
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                    max_output_tokens=300,
                )
            )
            text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            return {"score": int(result.get("score", 50)) / 100, "summary": result.get("summary", ""), "ok": True}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 15 * (2 ** attempt)   # 15s → 30s → 60s
                log.warning(f"Gemini quota hit — waiting {wait}s before retry {attempt + 1}/3")
                time.sleep(wait)
            else:
                log.warning(f"News signal error: {e}")
                break
    return {"score": 0.5, "summary": "unavailable", "ok": False}

# ── SIGNAL 2: TWITTER SENTIMENT ─────────────────────────────────────────────────
def signal_twitter(question: str) -> dict:
    """
    Twitter keyword search requires a paid API tier (free tier returns 403).
    Returns a neutral 0.5 stub — weight redistributed to other signals below.
    """
    return {"score": 0.5, "count": 0, "ok": False}

# ── SIGNAL 3: REDDIT SENTIMENT ────────────────────────────────────────────────
def signal_reddit(question: str) -> dict:
    """Search Reddit for relevant posts and score sentiment."""
    try:
        keywords = question[:60].replace("?", "").replace("Will ", "")
        url = f"https://www.reddit.com/search.json"
        headers = {"User-Agent": "polybot/1.0"}
        params = {"q": keywords, "sort": "new", "limit": 25, "t": "week"}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        posts = r.json().get("data", {}).get("children", [])
        if not posts:
            return {"score": 0.5, "count": 0, "ok": False}

        pos_words = {"yes","likely","will","confirmed","bullish","positive","up","win"}
        neg_words = {"no","unlikely","won't","bearish","negative","down","lose","rejected"}
        scores = []
        for p in posts:
            d = p["data"]
            text = (d.get("title","") + " " + d.get("selftext","")).lower()
            words = set(text.split())
            pos = len(words & pos_words)
            neg = len(words & neg_words)
            if pos + neg > 0:
                upvote_w = min(2.0, 1 + d.get("score", 0) / 500)
                scores.append((pos / (pos + neg)) * upvote_w)
        avg = sum(scores) / len(scores) if scores else 0.5
        return {"score": round(min(1.0, avg), 3), "count": len(posts), "ok": True}
    except Exception as e:
        log.warning(f"Reddit signal error: {e}")
        return {"score": 0.5, "count": 0, "ok": False}

# ── SIGNAL 4: ML ENSEMBLE ─────────────────────────────────────────────────────
def signal_ml(question: str, market_price: float, days_to_resolve: int) -> dict:
    """
    Lightweight ML-style ensemble using heuristic features.
    Replace with real GBM/LSTM model by loading a trained .pkl or .pt file.
    """
    try:
        # Feature 1: market price itself is a strong prior
        f_price = market_price

        # Feature 2: time decay — markets closer to resolution are more certain
        f_time = 1.0 - min(1.0, days_to_resolve / 180)

        # Feature 3: question category heuristics
        q_lower = question.lower()
        f_category = 0.5
        if any(w in q_lower for w in ["fed", "rate", "inflation", "gdp"]):
            f_category = 0.55   # macro events slight YES bias
        elif any(w in q_lower for w in ["bitcoin", "btc", "crypto", "eth"]):
            f_category = 0.48   # crypto slight NO bias (volatile)
        elif any(w in q_lower for w in ["election", "vote", "president"]):
            f_category = 0.50   # elections neutral
        elif any(w in q_lower for w in ["apple", "google", "microsoft", "product"]):
            f_category = 0.60   # tech product launches high

        # Weighted ensemble
        score = 0.50 * f_price + 0.20 * f_time + 0.30 * f_category
        score = round(min(0.99, max(0.01, score)), 3)
        return {"score": score, "features": {"price": f_price, "time": f_time, "category": f_category}, "ok": True}
    except Exception as e:
        log.warning(f"ML signal error: {e}")
        return {"score": 0.5, "ok": False}

# ── SIGNAL 5: STATISTICAL MODEL ───────────────────────────────────────────────
def signal_statistical(market_price: float, volume: float) -> dict:
    """
    Checks if market is mispriced using volume-weighted calibration.
    High-volume markets are better calibrated; low-volume = more edge opportunity.
    """
    try:
        # Volume calibration factor — low volume = less efficient = more edge
        vol_factor = max(0.5, min(2.0, 1_000_000 / max(volume, 1)))
        # Price extremes are often mispriced (overreaction)
        if market_price < 0.10:
            adj = market_price * 1.15 * vol_factor   # market may under-price rare events
        elif market_price > 0.90:
            adj = market_price * 0.95 * vol_factor   # market may over-price favorites
        else:
            adj = market_price * (1 + (0.5 - market_price) * 0.10 * vol_factor)
        adj = round(min(0.99, max(0.01, adj)), 3)
        edge = round(adj - market_price, 3)
        return {"score": adj, "edge": edge, "vol_factor": round(vol_factor, 2), "ok": True}
    except Exception as e:
        log.warning(f"Statistical signal error: {e}")
        return {"score": market_price, "edge": 0.0, "ok": False}

# ── SIGNAL 6: HISTORICAL BASE RATE ───────────────────────────────────────────
def signal_base_rate(question: str) -> dict:
    """Historical YES resolution rates by category."""
    base_rates = {
        "fed":        0.48,
        "rate":       0.48,
        "bitcoin":    0.42,
        "btc":        0.42,
        "crypto":     0.40,
        "election":   0.50,
        "president":  0.50,
        "apple":      0.62,
        "iphone":     0.65,
        "google":     0.60,
        "microsoft":  0.60,
        "war":        0.35,
        "recession":  0.30,
        "inflation":  0.55,
        "default":    0.50,
    }
    q_lower = question.lower()
    for keyword, rate in base_rates.items():
        if keyword in q_lower:
            return {"score": rate, "matched": keyword, "ok": True}
    return {"score": 0.50, "matched": "default", "ok": True}

# Signal weights — Twitter disabled (free-tier 403), its 15% redistributed:
# news +5%, reddit +5%, ml +3%, base_rate +2%
SIGNAL_WEIGHTS = {
    "news":        0.25,   # up from 0.20
    "twitter":     0.00,   # disabled — free tier does not support search
    "reddit":      0.15,   # up from 0.10
    "ml":          0.28,   # up from 0.25
    "statistical": 0.20,
    "base_rate":   0.12,   # up from 0.10
}

def combine_signals(signals: dict) -> dict:
    weighted_sum = sum(SIGNAL_WEIGHTS[k] * signals[k]["score"] for k in SIGNAL_WEIGHTS)
    scores = [signals[k]["score"] for k in SIGNAL_WEIGHTS]
    variance = sum((s - weighted_sum) ** 2 for s in scores) / len(scores)
    # Confidence = inverse of variance (signals agreeing = high confidence)
    confidence = max(0.0, min(1.0, 1.0 - variance * 4))
    return {"yes_prob": round(weighted_sum, 3), "confidence": round(confidence, 3)}

# ── POLYMARKET API ────────────────────────────────────────────────────────────
def get_markets(limit: int = 20) -> list:
    """Fetch active markets from Polymarket CLOB."""
    try:
        headers = {"Authorization": f"Bearer {CLOB_API_KEY}"}
        r = requests.get(f"{POLYMARKET_HOST}/markets", headers=headers,
                         params={"active": True, "closed": False, "limit": limit}, timeout=10)
        return r.json().get("data", [])
    except Exception as e:
        log.warning(f"Failed to fetch markets: {e}")
        return []

def get_market_price(condition_id: str) -> Optional[float]:
    """Get current YES token price for a market."""
    try:
        headers = {"Authorization": f"Bearer {CLOB_API_KEY}"}
        r = requests.get(f"{POLYMARKET_HOST}/midpoint", headers=headers,
                         params={"token_id": condition_id}, timeout=5)
        return float(r.json().get("mid", 0.5))
    except:
        return None

def place_order(token_id: str, side: str, price: float, size: float) -> dict:
    """Place a limit order on Polymarket CLOB."""
    if PAPER_MODE:
        log.info(f"[PAPER] {side} {size:.2f} units @ {price:.2f} | token: {token_id[:12]}...")
        return {"status": "paper_filled", "price": price, "size": size}
    try:
        headers = {
            "Authorization": f"Bearer {CLOB_API_KEY}",
            "CLOB-API-KEY": CLOB_API_KEY,
            "CLOB-SECRET": CLOB_SECRET,
            "CLOB-PASSPHRASE": CLOB_PASSPHRASE,
            "content-type": "application/json"
        }
        order = {
            "tokenID": token_id,
            "price": str(round(price, 4)),
            "size": str(round(size, 2)),
            "side": side.upper(),
            "orderType": "LIMIT",
            "feeRateBps": "0",
            "nonce": str(int(time.time() * 1000)),
        }
        r = requests.post(f"{POLYMARKET_HOST}/order", headers=headers, json=order, timeout=5)
        result = r.json()
        log.info(f"Order placed: {result}")
        return result
    except Exception as e:
        log.error(f"Order placement failed: {e}")
        return {"status": "error", "error": str(e)}

# ── STOP LOSS MONITOR ─────────────────────────────────────────────────────────
def check_stop_losses():
    """Check all open positions and exit if stop loss triggered."""
    for market_id, pos in list(state.open_positions.items()):
        current_price = get_market_price(pos["token_id"])
        if current_price is None:
            continue
        entry = pos["entry_price"]
        sl_price = entry * (1 - STOP_LOSS_PCT)
        if current_price <= sl_price:
            log.warning(f"STOP LOSS triggered | market: {pos['question'][:40]} | entry: {entry:.2f} | current: {current_price:.2f}")
            result = place_order(pos["token_id"], "SELL", current_price, pos["size"])
            loss = (current_price - entry) * pos["size"]
            state.pnl += loss
            state.trades += 1
            del state.open_positions[market_id]
            log.warning(f"Stop loss exit: ${loss:.2f} | daily P&L: ${state.pnl:.2f}")

# ── MAIN SCAN LOOP ────────────────────────────────────────────────────────────
def scan_and_trade():
    """Main trading loop — scan markets, run signals, execute trades."""
    can, reason = state.can_trade()
    if not can:
        log.info(f"Trading paused: {reason}")
        return

    log.info(f"Scanning markets... | daily P&L: ${state.pnl:.2f} | trades: {state.trades}")
    markets = get_markets(limit=20)
    if not markets:
        log.warning("No markets returned from API.")
        return

    check_stop_losses()

    for market in markets[:10]:   # scan top 10
        question    = market.get("question", "")
        token_id    = market.get("conditionId", "")
        volume      = float(market.get("volumeNum", 0))
        days_left   = int(market.get("daysUntilClose", 30))
        market_price = get_market_price(token_id) or float(market.get("bestAsk", 0.5))

        if not question or not token_id:
            continue

        # Run all 6 signals
        signals = {
            "news":        signal_news(question),
            "twitter":     signal_twitter(question),
            "reddit":      signal_reddit(question),
            "ml":          signal_ml(question, market_price, days_left),
            "statistical": signal_statistical(market_price, volume),
            "base_rate":   signal_base_rate(question),
        }
        combined = combine_signals(signals)
        yes_prob   = combined["yes_prob"]
        confidence = combined["confidence"]
        edge       = abs(yes_prob - market_price)
        buy_yes    = yes_prob > market_price

        log.info(f"Market: {question[:50]}")
        log.info(f"  Price: {market_price:.2f} | Model: {yes_prob:.2f} | Edge: {edge:.2f} | Conf: {confidence:.2f}")

        # Trade filter
        if confidence < MIN_CONFIDENCE:
            log.info(f"  SKIP — confidence {confidence:.2f} < {MIN_CONFIDENCE}")
            continue
        if edge < MIN_EDGE:
            log.info(f"  SKIP — edge {edge:.2f} < {MIN_EDGE}")
            continue
        if token_id in state.open_positions:
            log.info(f"  SKIP — already have position")
            continue

        # Size using Kelly criterion
        size_dollars = state.kelly_size(edge, yes_prob if buy_yes else 1 - yes_prob)
        if size_dollars < 0.50:
            log.info(f"  SKIP — position too small (${size_dollars:.2f})")
            continue

        side  = "BUY" if buy_yes else "SELL"
        price = market_price
        size  = round(size_dollars / price, 2)   # convert $ to token units

        log.info(f"  TRADE: {side} | ${size_dollars:.2f} | {size} tokens @ {price:.2f}")
        start = time.time()
        result = place_order(token_id, side, price, size)
        exec_ms = round((time.time() - start) * 1000, 1)
        log.info(f"  Execution: {exec_ms}ms | result: {result.get('status','?')}")

        if result.get("status") in ("paper_filled", "matched", "delayed"):
            state.open_positions[token_id] = {
                "question":    question,
                "token_id":    token_id,
                "entry_price": price,
                "size":        size,
                "side":        side,
                "entry_time":  datetime.now().isoformat(),
                "stop_loss":   round(price * (1 - STOP_LOSS_PCT), 4),
            }
            log.info(f"  Position opened | stop loss @ {state.open_positions[token_id]['stop_loss']:.2f}")

        time.sleep(1)   # small delay between orders to avoid rate limits

# ── CONNECTION TEST ───────────────────────────────────────────────────────────
def test_connections():
    print("\n" + "="*55)
    print("  PolyBot -- Connection Test")
    print("="*55)

    # 1. Polymarket
    print("\n[1] Polymarket CLOB API...")
    try:
        r = requests.get(f"{POLYMARKET_HOST}/markets", params={"limit":1},
                         headers={"Authorization": f"Bearer {CLOB_API_KEY}"}, timeout=8)
        if r.status_code == 200:
            print("    [OK]   CONNECTED - markets endpoint reachable")
        else:
            print(f"    [FAIL] status {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # 2. Gemini
    print("\n[2] Google Gemini API...")
    try:
        resp = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="hi",
            config=genai_types.GenerateContentConfig(max_output_tokens=5)
        )
        if resp.text:
            print("    [OK]   CONNECTED - Gemini API responding")
        else:
            print("    [FAIL] Empty response from Gemini")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # 3. Twitter (disabled - free tier does not support search)
    print("\n[3] Twitter API...")
    print("    [SKIP] Disabled - free tier does not allow keyword search (403)")

    # 4. Reddit (no auth needed)
    print("\n[4] Reddit API...")
    try:
        r = requests.get("https://www.reddit.com/r/politics/new.json",
            headers={"User-Agent": "polybot/1.0"}, timeout=8)
        if r.status_code == 200:
            print("    [OK]   CONNECTED - Reddit accessible (no key needed)")
        else:
            print(f"    [FAIL] {r.status_code}")
    except Exception as e:
        print(f"    [FAIL] {e}")

    print("\n" + "="*55)
    print(f"  Capital: ${CAPITAL:.2f} | Profit limit: +${DAILY_PROFIT_LIMIT:.2f} | Loss limit: -${DAILY_LOSS_LIMIT:.2f}")
    print(f"  Stop loss: {int(STOP_LOSS_PCT*100)}% | Min confidence: {int(MIN_CONFIDENCE*100)}% | Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
    print("="*55 + "\n")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connections()
    else:
        log.info("PolyBot starting...")
        log.info(f"Capital: ${CAPITAL} | Profit cap: +${DAILY_PROFIT_LIMIT} | Loss cap: -${DAILY_LOSS_LIMIT}")
        log.info(f"Mode: {'PAPER (no real money)' if PAPER_MODE else '*** LIVE — REAL MONEY ***'}")
        test_connections()
        while True:
            try:
                scan_and_trade()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                log.info("Bot stopped by user.")
                log.info(f"Final daily P&L: ${state.pnl:.2f} | Trades: {state.trades} | Wins: {state.wins}")
                break
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                time.sleep(30)
