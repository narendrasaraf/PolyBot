"""
polybot/executor.py
===================
Trade Execution Layer — places, tracks, and cancels orders on Polymarket.

Features:
  - Paper mode (simulate fills without real orders)
  - Live mode (signed CLOB orders for YES and NO tokens)
  - Slippage control (reject if fill worse than tolerance)
  - Exit monitoring (stop-loss and take-profit per position)
  - HMAC-signed CLOB API requests
"""

from __future__ import annotations

import time
import hmac
import hashlib
import base64
import json
import requests
from datetime import datetime
from typing import Optional

from polybot.config import (
    CLOB_HOST, CLOB_API_KEY, CLOB_SECRET, CLOB_PASSPHRASE,
    PAPER_MODE, SLIPPAGE_TOLERANCE, ORDER_TIMEOUT_SEC,
)
from polybot.risk_manager import RiskManager, Position, risk, state
from polybot.data_layer import fetch_clob_price
from polybot.logger import get_logger

log = get_logger("executor")


# ── CLOB Authentication ───────────────────────────────────────────────────────

def _build_clob_headers(method: str, path: str, body: str = "") -> dict:
    """
    Build HMAC-SHA256 signed headers for CLOB API requests.
    Reference: https://docs.polymarket.com/#authentication
    """
    ts      = str(int(time.time()))
    nonce   = str(int(time.time() * 1000))
    message = ts + method.upper() + path + body
    secret_bytes = base64.b64decode(CLOB_SECRET) if CLOB_SECRET else b""
    sig = hmac.new(secret_bytes, message.encode(), hashlib.sha256).hexdigest()
    return {
        "Authorization":  f"Bearer {CLOB_API_KEY}",
        "CLOB-API-KEY":   CLOB_API_KEY,
        "CLOB-SECRET":    CLOB_SECRET,
        "CLOB-PASSPHRASE": CLOB_PASSPHRASE,
        "CLOB-TIMESTAMP": ts,
        "CLOB-NONCE":     nonce,
        "CLOB-SIGNATURE": sig,
        "Content-Type":   "application/json",
    }


# ── Order placement ───────────────────────────────────────────────────────────

class OrderResult:
    """Encapsulates the result of an order placement attempt."""
    def __init__(self, success: bool, order_id: str = "", fill_price: float = 0, msg: str = ""):
        self.success    = success
        self.order_id   = order_id
        self.fill_price = fill_price
        self.msg        = msg

    def __repr__(self):
        return f"OrderResult(success={self.success}, id={self.order_id}, fill={self.fill_price:.4f}, msg={self.msg})"


def _check_slippage(target_price: float, fill_price: float, side: str) -> bool:
    """Return True if the fill is acceptable within slippage tolerance."""
    if side == "BUY":
        # We paid at most target + tolerance
        return fill_price <= target_price * (1 + SLIPPAGE_TOLERANCE)
    else:
        # We received at least target - tolerance
        return fill_price >= target_price * (1 - SLIPPAGE_TOLERANCE)


def place_order(
    token_id:    str,
    side:        str,      # "BUY" or "SELL"
    target_price: float,
    size_tokens:  float,
    size_dollars: float,
) -> OrderResult:
    """
    Place a LIMIT order on the Polymarket CLOB.

    Paper mode: simulates a fill at target_price.
    Live mode: signs and posts to the CLOB REST endpoint.
    """
    if PAPER_MODE:
        log.info(
            f"[PAPER] {side} {size_tokens:.4f} tokens @ {target_price:.4f} "
            f"(${size_dollars:.2f}) | token: {token_id[:16]}..."
        )
        return OrderResult(
            success=True,
            order_id=f"paper_{int(time.time())}",
            fill_price=target_price,
            msg="paper_filled",
        )

    # ── Live order ─────────────────────────────────────────────────────────────
    path  = "/order"
    order = {
        "tokenID":    token_id,
        "price":      str(round(target_price, 4)),
        "size":       str(round(size_tokens, 4)),
        "side":       side.upper(),
        "orderType":  "LIMIT",
        "feeRateBps": "0",
        "nonce":      str(int(time.time() * 1000)),
        "expiration": str(int(time.time()) + ORDER_TIMEOUT_SEC),
    }
    body    = json.dumps(order)
    headers = _build_clob_headers("POST", path, body)

    try:
        r = requests.post(
            f"{CLOB_HOST}{path}",
            headers=headers,
            data=body,
            timeout=ORDER_TIMEOUT_SEC,
        )
        resp = r.json()
        log.debug(f"CLOB response [{r.status_code}]: {resp}")

        if r.status_code in (200, 201):
            order_id   = resp.get("orderID", resp.get("id", ""))
            fill_price = float(resp.get("price", target_price))
            if not _check_slippage(target_price, fill_price, side):
                log.warning(
                    f"Slippage exceeded: target={target_price:.4f} fill={fill_price:.4f} — cancelling"
                )
                _cancel_order(order_id)
                return OrderResult(False, order_id, fill_price, "slippage_exceeded")
            return OrderResult(True, order_id, fill_price, "filled")
        else:
            msg = resp.get("error", resp.get("message", str(resp)))
            log.error(f"Order rejected [{r.status_code}]: {msg}")
            return OrderResult(False, "", 0, msg)

    except Exception as exc:
        log.error(f"Order placement exception: {exc}")
        return OrderResult(False, "", 0, str(exc))


def _cancel_order(order_id: str) -> bool:
    """Cancel an open CLOB order (used after slippage rejection)."""
    if not order_id or PAPER_MODE:
        return True
    try:
        path    = f"/order/{order_id}"
        headers = _build_clob_headers("DELETE", path)
        r       = requests.delete(f"{CLOB_HOST}{path}", headers=headers, timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ── Executor — high-level trade actions ───────────────────────────────────────

class Executor:
    """
    Orchestrates entry, monitoring, and exit of positions.
    Works in tandem with RiskManager.
    """

    def __init__(self, risk_mgr: RiskManager = risk):
        self.risk = risk_mgr

    # ── Entry ─────────────────────────────────────────────────────────────────

    def enter_position(self, snap, signal, size_dollars: float) -> Optional[Position]:
        """
        Executes a trade and registers the position with the risk manager.
        Returns the Position object on success, None on failure.
        """
        side_token = "BUY"    # we always BUY either YES or NO tokens
        token_id   = snap.yes_token_id if signal.direction == "BUY_YES" else snap.no_token_id
        if not token_id:
            token_id = snap.yes_token_id or snap.condition_id   # fallback

        target_price = snap.best_yes_ask if signal.direction == "BUY_YES" else (
            1.0 - snap.best_yes_bid  # NO token bid price ≈ 1 - YES bid
        )

        size_tokens = round(size_dollars / max(target_price, 0.001), 4)

        log.info(
            f"ENTER | {signal.direction} | {snap.question[:50]} | "
            f"${size_dollars:.2f} | {size_tokens:.4f} tkns @ {target_price:.4f} | "
            f"conf={signal.confidence:.2f} edge={signal.edge:.4f}"
        )

        result = place_order(token_id, side_token, target_price, size_tokens, size_dollars)

        if not result.success:
            log.warning(f"Entry failed: {result.msg}")
            return None

        position = self.risk.build_position(snap, signal, size_dollars)
        position.order_id    = result.order_id
        position.entry_price = result.fill_price   # actual fill price

        self.risk.state.open_positions[snap.condition_id] = position
        self.risk.state.save_state()
        log.info(
            f"Position opened | SL={position.stop_loss:.4f} | TP={position.take_profit:.4f} | "
            f"Mode: {'PAPER' if PAPER_MODE else 'LIVE'}"
        )
        return position

    # ── Exit ──────────────────────────────────────────────────────────────────

    def exit_position(self, pos: Position, current_price: float, reason: str) -> float:
        """Sell the position and record the trade."""
        # To exit a BUY_YES position, we SELL the YES token
        # To exit a BUY_NO position, we SELL the NO token
        side_token = "SELL"
        token_id   = pos.token_id
        size_tokens = pos.size_tokens

        log.info(
            f"EXIT | {pos.side} | {pos.question[:50]} | "
            f"{size_tokens:.4f} tkns @ {current_price:.4f} | Reason: {reason}"
        )

        result = place_order(token_id, side_token, current_price, size_tokens, pos.size_dollars)
        actual_exit = result.fill_price if result.success else current_price

        pnl = self.risk.close_position(pos, actual_exit, reason)
        return pnl

    # ── Position monitoring ────────────────────────────────────────────────────

    def monitor_positions(self) -> list[dict]:
        """
        Check all open positions against current market prices.
        Triggers stop-loss or take-profit exits as needed.
        Returns list of exit events.
        """
        exits = []
        for cid, pos in list(self.risk.state.open_positions.items()):
            price_data = fetch_clob_price(pos.token_id)
            if not price_data:
                log.debug(f"Could not fetch price for {pos.question[:40]}")
                continue
            current_price = price_data["mid"]
            unrealised    = pos.pnl_at(current_price)

            log.debug(
                f"Monitor | {pos.question[:40]} | price={current_price:.4f} "
                f"entry={pos.entry_price:.4f} pnl=${unrealised:+.2f}"
            )

            exit_reason = self.risk.get_exit_reason(pos, current_price)
            if exit_reason:
                pnl = self.exit_position(pos, current_price, exit_reason)
                exits.append({"condition_id": cid, "pnl": pnl, "reason": exit_reason})

        return exits

    # ── Market condition reversal exit ─────────────────────────────────────────

    def check_signal_reversal(self, pos: Position, new_signal) -> bool:
        """
        Exit if the market signal has reversed against our position.
        Example: we hold BUY_YES but new signal now says BUY_NO.
        """
        if pos.side == "BUY_YES" and new_signal.direction == "BUY_NO":
            price_data = fetch_clob_price(pos.token_id)
            if price_data:
                pnl = self.exit_position(
                    pos, price_data["mid"], "SIGNAL_REVERSAL"
                )
                log.info(f"Exited on signal reversal | P&L: ${pnl:+.2f}")
                return True
        if pos.side == "BUY_NO" and new_signal.direction == "BUY_YES":
            price_data = fetch_clob_price(pos.token_id)
            if price_data:
                pnl = self.exit_position(
                    pos, price_data["mid"], "SIGNAL_REVERSAL"
                )
                log.info(f"Exited on signal reversal | P&L: ${pnl:+.2f}")
                return True
        return False


# Module-level singleton
executor = Executor()
