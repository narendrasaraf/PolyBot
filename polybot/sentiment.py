"""
polybot/sentiment.py
====================
Optional Sentiment Analysis Module (no Twitter/X dependency).

News sources:
  1. NewsAPI.org    — requires NEWSAPI_KEY in .env
  2. GNews.io       — requires GNEWS_KEY in .env
  3. Reddit r/news  — no key required (public JSON)
  4. Gemini Search  — grounded web search (fallback)

Usage:
    from polybot.sentiment import get_sentiment_score
    score = get_sentiment_score("Will the Fed cut rates in 2025?")
    # Returns float in [0.0, 1.0] — 0.5 = neutral
"""

from __future__ import annotations

import time
import json
import requests
from typing import Optional

from polybot.config import NEWSAPI_KEY, GNEWS_KEY, GEMINI_API_KEY, GEMINI_MODEL
from polybot.logger import get_logger

log = get_logger("sentiment")

_HEADERS = {"User-Agent": "PolyBot/2.0 (news sentiment analysis)"}

# ── Positive / Negative keyword sets ─────────────────────────────────────────
_POS = {
    "approved","confirmed","passed","rally","surge","bullish","win","victory",
    "positive","growth","recovery","increase","up","gain","boom","deal","agreement",
    "support","backing","green","success","record","breakout","upgrade",
}
_NEG = {
    "rejected","denied","failed","crash","dump","bearish","loss","defeat",
    "negative","decline","recession","decrease","down","collapse","ban","sanction",
    "warning","risk","danger","plunge","downgrade","cut","delay","cancelled",
}


def _score_text(text: str) -> Optional[float]:
    """Score a text block: returns 0.0–1.0 or None if no keywords found."""
    words = set(text.lower().split())
    pos   = len(words & _POS)
    neg   = len(words & _NEG)
    if pos + neg == 0:
        return None
    return pos / (pos + neg)


# ── Source 1: NewsAPI.org ────────────────────────────────────────────────────

def _newsapi_sentiment(query: str) -> Optional[float]:
    if not NEWSAPI_KEY:
        return None
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            headers=_HEADERS,
            params={
                "q":       query[:100],
                "sortBy":  "publishedAt",
                "pageSize": 20,
                "language": "en",
                "apiKey":  NEWSAPI_KEY,
            },
            timeout=10,
        )
        articles = r.json().get("articles", [])
        if not articles:
            return None
        scores = []
        for art in articles:
            text = (art.get("title", "") + " " + art.get("description", "")).strip()
            s = _score_text(text)
            if s is not None:
                scores.append(s)
        return sum(scores) / len(scores) if scores else None
    except Exception as exc:
        log.debug(f"NewsAPI error: {exc}")
        return None


# ── Source 2: GNews.io ────────────────────────────────────────────────────────

def _gnews_sentiment(query: str) -> Optional[float]:
    if not GNEWS_KEY:
        return None
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            headers=_HEADERS,
            params={
                "q":      query[:100],
                "lang":   "en",
                "max":    10,
                "token":  GNEWS_KEY,
            },
            timeout=10,
        )
        articles = r.json().get("articles", [])
        if not articles:
            return None
        scores = [
            s for art in articles
            for s in [_score_text(art.get("title", "") + " " + art.get("description", ""))]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else None
    except Exception as exc:
        log.debug(f"GNews error: {exc}")
        return None


# ── Source 3: Reddit r/news / r/worldnews ──────────────────────────────────

def _reddit_sentiment(query: str) -> Optional[float]:
    """Lightweight Reddit news sentiment without auth."""
    try:
        r = requests.get(
            "https://www.reddit.com/search.json",
            headers={"User-Agent": "polybot/2.0"},
            params={"q": query[:80], "sort": "new", "limit": 25, "t": "week"},
            timeout=10,
        )
        posts = r.json().get("data", {}).get("children", [])
        scores = []
        for post in posts:
            d = post.get("data", {})
            text  = d.get("title", "") + " " + d.get("selftext", "")
            upvote_w = min(2.0, 1 + d.get("score", 0) / 300)
            s = _score_text(text)
            if s is not None:
                scores.append(s * upvote_w)
        return sum(scores) / len(scores) if scores else None
    except Exception as exc:
        log.debug(f"Reddit sentiment error: {exc}")
        return None


# ── Source 4: Gemini Search grounding ────────────────────────────────────────

def _gemini_sentiment(question: str) -> Optional[float]:
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types as gt
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "Search the latest news about this prediction market and return ONLY valid JSON: "
            '{"bullish": <0-10>, "bearish": <0-10>, "summary": "<10 words>"}. '
            f"Market: {question}"
        )
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=gt.GenerateContentConfig(
                tools=[gt.Tool(google_search=gt.GoogleSearch())],
                max_output_tokens=150,
            ),
        )
        text = resp.text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(text)
        bull = int(data.get("bullish", 5))
        bear = int(data.get("bearish", 5))
        return bull / max(bull + bear, 1)
    except Exception as exc:
        log.debug(f"Gemini sentiment error: {exc}")
        return None


# ── Aggregator ────────────────────────────────────────────────────────────────

def get_sentiment_score(question: str) -> dict:
    """
    Query all available news sources, combine scores, and return:
      {
        "score": float,        # 0.0–1.0 aggregate (0.5 = neutral)
        "sources_used": int,
        "breakdown": {...},
        "direction": "positive" | "negative" | "neutral",
      }
    """
    keywords = " ".join(question.replace("?", "").split()[:8])   # first 8 words

    sources: dict[str, Optional[float]] = {
        "newsapi": _newsapi_sentiment(keywords),
        "gnews":   _gnews_sentiment(keywords),
        "reddit":  _reddit_sentiment(keywords),
        "gemini":  _gemini_sentiment(question),
    }

    valid = {k: v for k, v in sources.items() if v is not None}
    if not valid:
        return {"score": 0.5, "sources_used": 0, "breakdown": sources, "direction": "neutral"}

    # Weight Gemini higher (uses grounded search)
    weights = {"newsapi": 1.0, "gnews": 1.0, "reddit": 0.8, "gemini": 1.5}
    total_w = sum(weights[k] for k in valid)
    avg     = sum(weights[k] * v for k, v in valid.items()) / total_w
    avg     = round(max(0.01, min(0.99, avg)), 4)

    direction = (
        "positive" if avg > 0.55 else
        "negative" if avg < 0.45 else
        "neutral"
    )

    log.debug(f"Sentiment for '{question[:40]}': {avg:.3f} ({direction}) | sources: {list(valid.keys())}")

    return {
        "score":       avg,
        "sources_used": len(valid),
        "breakdown":   {k: round(v, 3) if v is not None else None for k, v in sources.items()},
        "direction":   direction,
    }
