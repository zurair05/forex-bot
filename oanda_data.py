"""
OANDA v20 REST data source.

Drop-in replacement for yfinance in server.py. Returns bars and prices
in the exact shape the rest of the bot already consumes, with one bonus
field: bid/ask spread on every live tick.

Configuration (environment variables)
-------------------------------------
    OANDA_API_KEY       Personal access token  (required)
    OANDA_ACCOUNT_ID    Account id like 101-001-12345678-001 (required for
                        live prices; not needed for historical candles)
    OANDA_ENV           "practice" (default, free demo) or "live"

If OANDA_API_KEY is not set, `is_configured()` returns False and the
server falls back to yfinance silently.

Why OANDA
---------
- Free demo account: no KYC, no funding required, full read access
- Returns true bid/ask on every tick → real spread modeling
- Stable, documented, no ToS gray area (vs scraping TradingView)
- Same data your real OANDA account would see → backtests match live

Rate limits
-----------
20 requests/second per account, 100 connections, plenty for our use.
We cache aggressively in server.py (BAR_TTL=300s, PRICE_TTL=60s) so
real load is well under that.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    import requests
except ImportError:                # pragma: no cover
    requests = None  # type: ignore

log = logging.getLogger("oanda_data")


# ── Config (env-driven, evaluated lazily so tests can monkeypatch) ────
def _api_key() -> str:    return os.environ.get("OANDA_API_KEY", "")
def _account_id() -> str: return os.environ.get("OANDA_ACCOUNT_ID", "")
def _env() -> str:        return os.environ.get("OANDA_ENV", "practice").lower()

def _base_url() -> str:
    return ("https://api-fxpractice.oanda.com/v3"
            if _env() == "practice"
            else "https://api-fxtrade.oanda.com/v3")


# ── Static maps ───────────────────────────────────────────────────────
# Bot pair name (EURUSD) → OANDA instrument (EUR_USD).
PAIR_MAP = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "GBPJPY": "GBP_JPY",
    "AUDUSD": "AUD_USD",
    "USDCAD": "USD_CAD",
    "XAUUSD": "XAU_USD",
}

GRANULARITY = {
    "M15": "M15",
    "H1":  "H1",
    "H4":  "H4",
    "D1":  "D",
}

# Bars per request, by timeframe — chosen to match the previous
# yfinance window so analyse() / detect_swings() see comparable depth.
CANDLE_COUNT = {
    "M15": 500,
    "H1":  500,
    "H4":  500,
    "D1":  300,
}

TIMEOUT_S = 12


# ── Public API ────────────────────────────────────────────────────────
def is_configured() -> bool:
    """Cheap probe — server.py uses this to decide between OANDA and yfinance."""
    return bool(_api_key()) and requests is not None


def fetch_bars(pair: str, tf: str = "H1") -> Optional[List[dict]]:
    """
    Returns a list of bar dicts in server.py's native shape:
        [{time:int_unix, open, high, low, close, volume}, ...]
    or None on any failure (caller will fall back to yfinance).
    """
    if not is_configured():
        return None
    instr = PAIR_MAP.get(pair)
    if not instr:
        return None
    gran  = GRANULARITY.get(tf, "H1")
    count = CANDLE_COUNT.get(tf, 500)

    try:
        r = requests.get(
            f"{_base_url()}/instruments/{instr}/candles",
            headers=_headers(),
            params={
                "granularity": gran,
                "count":       count,
                "price":       "M",        # mid only — same as yfinance
                "smooth":      "false",
            },
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"OANDA fetch_bars({pair},{tf}) failed: {e}")
        return None

    out: List[dict] = []
    for c in data.get("candles", []):
        # Drop the in-progress candle so analyse() never sees a half-formed bar.
        if not c.get("complete", True):
            continue
        m = c.get("mid") or {}
        if not m:
            continue
        try:
            out.append({
                "time":   _parse_time(c.get("time")),
                "open":   float(m["o"]),
                "high":   float(m["h"]),
                "low":    float(m["l"]),
                "close":  float(m["c"]),
                "volume": int(c.get("volume", 0)),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


def fetch_prices(pairs: List[str]) -> Optional[Dict[str, dict]]:
    """
    Returns {pair: {price, day_high, day_low, prev_close, change_pct,
                     change_abs, bid, ask, spread_abs}}

    `price` is the mid (matches yfinance shape so the dashboard keeps
    working). `bid`, `ask`, and `spread_abs` are new — server.py uses
    spread_abs to surface "real cost" on every signal.
    """
    if not is_configured() or not _account_id():
        return None
    instrs = [PAIR_MAP[p] for p in pairs if p in PAIR_MAP]
    if not instrs:
        return None

    # Live pricing
    try:
        r = requests.get(
            f"{_base_url()}/accounts/{_account_id()}/pricing",
            headers=_headers(),
            params={"instruments": ",".join(instrs)},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"OANDA fetch_prices failed: {e}")
        return None

    inv = {v: k for k, v in PAIR_MAP.items()}

    # Daily H/L/prev-close for the ticker UI — one cheap call per pair.
    daily: Dict[str, list] = {}
    for instr in instrs:
        try:
            r2 = requests.get(
                f"{_base_url()}/instruments/{instr}/candles",
                headers=_headers(),
                params={"granularity": "D", "count": 2, "price": "M"},
                timeout=TIMEOUT_S,
            )
            r2.raise_for_status()
            daily[instr] = r2.json().get("candles", [])
        except Exception:
            daily[instr] = []

    out: Dict[str, dict] = {}
    for p in data.get("prices", []):
        instr = p.get("instrument")
        pair  = inv.get(instr)
        if not pair:
            continue
        try:
            bid = float(p.get("closeoutBid") or p["bids"][0]["price"])
            ask = float(p.get("closeoutAsk") or p["asks"][0]["price"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        mid = (bid + ask) / 2.0

        cs = daily.get(instr) or []
        try:
            today = (cs[-1] or {}).get("mid", {}) if cs else {}
            prev  = (cs[-2] or {}).get("mid", {}) if len(cs) >= 2 else today
            day_high   = float(today.get("h", mid))
            day_low    = float(today.get("l", mid))
            prev_close = float(prev.get("c", mid))
        except Exception:
            day_high, day_low, prev_close = mid, mid, mid

        out[pair] = {
            "price":      round(mid, 5),
            "day_high":   round(day_high, 5),
            "day_low":    round(day_low, 5),
            "prev_close": round(prev_close, 5),
            "change_pct": round((mid - prev_close) / prev_close * 100, 3) if prev_close else 0,
            "change_abs": round(mid - prev_close, 5),
            "bid":        bid,
            "ask":        ask,
            "spread_abs": round(ask - bid, 5),
        }
    return out or None


# ── Internals ─────────────────────────────────────────────────────────
def _headers() -> Dict[str, str]:
    return {
        "Authorization":          f"Bearer {_api_key()}",
        "Accept-Datetime-Format": "UNIX",
        "Content-Type":           "application/json",
    }


def _parse_time(t) -> int:
    """OANDA with Accept-Datetime-Format: UNIX returns '1705299600.000000000'."""
    if t is None:
        return int(time.time())
    if isinstance(t, (int, float)):
        return int(float(t))
    try:
        return int(float(t))
    except (TypeError, ValueError):
        try:
            return int(datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp())
        except Exception:
            return int(time.time())
