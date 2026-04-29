"""
News blackout filter.

Pulls the Forex Factory weekly economic calendar (free JSON, no auth)
and blocks signal generation in a window around HIGH-impact events for
the relevant currencies. SMC theory explicitly says skip news; this
implementation enforces that.

Source: https://nfs.faireconomy.media/ff_calendar_thisweek.json
(Forex Factory's public mirror — used by many open-source bots.)

Cache: 1 hour. Failures are non-fatal — `is_blocked()` returns False on
any error, so a network blip can't accidentally veto every signal.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:                  # pragma: no cover
    requests = None  # type: ignore

log = logging.getLogger("news_filter")

FEED_URL    = os.environ.get("NEWS_FEED_URL",
                             "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
CACHE_TTL_S = 3600       # 1 hour
WINDOW_MIN  = 30         # block ±30 minutes around HIGH-impact events

# Pair → currencies that move it. Block when EITHER currency has news.
PAIR_CURRENCIES = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "GBPJPY": ("GBP", "JPY"),
    "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "XAUUSD": ("USD",),                # gold mostly tracks USD events
}

_cache: List[dict] = []
_cache_ts: float = 0.0
_lock = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────
def is_blocked(pair: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Returns (blocked, reason). Reason is empty when not blocked.
    """
    if requests is None:
        return False, ""
    events = _events()
    if not events:
        return False, ""

    now = now or datetime.now(timezone.utc)
    ccys = PAIR_CURRENCIES.get(pair, ())
    if not ccys:
        return False, ""

    for ev in events:
        if ev.get("impact", "").lower() != "high":
            continue
        if ev.get("currency", "").upper() not in ccys:
            continue
        ev_time = _parse_iso(ev.get("date") or ev.get("time"))
        if ev_time is None:
            continue
        delta_min = abs((now - ev_time).total_seconds()) / 60.0
        if delta_min <= WINDOW_MIN:
            return True, f"{ev.get('currency')} {ev.get('title','News')} ({int(delta_min)}m away)"
    return False, ""


def upcoming(hours: float = 24.0) -> List[dict]:
    """Return upcoming HIGH events in the next `hours` — for /api/news endpoint."""
    events = _events()
    now = datetime.now(timezone.utc)
    out: List[dict] = []
    for ev in events:
        if ev.get("impact", "").lower() != "high":
            continue
        t = _parse_iso(ev.get("date") or ev.get("time"))
        if not t:
            continue
        delta_h = (t - now).total_seconds() / 3600.0
        if 0 <= delta_h <= hours:
            out.append({
                "currency": ev.get("currency"),
                "title":    ev.get("title"),
                "time":     ev.get("date") or ev.get("time"),
                "in_hours": round(delta_h, 2),
            })
    out.sort(key=lambda x: x["in_hours"])
    return out


# ── Internals ─────────────────────────────────────────────────────────
def _events() -> List[dict]:
    global _cache, _cache_ts
    with _lock:
        if _cache and time.time() - _cache_ts < CACHE_TTL_S:
            return _cache
        try:
            r = requests.get(FEED_URL, timeout=8)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                _cache    = data
                _cache_ts = time.time()
                log.info(f"News feed loaded: {len(data)} events")
            else:
                log.warning("News feed returned non-list payload")
        except Exception as e:
            log.warning(f"News feed fetch failed: {e}")
        return _cache


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Forex Factory format: "2024-01-15T08:30:00-05:00"
        return datetime.fromisoformat(str(s)).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None
