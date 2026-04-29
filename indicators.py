"""
ATR (Average True Range) + spread constants for SL sizing and P&L math.

ATR-based SL replaces the bot's old fixed 30-pip SL. Per-pair noise
floors are very different — XAUUSD moves 30 pips in a single H1 candle,
so a 30-pip SL just means random stop-outs on normal volatility. EURUSD
30 pips is too generous in calm regimes. ATR fixes both.

Spread constants reflect typical OANDA / IC Markets / IG demo spreads
during normal session hours. They're per-pair so the bot can subtract
the broker's drag from every reported P&L. If you switch to OANDA's
live feed (oanda_data.fetch_prices), the live spread is available per
tick — `server.fetch_prices` populates `spread_pips` and the live value
should be preferred over these defaults when present.
"""
from __future__ import annotations

from typing import List, Optional


# ── Per-pair tuning ───────────────────────────────────────────────────
# Conservative typical spreads (pips). Adjust to your broker.
SPREAD_PIPS = {
    "EURUSD": 0.6,
    "GBPUSD": 1.0,
    "USDJPY": 0.8,
    "GBPJPY": 1.8,
    "AUDUSD": 1.0,
    "USDCAD": 1.4,
    "XAUUSD": 2.5,   # gold spreads are wider
}

# Per-pair SL bounds (pips) — clamps the ATR-based SL into a sane range.
# Min: tight enough to make 1.5R/3R targets reachable. Max: wide enough
# to survive normal regime noise but not absurd (no 200-pip stops).
SL_BOUNDS = {
    "EURUSD": (12, 60),
    "GBPUSD": (15, 70),
    "USDJPY": (12, 60),
    "GBPJPY": (25, 100),
    "AUDUSD": (12, 60),
    "USDCAD": (15, 70),
    "XAUUSD": (40, 250),
}

# ATR multiplier — 1.2× ATR is the SMC community sweet spot. Lower =
# tighter stop-outs but better R:R; higher = fewer false stops but worse
# R:R per win.
ATR_MULT = 1.2
ATR_PERIOD = 14


# ── ATR computation ───────────────────────────────────────────────────
def calc_atr(bars: List[dict], period: int = ATR_PERIOD) -> Optional[float]:
    """
    Wilder-style ATR over `period` bars. Returns absolute price units
    (not pips) — caller divides by `pip` size for the pair.

    Returns None on too-few bars.
    """
    if not bars or len(bars) < period + 2:
        return None

    trs: list[float] = []
    for i in range(1, len(bars)):
        h, l = bars[i]["high"], bars[i]["low"]
        prev_c = bars[i - 1]["close"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)

    if len(trs) < period:
        return None

    # Wilder's smoothing: first ATR = simple mean of first `period` TRs,
    # then ATR_n = (ATR_{n-1} * (period-1) + TR_n) / period
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return float(atr)


def atr_sl_pips(bars: List[dict], pair: str, pip: float) -> int:
    """
    Returns ATR-based SL in pips, clamped to per-pair bounds.
    Falls back to 30 if ATR can't be computed (too few bars).
    """
    atr = calc_atr(bars, ATR_PERIOD)
    if atr is None or pip <= 0:
        return 30
    raw_pips = (atr * ATR_MULT) / pip
    lo, hi = SL_BOUNDS.get(pair, (12, 100))
    return max(lo, min(hi, int(round(raw_pips))))


def spread_pips(pair: str, live_spread_pips: Optional[float] = None) -> float:
    """Live spread (from OANDA tick) preferred; else per-pair default."""
    if live_spread_pips is not None and live_spread_pips > 0:
        return float(live_spread_pips)
    return SPREAD_PIPS.get(pair, 1.0)
