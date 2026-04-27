"""
Feature extraction — analyse() dicts → fixed-length numpy state vector.

This module defines the *exact* state representation the RL agent sees.
Both training (rl.trainer / rl.forex_env) and live inference
(rl.policy via server.py) MUST use this function so that a saved model
keeps working across both regimes.

Vector layout (35 dims, all in ~[-1, 1] or {0,1}):

    [0:2]    HTF trend       (bull, bear)
    [2:4]    MTF trend       (bull, bear)
    [4:6]    LTF trend       (bull, bear)
    [6:8]    MTF CHoCH       (bull, bear)
    [8:10]   MTF BOS         (bull, bear)
    [10:12]  MTF zone        (discount, premium)
    [12:14]  MTF OB tapped   (bull, bear)
    [14:16]  MTF FVG near    (bull, bear)
    [16:18]  MTF Breaker     (bull, bear)
    [18:20]  Judas swing     (bull_reversal, bear_reversal)
    [20:24]  Asian range pos (near_hi, near_lo, above, below)
    [24:28]  Killzone        (London, NY, Asian, none)
    [28:35]  Pair one-hot    (EURUSD, GBPUSD, USDJPY, GBPJPY, AUDUSD, USDCAD, XAUUSD)

Why this layout
---------------
- All boolean — keeps the network small and lets DQN converge fast.
- Bull/bear pairs (rather than signed scalars) let the network treat
  long and short setups asymmetrically, which they often are in practice.
- Pair one-hot is included so a single network can serve all 7 pairs;
  the model can learn pair-specific biases.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np

# Public — keep these in sync with the layout above
STATE_DIM = 35
ACTION_DIM = 5  # HOLD, BUY_HALF, BUY_FULL, SELL_HALF, SELL_FULL

ACTIONS = ["HOLD", "BUY_HALF", "BUY_FULL", "SELL_HALF", "SELL_FULL"]
ACTION_HOLD       = 0
ACTION_BUY_HALF   = 1
ACTION_BUY_FULL   = 2
ACTION_SELL_HALF  = 3
ACTION_SELL_FULL  = 4

PAIR_ORDER = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD", "USDCAD", "XAUUSD"]


def _trend_bits(trend: Optional[str]) -> tuple[float, float]:
    if trend == "bullish":
        return 1.0, 0.0
    if trend == "bearish":
        return 0.0, 1.0
    return 0.0, 0.0


def _killzone_bits(kz: Optional[str]) -> tuple[float, float, float, float]:
    return (
        1.0 if kz == "London"   else 0.0,
        1.0 if kz == "New York" else 0.0,
        1.0 if kz == "Asian"    else 0.0,
        1.0 if kz is None        else 0.0,
    )


def _pair_onehot(pair: str) -> list[float]:
    out = [0.0] * len(PAIR_ORDER)
    if pair in PAIR_ORDER:
        out[PAIR_ORDER.index(pair)] = 1.0
    return out


def extract_state(htf: Dict, mtf: Dict, ltf: Dict, pair: str) -> np.ndarray:
    """
    Build the 35-dim state vector from the dicts produced by server.analyse().

    Any missing field is treated as falsy (zero) — this keeps the function
    robust to partial data (e.g. when LTF bars failed to fetch).
    """
    htf = htf or {}
    mtf = mtf or {}
    ltf = ltf or {}

    htf_b, htf_s = _trend_bits(htf.get("trend"))
    mtf_b, mtf_s = _trend_bits(mtf.get("trend"))
    ltf_b, ltf_s = _trend_bits(ltf.get("trend"))

    choch_b = 1.0 if mtf.get("choch") and mtf.get("cd") == "bullish" else 0.0
    choch_s = 1.0 if mtf.get("choch") and mtf.get("cd") == "bearish" else 0.0
    bos_b   = 1.0 if mtf.get("bos")   and mtf.get("bd") == "bullish" else 0.0
    bos_s   = 1.0 if mtf.get("bos")   and mtf.get("bd") == "bearish" else 0.0

    disc = 1.0 if mtf.get("disc") else 0.0
    prem = 1.0 if mtf.get("prem") else 0.0

    tapped = mtf.get("tapped", []) or []
    ob_tap_b = 1.0 if any(o.get("bull")        for o in tapped) else 0.0
    ob_tap_s = 1.0 if any(o.get("bull") is False for o in tapped) else 0.0

    near = mtf.get("near", []) or []
    fvg_b = 1.0 if any(f.get("bull")        for f in near) else 0.0
    fvg_s = 1.0 if any(f.get("bull") is False for f in near) else 0.0

    brk = mtf.get("breakers", []) or []
    brk_b = 1.0 if any(b.get("bull")        for b in brk) else 0.0
    brk_s = 1.0 if any(b.get("bull") is False for b in brk) else 0.0

    judas = mtf.get("judas")
    jud_b = 1.0 if judas == "bull_reversal" else 0.0
    jud_s = 1.0 if judas == "bear_reversal" else 0.0

    asian_hi = 1.0 if mtf.get("near_asian_hi") else 0.0
    asian_lo = 1.0 if mtf.get("near_asian_lo") else 0.0
    above    = 1.0 if mtf.get("above_asian")   else 0.0
    below    = 1.0 if mtf.get("below_asian")   else 0.0

    kz_l, kz_n, kz_a, kz_o = _killzone_bits(mtf.get("killzone"))

    pair_oh = _pair_onehot(pair)

    vec = [
        htf_b, htf_s,
        mtf_b, mtf_s,
        ltf_b, ltf_s,
        choch_b, choch_s,
        bos_b,   bos_s,
        disc,    prem,
        ob_tap_b, ob_tap_s,
        fvg_b,    fvg_s,
        brk_b,    brk_s,
        jud_b,    jud_s,
        asian_hi, asian_lo, above, below,
        kz_l,     kz_n,     kz_a,  kz_o,
        *pair_oh,
    ]

    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape[0] == STATE_DIM, f"state dim mismatch: got {arr.shape[0]}, want {STATE_DIM}"
    return arr


def state_summary(vec: np.ndarray) -> Dict[str, float]:
    """Pretty-print helper — turn a state vector back into named fields. Debug only."""
    names = [
        "htf_bull", "htf_bear",
        "mtf_bull", "mtf_bear",
        "ltf_bull", "ltf_bear",
        "choch_bull", "choch_bear",
        "bos_bull", "bos_bear",
        "disc", "prem",
        "ob_tap_bull", "ob_tap_bear",
        "fvg_bull", "fvg_bear",
        "brk_bull", "brk_bear",
        "judas_bull", "judas_bear",
        "asian_hi", "asian_lo", "above_asian", "below_asian",
        "kz_london", "kz_ny", "kz_asian", "kz_none",
        *[f"pair_{p}" for p in PAIR_ORDER],
    ]
    return {n: float(v) for n, v in zip(names, vec)}
