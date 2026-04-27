"""
Historical replay environment for offline DQN training.

The env walks bar-by-bar through a list of candles (as produced by
server.fetch_bars()) and at each step:

  1. Builds an analyse() dict on a sliding window for HTF / MTF / LTF.
  2. Asks the agent for an action (HOLD / BUY_HALF / BUY_FULL / SELL_HALF / SELL_FULL).
  3. If the action is a trade, simulates the trade forward through the
     subsequent bars using the same SL/TP rules the live server uses
     (30-pip SL, 1.5R TP1, 3R TP2). Reward = pip pnl × size_factor minus
     a tiny per-trade cost to discourage overtrading.
  4. Skips ahead past the trade close and continues.

Why this design
---------------
- We DON'T re-implement the SMC engine. We import server.analyse so a
  trained model is guaranteed to see the same features the live bot sees.
- We DON'T model spread / slippage explicitly — the per-trade cost
  abstracts that. You can crank it up later for stricter realism.
- "Episode" = one full pass through one pair's history. That's enough
  variation per epoch to keep the gradient noisy in a useful way.

Runtime cost
------------
analyse() is O(window_size); we slide a 200-bar window through ~5000 bars
per pair so a full backtest is a few seconds per pair on CPU. Feature
caching (see trainer.py) makes repeated epochs nearly free.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Make `import server` work whether trainer is run from repo root or rl/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import server  # noqa: E402  pylint: disable=wrong-import-position

from rl.features import (  # noqa: E402
    STATE_DIM, ACTION_DIM, extract_state,
    ACTION_HOLD, ACTION_BUY_HALF, ACTION_BUY_FULL,
    ACTION_SELL_HALF, ACTION_SELL_FULL,
)


# ── Trade simulation rules (mirrors server.make_signal logic) ─────────
SL_PIPS_DEFAULT = 30
TP1_R = 1.5
TP2_R = 3.0
PER_TRADE_COST_PIPS = 1.0   # spread / slippage abstraction


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict


@dataclass
class EnvConfig:
    pair: str
    bars: List[dict]                # main timeframe (H1) bars, oldest → newest
    htf_bars: Optional[List[dict]] = None   # H4 bars (defaults to H1)
    ltf_bars: Optional[List[dict]] = None   # M15 bars (defaults to H1)
    window: int = 200               # how many bars analyse() sees per step
    max_trade_bars: int = 48        # safety cap (~2 trading days on H1)
    sl_pips: int = SL_PIPS_DEFAULT


class ForexEnv:
    """Single-pair, single-episode replay env."""

    def __init__(self, cfg: EnvConfig):
        if not cfg.bars or len(cfg.bars) < cfg.window + 50:
            raise ValueError(f"Need at least {cfg.window + 50} bars, got {len(cfg.bars or [])}")
        self.cfg = cfg
        self.pair = cfg.pair
        self.pip = server.PIP.get(cfg.pair, 0.0001)
        # Index of the *current* bar — analyse() sees [i-window+1 ... i]
        self._i = cfg.window
        self._n = len(cfg.bars)
        # Track bookkeeping for the episode
        self.trades: List[dict] = []

    # ── Gym-style API ─────────────────────────────────────────────────
    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def reset(self) -> np.ndarray:
        self._i = self.cfg.window
        self.trades = []
        return self._build_state()

    def step(self, action: int) -> StepResult:
        if self._i >= self._n - 1:
            return StepResult(self._build_state(), 0.0, True, {"reason": "end_of_data"})

        if action == ACTION_HOLD:
            self._i += 1
            done = self._i >= self._n - 1
            return StepResult(self._build_state(), 0.0, done, {"action": "HOLD"})

        # Trade actions
        direction, size = self._decode_action(action)
        reward, info = self._simulate_trade(direction, size)
        self.trades.append(info)
        # Advance past the bars consumed by the trade
        self._i += info.get("bars_used", 1)
        done = self._i >= self._n - 1
        return StepResult(self._build_state(), reward, done, info)

    # ── Internals ─────────────────────────────────────────────────────
    def _build_state(self) -> np.ndarray:
        i = min(self._i, self._n - 1)
        window = self.cfg.bars[max(0, i - self.cfg.window + 1): i + 1]
        # Use same window for HTF/LTF if dedicated arrays not provided —
        # the model still works (less granular HTF context, but consistent).
        htf_window = self.cfg.htf_bars[: i + 1][-self.cfg.window:] if self.cfg.htf_bars else window
        ltf_window = self.cfg.ltf_bars[: i + 1][-self.cfg.window:] if self.cfg.ltf_bars else window
        htf = server.analyse(htf_window, self.pair)
        mtf = server.analyse(window,     self.pair)
        ltf = server.analyse(ltf_window, self.pair)
        return extract_state(htf, mtf, ltf, self.pair)

    @staticmethod
    def _decode_action(action: int) -> Tuple[str, float]:
        if action == ACTION_BUY_HALF:  return "BUY",  0.5
        if action == ACTION_BUY_FULL:  return "BUY",  1.0
        if action == ACTION_SELL_HALF: return "SELL", 0.5
        if action == ACTION_SELL_FULL: return "SELL", 1.0
        raise ValueError(f"Bad action: {action}")

    def _simulate_trade(self, direction: str, size: float) -> Tuple[float, dict]:
        """
        March forward through subsequent bars and resolve the trade with
        the same rules the live server uses. Returns (reward_pips, info).

        Reward shaping:
          win  : +pnl_pips × size − cost
          loss : −sl_pips  × size − cost
          partial (TP1 then SL@BE): +tp1_pips × size − cost
        """
        cfg = self.cfg
        i0 = self._i
        entry_bar = cfg.bars[i0]
        entry = entry_bar["close"]
        pip = self.pip
        sl_d  = cfg.sl_pips * pip
        tp1_d = sl_d * TP1_R
        tp2_d = sl_d * TP2_R

        if direction == "BUY":
            sl  = entry - sl_d
            tp1 = entry + tp1_d
            tp2 = entry + tp2_d
        else:
            sl  = entry + sl_d
            tp1 = entry - tp1_d
            tp2 = entry - tp2_d

        outcome = "open"
        bars_used = 0
        tp1_hit = False
        be_sl = sl  # SL after TP1 moves to break-even
        for j in range(i0 + 1, min(self._n, i0 + 1 + cfg.max_trade_bars)):
            b = cfg.bars[j]
            hi, lo = b["high"], b["low"]
            bars_used = j - i0
            if direction == "BUY":
                # TP2 first if both hit in same bar — pessimistic? Use intrabar order:
                # check SL before TP to be conservative.
                if lo <= be_sl:
                    outcome = "tp1_be" if tp1_hit else "sl"
                    break
                if not tp1_hit and hi >= tp1:
                    tp1_hit = True
                    be_sl = entry  # move SL to break-even
                if hi >= tp2:
                    outcome = "tp2"
                    break
            else:  # SELL
                if hi >= be_sl:
                    outcome = "tp1_be" if tp1_hit else "sl"
                    break
                if not tp1_hit and lo <= tp1:
                    tp1_hit = True
                    be_sl = entry
                if lo <= tp2:
                    outcome = "tp2"
                    break

        if outcome == "open":
            outcome = "tp1_be" if tp1_hit else "timeout"
            bars_used = bars_used or 1

        # Pip pnl by outcome
        if outcome == "tp2":
            pip_pnl = cfg.sl_pips * TP2_R
        elif outcome == "tp1_be":
            pip_pnl = cfg.sl_pips * TP1_R   # TP1 hit, then SL@BE → keep TP1 profit
        elif outcome == "sl":
            pip_pnl = -cfg.sl_pips
        else:  # timeout — close at last bar's close
            last_close = cfg.bars[i0 + bars_used]["close"]
            raw = (last_close - entry) / pip if direction == "BUY" else (entry - last_close) / pip
            pip_pnl = float(raw)

        reward = pip_pnl * size - PER_TRADE_COST_PIPS
        info = {
            "direction": direction,
            "size": size,
            "outcome": outcome,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "pip_pnl": round(pip_pnl, 2),
            "reward": round(reward, 2),
            "bars_used": max(bars_used, 1),
        }
        return reward, info

    # ── Stats ─────────────────────────────────────────────────────────
    def episode_stats(self) -> dict:
        if not self.trades:
            return {"trades": 0, "wins": 0, "losses": 0, "net_pips": 0.0}
        wins = sum(1 for t in self.trades if t["pip_pnl"] > 0)
        losses = sum(1 for t in self.trades if t["pip_pnl"] <= 0)
        net = round(sum(t["pip_pnl"] for t in self.trades), 1)
        return {
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(len(self.trades), 1) * 100, 1),
            "net_pips": net,
            "avg_pips": round(net / len(self.trades), 2),
        }
