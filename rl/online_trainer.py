"""
Online fine-tuning — feed live trade outcomes into the agent.

The live server keeps a memory of every closed trade in
server._signal_history. Each entry has: pair, direction, score_pct,
opened_at, hit_at, pnl_pips, result, plus the SL/TP1/TP2 levels.

After each trade closes we want to:
  1. Reconstruct the *state* the agent saw at signal generation (or
     a near-equivalent), using the live analyse() output that was
     serialised onto the signal record.
  2. Map the chosen direction → the agent's action index. Live trades
     always trade "FULL" size, since the rule-based scorer doesn't yet
     pick HALF; that's fine for fine-tuning.
  3. Push the (state, action, reward, next_state, done=True) transition
     into the buffer and run a few gradient steps.
  4. Save weights and tell rl.policy to hot-reload.

Why this is safe to run in the request thread
---------------------------------------------
Each fine-tune call does only `tune_steps` (default 8) gradient updates
on a 64-batch — well under 100ms on CPU. If anything raises, we swallow
it: a missed online step is harmless; corrupting the live server is not.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from rl.features import (
    extract_state,
    ACTION_HOLD, ACTION_BUY_FULL, ACTION_SELL_FULL,
    STATE_DIM,
)

log = logging.getLogger("rl.online_trainer")
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_LOCK = threading.Lock()


def _direction_to_action(direction: str) -> int:
    if direction == "BUY":  return ACTION_BUY_FULL
    if direction == "SELL": return ACTION_SELL_FULL
    return ACTION_HOLD


def fine_tune_from_trade(
    closed_trade: dict,
    htf_snapshot: Optional[dict] = None,
    mtf_snapshot: Optional[dict] = None,
    ltf_snapshot: Optional[dict] = None,
    tune_steps: int = 8,
) -> bool:
    """
    Run a tiny fine-tune update from one closed trade.

    Parameters
    ----------
    closed_trade
        A record from server._signal_history (pair, direction, pnl_pips, ...).
    htf_snapshot, mtf_snapshot, ltf_snapshot
        The analyse() outputs captured at signal time, if available.
        If absent, we reconstruct a *minimal* state from the trade record;
        less informative but still better than nothing.
    tune_steps
        Number of gradient steps. 4–16 is the right range — more risks
        catastrophic forgetting on a single sample.

    Returns True if a fine-tune actually ran.
    """
    pair = closed_trade.get("pair")
    direction = closed_trade.get("direction")
    pip_pnl = closed_trade.get("pnl_pips", 0)
    if not pair or direction not in ("BUY", "SELL"):
        return False

    with _LOCK:
        try:
            # Lazy import — keep this module cheap for non-RL deployments
            from rl.policy import _get_loaded_agent, reload_weights
            agent = _get_loaded_agent()
            if agent is None:
                return False

            # Build state. Prefer live snapshots; fall back to a minimal vec.
            if htf_snapshot or mtf_snapshot or ltf_snapshot:
                s = extract_state(htf_snapshot or {}, mtf_snapshot or {}, ltf_snapshot or {}, pair)
            else:
                s = _minimal_state(pair, direction)

            a = _direction_to_action(direction)
            r = float(pip_pnl) - 1.0   # apply same per-trade cost the env used
            # No meaningful s' for a one-shot trade — terminal transition
            s2 = np.zeros_like(s)

            from rl.dqn_agent import Transition
            agent.observe(s, a, r, s2, True)

            losses = []
            for _ in range(tune_steps):
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            agent.save(_WEIGHTS_DIR)
            reload_weights()

            log.info(
                f"online fine-tune: {pair} {direction} pnl={pip_pnl:+.1f}p "
                f"steps={len(losses)} mean_loss={np.mean(losses) if losses else 0:.4f}"
            )
            return True
        except Exception as e:
            log.warning(f"online fine-tune skipped: {e}")
            return False


def _minimal_state(pair: str, direction: str) -> np.ndarray:
    """
    Last-resort state when live snapshots aren't available.

    We can at least encode the pair and a vague trend hint from the chosen
    direction. This is biased — the model is seeing "the trader thought
    bullish" rather than the actual SMC features — so it should NEVER be
    the only training signal. It exists so a single missing snapshot
    doesn't drop the sample entirely.
    """
    # Build a default analyse-shaped dict
    fake = {"trend": "bullish" if direction == "BUY" else "bearish"}
    return extract_state(fake, fake, fake, pair)
