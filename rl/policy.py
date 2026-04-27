"""
Runtime inference wrapper used by server.py.

Design priorities (in order)
----------------------------
1. NEVER break the live server. If torch is missing, weights are absent,
   or anything raises during inference, the policy quietly returns None
   and the rule-based scorer remains in charge.
2. Lazy-load: torch is only imported the first time the policy is asked.
   That way a deployment without ML deps starts up the same as before.
3. Hot-reload: get_policy(refresh=True) reloads model.pt from disk —
   used by online_trainer after it fine-tunes the weights.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from rl.features import (
    extract_state, ACTIONS,
    ACTION_HOLD, ACTION_BUY_HALF, ACTION_BUY_FULL,
    ACTION_SELL_HALF, ACTION_SELL_FULL,
)

log = logging.getLogger("rl.policy")

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_LOCK = threading.Lock()
_AGENT = None        # cached DQNAgent (or False if unavailable)


# ── Public ───────────────────────────────────────────────────────────
class RLDecision:
    """
    Result of querying the RL policy.

    Fields
    ------
    direction : "BUY" | "SELL" | "HOLD"
    size      : 0.5 (half) or 1.0 (full); 0.0 for HOLD
    q_values  : 5-vector of raw Q-values
    confidence: softmax probability assigned to the chosen action [0,1]
    """
    __slots__ = ("direction", "size", "q_values", "confidence", "action_idx")

    def __init__(self, direction: str, size: float,
                 q_values: np.ndarray, confidence: float, action_idx: int):
        self.direction = direction
        self.size = size
        self.q_values = q_values
        self.confidence = confidence
        self.action_idx = action_idx

    def as_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "size":       self.size,
            "confidence": round(self.confidence, 3),
            "action":     ACTIONS[self.action_idx],
            "q_values":   {ACTIONS[i]: float(round(q, 3)) for i, q in enumerate(self.q_values)},
        }


def is_available() -> bool:
    """Cheap probe used by server.py to decide whether to query."""
    return _load_agent() is not None


def predict(htf: dict, mtf: dict, ltf: dict, pair: str) -> Optional[RLDecision]:
    """
    Run the policy on one analyse() snapshot.

    Returns None if the policy isn't loaded — caller falls back to rules.
    """
    agent = _load_agent()
    if agent is None:
        return None
    try:
        state = extract_state(htf, mtf, ltf, pair)
        q = agent.q_values(state)
        a = int(np.argmax(q))
        # Softmax over Q for a confidence read; T=1.0 is fine for small spans
        z = q - q.max()
        p = np.exp(z); p = p / p.sum()
        confidence = float(p[a])
        if a == ACTION_HOLD:
            return RLDecision("HOLD", 0.0, q, confidence, a)
        if a in (ACTION_BUY_HALF, ACTION_BUY_FULL):
            size = 0.5 if a == ACTION_BUY_HALF else 1.0
            return RLDecision("BUY", size, q, confidence, a)
        size = 0.5 if a == ACTION_SELL_HALF else 1.0
        return RLDecision("SELL", size, q, confidence, a)
    except Exception as e:
        log.warning(f"RL predict failed: {e}")
        return None


def reload_weights() -> bool:
    """
    Force a fresh load from disk — called by online_trainer after it
    writes updated weights. Returns True on success.
    """
    global _AGENT
    with _LOCK:
        _AGENT = None
    return _load_agent() is not None


# ── Internals ────────────────────────────────────────────────────────
def _load_agent():
    global _AGENT
    if _AGENT is not None:
        # False means we already tried and failed — don't keep retrying
        return _AGENT or None

    with _LOCK:
        if _AGENT is not None:
            return _AGENT or None
        try:
            if not (_WEIGHTS_DIR / "model.pt").exists():
                log.info("RL: no weights at %s — policy disabled", _WEIGHTS_DIR)
                _AGENT = False
                return None
            from rl.dqn_agent import DQNAgent  # lazy: imports torch
            agent = DQNAgent.load(_WEIGHTS_DIR)
            log.info("RL: loaded weights from %s (%d training steps)",
                     _WEIGHTS_DIR, agent.steps_done)
            _AGENT = agent
            return agent
        except ImportError:
            log.info("RL: torch not installed — policy disabled")
            _AGENT = False
            return None
        except Exception as e:
            log.warning(f"RL: failed to load policy: {e}")
            _AGENT = False
            return None


def _get_loaded_agent():
    """Internal helper for online_trainer — returns the loaded agent or None."""
    return _load_agent()
