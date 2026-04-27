"""
Autonomous RL training loop.

Runs two daemon threads in the background of the live server:

  1. Bootstrap thread (one-shot)
     If `weights/model.pt` doesn't exist yet, runs the offline backtest
     trainer immediately. The live server keeps responding to /api/scan
     while this happens — the rule-based scorer is always available;
     the RL overlay simply stays inert until the model finishes training
     and `policy.reload_weights()` swaps it in.

  2. Periodic retrain thread (long-running)
     Every `interval_hours` (default 24h), runs another training pass
     that *resumes* from the current weights (the trainer auto-detects
     existing model.pt). This keeps the policy fresh as new yfinance
     bars print, without ever requiring a server restart.

Why this is safe to run while online fine-tuning is also active
---------------------------------------------------------------
- Both code paths save to the same `model.pt` file.
- DQNAgent.save uses atomic temp-file + os.replace, so concurrent
  writers can't corrupt the file — last write wins.
- When the periodic retrain finishes, we call `policy.reload_weights()`,
  which is what the online fine-tuner does too. The in-memory policy
  used by `make_signal()` is reloaded atomically.

Failure mode
------------
If yfinance is rate-limiting, network is down, or torch isn't
installed, the auto-loop swallows the error and tries again at the next
interval. The server is never affected.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("rl.auto_loop")
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_LOCK    = threading.Lock()
_STARTED = False


# ── Public API ────────────────────────────────────────────────────────
def start(interval_hours: float = 24.0, bootstrap_epochs: int = 5,
          retrain_epochs: int = 2) -> bool:
    """
    Idempotent — safe to call from server startup. Returns True if the
    background threads were just started (False if already running).
    """
    global _STARTED
    with _LOCK:
        if _STARTED:
            return False
        _STARTED = True

    threading.Thread(
        target=_bootstrap,
        args=(bootstrap_epochs,),
        name="rl-bootstrap",
        daemon=True,
    ).start()
    threading.Thread(
        target=_periodic,
        args=(interval_hours, retrain_epochs),
        name="rl-periodic",
        daemon=True,
    ).start()

    log.info(
        f"RL auto-loop started — bootstrap (if needed) + periodic "
        f"retrain every {interval_hours}h"
    )
    return True


# ── Threads ───────────────────────────────────────────────────────────
def _bootstrap(epochs: int) -> None:
    """One-shot: train from history if no weights exist yet."""
    # Wait briefly so the Flask server has finished binding to the port
    # before we hammer yfinance. Avoids a confusing log interleave.
    time.sleep(20)

    if (_WEIGHTS_DIR / "model.pt").exists():
        log.info("RL bootstrap: weights already present — skipping")
        return

    log.info("RL bootstrap: no weights yet — kicking off initial training "
             "(epochs=%d). The bot uses pure-rule signals until this "
             "completes.", epochs)
    try:
        from rl.trainer import train
        train(epochs=epochs)
    except Exception as e:
        log.warning(f"RL bootstrap failed (will retry on next periodic cycle): {e}")
        return

    # Bootstrap done — tell the live policy to load the fresh weights
    try:
        from rl.policy import reload_weights
        if reload_weights():
            log.info("RL bootstrap: complete — policy is now live")
        else:
            log.info("RL bootstrap: weights saved but reload reported unavailable")
    except Exception as e:
        log.warning(f"RL bootstrap: weights saved but reload failed: {e}")


def _periodic(interval_hours: float, epochs: int) -> None:
    """Re-train every `interval_hours`, resuming from the current weights."""
    interval_s = max(60.0, float(interval_hours) * 3600.0)
    while True:
        try:
            time.sleep(interval_s)
        except Exception:
            return

        log.info("RL periodic retrain: starting top-up pass (epochs=%d)", epochs)
        try:
            from rl.trainer import train
            train(epochs=epochs)
        except Exception as e:
            log.warning(f"RL periodic retrain failed: {e}")
            continue

        try:
            from rl.policy import reload_weights
            reload_weights()
            log.info("RL periodic retrain: weights swapped in")
        except Exception as e:
            log.warning(f"RL periodic retrain: reload_weights failed: {e}")


def status() -> dict:
    """Cheap probe — used by /api/stats so the dashboard can show RL state."""
    weights_exists = (_WEIGHTS_DIR / "model.pt").exists()
    info = {
        "auto_loop_running": _STARTED,
        "weights_present":   weights_exists,
        "weights_dir":       str(_WEIGHTS_DIR),
    }
    if weights_exists:
        try:
            info["weights_mtime"] = os.path.getmtime(_WEIGHTS_DIR / "model.pt")
        except Exception:
            pass
    return info
