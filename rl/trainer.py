"""
Offline (backtest) bootstrap trainer.

Usage
-----
    # train on every supported pair, default settings
    python -m rl.trainer

    # or programmatically: from rl.trainer import train; train(epochs=20)

What it does
------------
1. Pulls historical bars per pair via server.fetch_bars (yfinance).
   If yfinance returns nothing (offline / rate-limited), falls back to
   server.demo_bars so training can still proceed for verification.
2. For each pair, builds a ForexEnv, runs N epochs of episodes,
   filling the agent's replay buffer and stepping the optimiser.
3. Writes weights to <repo>/rl/weights/{model.pt, metadata.json,
   training_log.csv}.

Design notes
------------
- We intentionally train ONE network across all pairs (pair one-hot is
  inside the state vector). This gives the model more samples and lets
  it learn pair-agnostic patterns. If you find one pair drowning out the
  others, switch to per-pair models — same code, run with --pair flag.
- Logging is intentionally minimal — a CSV per epoch — so you can plot
  it in Excel without extra deps.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Local
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import server  # noqa: E402

from rl.dqn_agent import DQNAgent, DQNConfig          # noqa: E402
from rl.forex_env import ForexEnv, EnvConfig          # noqa: E402
from rl.features import STATE_DIM, ACTION_DIM         # noqa: E402

WEIGHTS_DIR = _ROOT / "rl" / "weights"
LOG_PATH    = WEIGHTS_DIR / "training_log.csv"

log = logging.getLogger("rl.trainer")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")


def _safe_fetch(pair: str, tf: str) -> Optional[List[dict]]:
    try:
        bars = server.fetch_bars(pair, tf)
        if bars and len(bars) > 200:
            return bars
    except Exception as e:
        log.warning(f"fetch_bars({pair},{tf}) failed: {e}")
    log.info(f"  {pair}/{tf}: falling back to demo_bars (synthetic)")
    return server.demo_bars(pair, n=600)


def _run_episode(agent: DQNAgent, env: ForexEnv, train: bool = True) -> dict:
    s = env.reset()
    losses = []
    while True:
        a = agent.select_action(s, training=train)
        step = env.step(a)
        if train:
            agent.observe(s, a, step.reward, step.state, step.done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        s = step.state
        if step.done:
            break
    stats = env.episode_stats()
    stats["mean_loss"] = float(np.mean(losses)) if losses else 0.0
    return stats


def train(
    pairs: Optional[List[str]] = None,
    epochs: int = 5,
    out_dir: Path = WEIGHTS_DIR,
    cfg: Optional[DQNConfig] = None,
) -> DQNAgent:
    pairs = pairs or list(server.PAIRS.keys())
    cfg = cfg or DQNConfig(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Pairs: {pairs}")
    log.info(f"Epochs: {epochs} | state_dim={STATE_DIM} action_dim={ACTION_DIM}")
    log.info(f"Output: {out_dir}")

    # Resume if weights already exist (incremental training)
    if (out_dir / "model.pt").exists():
        log.info("Resuming from existing weights")
        agent = DQNAgent.load(out_dir)
    else:
        agent = DQNAgent(cfg)

    # Pre-fetch bars once — they don't change between epochs
    bars_by_pair = {}
    for p in pairs:
        h1 = _safe_fetch(p, "H1")
        if not h1 or len(h1) < 250:
            log.warning(f"  {p}: not enough bars, skipping")
            continue
        bars_by_pair[p] = h1

    if not bars_by_pair:
        raise RuntimeError("No usable historical bars for any pair")

    # CSV log
    log_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not log_exists:
            w.writerow(["epoch", "pair", "trades", "wins", "losses",
                        "win_rate", "net_pips", "avg_pips", "epsilon", "mean_loss"])

        t0 = time.time()
        for ep in range(1, epochs + 1):
            for pair, bars in bars_by_pair.items():
                env = ForexEnv(EnvConfig(pair=pair, bars=bars))
                stats = _run_episode(agent, env, train=True)
                w.writerow([ep, pair, stats["trades"], stats["wins"], stats["losses"],
                            stats.get("win_rate", 0), stats["net_pips"],
                            stats.get("avg_pips", 0),
                            round(agent.epsilon, 3), round(stats["mean_loss"], 4)])
                log.info(f"  ep{ep:02d} {pair}: trades={stats['trades']:3d} "
                         f"WR={stats.get('win_rate',0):5.1f}% "
                         f"net={stats['net_pips']:+7.1f}p "
                         f"loss={stats['mean_loss']:.4f} "
                         f"eps={agent.epsilon:.3f}")
            f.flush()
        elapsed = time.time() - t0
        log.info(f"Training done in {elapsed:.1f}s")

    agent.save(out_dir)
    log.info(f"Saved → {out_dir/'model.pt'}")
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Subset of pairs (default: all)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(WEIGHTS_DIR))
    args = ap.parse_args()
    train(
        pairs=args.pairs,
        epochs=args.epochs,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
