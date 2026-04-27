# RL — Reinforcement Learning Layer for the SMC Bot

A Double-DQN agent that overlays the existing rule-based SMC scorer.
The agent observes the same SMC/ICT features the rule-based scorer uses
and learns from realised pip-PnL whether to confirm, demote, or veto a
proposed trade.

> **Honest framing:** deep RL on forex is hard; fully replacing a
> hand-tuned scorer rarely beats it. This package is built to *augment*
> the scorer, not replace it. Set `USE_RL_POLICY=1` to enable; leave it
> off and the bot behaves exactly as before.

## Layout

| File | Purpose |
|---|---|
| `features.py` | `analyse()` → 35-dim state vector (single source of truth) |
| `forex_env.py` | Historical replay env (uses `server.analyse` directly) |
| `dqn_agent.py` | Double-DQN network, replay buffer, save/load |
| `trainer.py` | CLI: backtest bootstrap training |
| `policy.py` | Lazy-loaded inference wrapper used by `server.py` |
| `online_trainer.py` | Fine-tune one step per closed live trade |
| `weights/` | Saved `model.pt` + `metadata.json` + `training_log.csv` |

## Install

```bash
pip install -r requirements.txt   # includes torch
```

`torch` is optional — if not installed, the RL overlay disables itself
and the live server runs in pure-rule mode.

## Train (bootstrap from history)

```bash
# default: all 7 pairs, 5 epochs
python -m rl.trainer

# fewer pairs, more epochs
python -m rl.trainer --pairs EURUSD GBPUSD --epochs 20

# subsequent runs resume from weights/
python -m rl.trainer --epochs 20
```

The trainer pulls historical H1 bars via `server.fetch_bars` (yfinance).
If yfinance fails, it falls back to `server.demo_bars` so the pipeline
still completes — useful for verifying the wiring without network.

A few tens of epochs on CPU is enough for a sanity check. **For a
production model, expect to run hundreds of epochs over multiple years
of bars** and to inspect `weights/training_log.csv` for the win-rate
and net-pips trend per pair.

## Enable in the live server

```bash
# Linux/macOS
USE_RL_POLICY=1 python server.py

# Windows PowerShell
$env:USE_RL_POLICY=1; python server.py
```

Optional knob — softmax confidence above which the RL agent is allowed
to veto a rule-based BUY/SELL:

```bash
RL_VETO_THRESHOLD=0.6 USE_RL_POLICY=1 python server.py
```

## How the overlay actually works

For every signal the rule scorer emits:

| Rule says | RL says | Effect |
|---|---|---|
| BUY / SELL | HOLD (conf ≥ threshold) | **Veto** → forced WAIT |
| BUY / SELL | SAME direction | **Boost** score by 0.10 × confidence |
| BUY / SELL | OPPOSITE direction | **Demote** score by 0.10 |
| WAIT | anything | unchanged — RL never invents trades |

The chosen RL action and its Q-values are attached to every signal in
the `rl` field, so you can plot it on the dashboard or audit it via
`/api/scan`.

## Online fine-tuning

When a live trade closes (TP2 or SL — TP1 is a partial close, we wait
for the final outcome), `check_trade_outcomes` calls
`online_trainer.fine_tune_from_trade`. It:

1. Reconstructs the agent's state at signal time from `_rl_snap`
   (cached on the signal record).
2. Pushes the (state, action, reward, terminal) transition into the
   buffer.
3. Runs `tune_steps` (default 8) gradient updates.
4. Saves weights and asks `policy.py` to hot-reload.

The whole thing runs in the request thread (sub-100ms on CPU) and is
wrapped in a try/except that swallows any error — a missed online step
is harmless, corrupting the live server is not.

## Pitfalls — read these before going live

- **Reward shaping bias.** The env applies a 1-pip per-trade cost. If
  you change the live spread/slippage model, retrain — don't just hot-
  swap weights.
- **Single-network-for-all-pairs.** Pair one-hot is in the state. If
  you find one pair drowning the others (e.g. XAUUSD volume), train
  per-pair models with `--pairs <one>` and load them dynamically.
- **No SL/TP learning.** This first version learns only the
  buy/sell/wait action. Adapting SL/TP by killzone/ADR-bucket is the
  natural next milestone.
- **yfinance is rate-limited.** A long backtest on 7 pairs may stall.
  Cache the `_safe_fetch` output to disk if you iterate often.
- **Catastrophic forgetting.** A short burst of bad live trades can
  bias online fine-tuning. We cap to 8 grad steps per trade for this
  reason. If you see the agent drifting, restore an earlier `model.pt`
  from version control.

## Resetting

Delete `rl/weights/` and re-run the trainer. That's it.
