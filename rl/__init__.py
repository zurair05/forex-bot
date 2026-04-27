"""
RL package for the SMC Signal Dashboard.

Modules:
    features        — analyse() output → fixed-length numpy state vector.
    forex_env       — Gymnasium-style historical replay environment.
    dqn_agent       — Double DQN network + replay buffer + training step.
    trainer         — CLI entrypoint for offline (backtest) training.
    online_trainer  — incremental fine-tuning hooked to live trade closes.
    policy          — runtime inference wrapper used by server.py.

Design note
-----------
State extraction is the single source of truth shared by training and
inference. Everything funnels through features.extract_state() so that a
trained model is guaranteed to see the same vector layout in both regimes.
"""
__all__ = [
    "features",
    "forex_env",
    "dqn_agent",
    "trainer",
    "online_trainer",
    "policy",
]
