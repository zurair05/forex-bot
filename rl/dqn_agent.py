"""
Double DQN agent — small MLP, replay buffer, epsilon-greedy.

Architecture choice: Double DQN over vanilla DQN to reduce the well-known
Q-value overestimation bias. PPO would be a cleaner choice for continuous
sizing, but discrete actions + replay buffer + offline data make DQN
strictly easier to train and ship. Network is intentionally small
(2 hidden layers × 64 units) — forex features are low-dimensional and a
bigger net just memorises noise.

This module is import-light: torch is the only heavy dep, and it's only
required at training time. server.py loads policy.py which lazy-imports
torch — so a torch-less deployment of the bot still runs.
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:                  # noqa: E722
    TORCH_OK = False


# ── Config ───────────────────────────────────────────────────────────
@dataclass
class DQNConfig:
    state_dim: int       = 35
    action_dim: int      = 5
    hidden: int          = 64
    lr: float            = 1e-3
    gamma: float         = 0.95
    batch_size: int      = 64
    buffer_size: int     = 50_000
    target_sync: int     = 500       # steps between target-net updates
    epsilon_start: float = 1.0
    epsilon_end: float   = 0.05
    epsilon_decay: int   = 20_000    # steps to decay over
    grad_clip: float     = 1.0
    seed: int            = 42

    def to_json(self) -> dict:
        return asdict(self)


# ── Network ──────────────────────────────────────────────────────────
if TORCH_OK:
    class QNet(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden: int):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
else:
    QNet = None  # type: ignore


# ── Replay buffer ────────────────────────────────────────────────────
@dataclass
class Transition:
    s:  np.ndarray
    a:  int
    r:  float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self): return len(self.buf)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, n: int):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s  = np.stack([t.s  for t in batch]).astype(np.float32)
        a  = np.array([t.a  for t in batch], dtype=np.int64)
        r  = np.array([t.r  for t in batch], dtype=np.float32)
        s2 = np.stack([t.s2 for t in batch]).astype(np.float32)
        d  = np.array([t.done for t in batch], dtype=np.float32)
        return s, a, r, s2, d


# ── Agent ────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Double DQN. Public API:
        select_action(state, training=True) -> int
        observe(s, a, r, s2, done)
        train_step() -> loss or None
        save(path)
        load(path)
    """

    def __init__(self, cfg: DQNConfig, device: Optional[str] = None):
        if not TORCH_OK:
            raise ImportError("torch is required to use DQNAgent. `pip install torch`")
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

        self.online = QNet(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.target = QNet(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.steps_done = 0

    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(self.steps_done / c.epsilon_decay, 1.0)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * frac

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)
        with torch.no_grad():
            t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.online(t).squeeze(0).cpu().numpy()
        return int(np.argmax(q))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return self.online(t).squeeze(0).cpu().numpy()

    def observe(self, s, a, r, s2, done):
        self.buffer.push(Transition(s, a, r, s2, done))

    def train_step(self) -> Optional[float]:
        c = self.cfg
        if len(self.buffer) < c.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(c.batch_size)
        s  = torch.from_numpy(s).to(self.device)
        a  = torch.from_numpy(a).to(self.device)
        r  = torch.from_numpy(r).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        d  = torch.from_numpy(d).to(self.device)

        # Q(s, a)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double-DQN target: argmax with online net, value from target net
        with torch.no_grad():
            a2_online = self.online(s2).argmax(dim=1, keepdim=True)
            q_s2 = self.target(s2).gather(1, a2_online).squeeze(1)
            target = r + (1.0 - d) * c.gamma * q_s2

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.opt.step()

        self.steps_done += 1
        if self.steps_done % c.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, dir_path: str | Path):
        """
        Atomic save — write to a temp path then os.replace.

        Why: the live server runs the online fine-tuner inline with
        request handling, AND a periodic retrain thread (rl.auto_loop)
        writes from the background. Both target the same model.pt. A
        non-atomic save can leave a half-written file that the policy
        loader then chokes on. os.replace is atomic on the same volume
        on every supported OS, so the worst case is "last writer wins".
        """
        import os as _os
        d = Path(dir_path); d.mkdir(parents=True, exist_ok=True)
        tmp_pt   = d / "model.pt.tmp"
        tmp_json = d / "metadata.json.tmp"
        torch.save(self.online.state_dict(), tmp_pt)
        with open(tmp_json, "w") as f:
            json.dump({
                "config": self.cfg.to_json(),
                "steps_done": self.steps_done,
            }, f, indent=2)
        _os.replace(tmp_pt,   d / "model.pt")
        _os.replace(tmp_json, d / "metadata.json")

    @classmethod
    def load(cls, dir_path: str | Path, device: Optional[str] = None) -> "DQNAgent":
        d = Path(dir_path)
        with open(d / "metadata.json") as f:
            meta = json.load(f)
        cfg = DQNConfig(**meta["config"])
        agent = cls(cfg, device=device)
        agent.online.load_state_dict(torch.load(d / "model.pt", map_location=agent.device))
        agent.target.load_state_dict(agent.online.state_dict())
        agent.steps_done = meta.get("steps_done", 0)
        agent.online.eval()
        return agent
