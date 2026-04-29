"""
SQLite persistence for the SMC Signal Dashboard.

Two tables:
    signals_history  — one row per closed trade (TP1 / TP2 / SL / cancel)
    session_stats    — single-row counters keyed by date (yyyy-mm-dd)

Why SQLite (vs JSON / pickle)
-----------------------------
- Atomic writes (no half-flushed file)
- Survives a crash mid-write
- Queryable for backtests, win-rate analysis, RL retraining
- Zero deps — sqlite3 is in the stdlib
- File-based, single user, no server to run

Threadsafety
------------
Each call opens its own connection (`sqlite3.connect(... isolation_level=None ...)`).
Concurrent writes from the auto-scan thread + request handlers + RL
fine-tuner are all serialised by SQLite's own file lock.

Failure model
-------------
Any DB error logs a warning and returns a sensible default (0, [], etc.).
The live signal pipeline never crashes because of a bad DB call.
"""
from __future__ import annotations

import logging
import sqlite3
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("db")

_DB_PATH = Path(__file__).resolve().parent / "signals.db"
_LOCK    = threading.Lock()


# ── Schema ─────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pair            TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    score_pct       INTEGER,
    entry           REAL,
    sl              REAL,
    tp1             REAL,
    tp2             REAL,
    sl_pips         REAL,
    spread_pips     REAL,
    status          TEXT,                      -- tp1_hit / tp2_hit / sl_hit / cancelled
    pnl_pips        REAL,
    result          TEXT,                      -- win / loss
    opened_at       TEXT,
    filled_at       TEXT,
    hit_at          TEXT,
    hit_price       REAL,
    conf            TEXT,                      -- JSON-encoded list
    rl_action       TEXT,                      -- last RL action seen at signal time
    rl_confidence   REAL,
    extra           TEXT                       -- JSON blob for anything else
);
CREATE INDEX IF NOT EXISTS idx_signals_pair_hit  ON signals_history(pair, hit_at);
CREATE INDEX IF NOT EXISTS idx_signals_result    ON signals_history(result);

CREATE TABLE IF NOT EXISTS session_stats (
    day              TEXT PRIMARY KEY,         -- yyyy-mm-dd UTC
    total            INTEGER NOT NULL DEFAULT 0,
    wins             INTEGER NOT NULL DEFAULT 0,
    losses           INTEGER NOT NULL DEFAULT 0,
    net_pips         REAL    NOT NULL DEFAULT 0,
    consecutive_loss INTEGER NOT NULL DEFAULT 0,
    daily_pnl_pips   REAL    NOT NULL DEFAULT 0,
    best_trade_pips  REAL    NOT NULL DEFAULT 0,
    worst_trade_pips REAL    NOT NULL DEFAULT 0,
    total_pips_won   REAL    NOT NULL DEFAULT 0,
    total_pips_lost  REAL    NOT NULL DEFAULT 0,
    updated_at       TEXT
);
"""


# ── Connection ─────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH), timeout=10, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA foreign_keys=ON;")
    return c


def init() -> None:
    """Idempotent — safe to call on every server start."""
    with _LOCK:
        try:
            with _conn() as c:
                c.executescript(_SCHEMA)
            log.info(f"DB ready at {_DB_PATH}")
        except Exception as e:
            log.warning(f"DB init failed: {e}")


# ── Signal writes ──────────────────────────────────────────────────────
def insert_signal(trade: Dict[str, Any]) -> bool:
    """Append a closed trade. Caller passes a dict matching server's signal_history format."""
    fields = ("pair", "direction", "score_pct", "entry", "sl", "tp1", "tp2",
              "sl_pips", "spread_pips", "status", "pnl_pips", "result",
              "opened_at", "filled_at", "hit_at", "hit_price",
              "conf", "rl_action", "rl_confidence", "extra")
    row = {k: trade.get(k) for k in fields}
    # JSON-encode list/dict fields
    if isinstance(row.get("conf"), (list, dict)):
        row["conf"] = json.dumps(row["conf"])
    if isinstance(row.get("extra"), (list, dict)):
        row["extra"] = json.dumps(row["extra"])
    cols = ", ".join(fields)
    q    = ", ".join(["?"] * len(fields))
    try:
        with _LOCK, _conn() as c:
            c.execute(f"INSERT INTO signals_history ({cols}) VALUES ({q})",
                      [row[k] for k in fields])
        return True
    except Exception as e:
        log.warning(f"DB insert_signal failed: {e}")
        return False


def load_recent_signals(limit: int = 500) -> List[Dict[str, Any]]:
    """Hydrate _signal_history on startup — newest first."""
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM signals_history ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("conf", "extra"):
                v = d.get(k)
                if isinstance(v, str) and v.startswith(("[", "{")):
                    try: d[k] = json.loads(v)
                    except Exception: pass
            out.append(d)
        return out
    except Exception as e:
        log.warning(f"DB load_recent_signals failed: {e}")
        return []


def count_signals() -> int:
    try:
        with _conn() as c:
            return c.execute("SELECT COUNT(*) FROM signals_history").fetchone()[0]
    except Exception:
        return 0


# ── Session stats ──────────────────────────────────────────────────────
def upsert_session_stats(day: str, stats: Dict[str, Any]) -> bool:
    cols = ("total", "wins", "losses", "net_pips", "consecutive_loss",
            "daily_pnl_pips", "best_trade_pips", "worst_trade_pips",
            "total_pips_won", "total_pips_lost", "updated_at")
    vals = [stats.get(c, 0) for c in cols]
    try:
        with _LOCK, _conn() as c:
            c.execute(f"""
                INSERT INTO session_stats (day, {", ".join(cols)})
                VALUES (?, {", ".join(["?"]*len(cols))})
                ON CONFLICT(day) DO UPDATE SET
                  {", ".join(f"{k}=excluded.{k}" for k in cols)}
            """, [day, *vals])
        return True
    except Exception as e:
        log.warning(f"DB upsert_session_stats failed: {e}")
        return False


def load_today_stats(day: str) -> Optional[Dict[str, Any]]:
    try:
        with _conn() as c:
            r = c.execute("SELECT * FROM session_stats WHERE day = ?", (day,)).fetchone()
            return dict(r) if r else None
    except Exception:
        return None


def load_all_stats() -> List[Dict[str, Any]]:
    try:
        with _conn() as c:
            rows = c.execute("SELECT * FROM session_stats ORDER BY day DESC").fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
