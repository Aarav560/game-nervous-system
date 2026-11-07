"""Experience log: append-only JSONL per episode + SQLite index (Architecture §12.1).

Every protocol message written here is one line in `<root>/<episode_id>.jsonl`. Decisions,
escalations and events are also indexed in `<root>/index.sqlite` so training data can be
selected without scanning every file.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from ..protocol import Decision, Escalation, Event, Goal, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (episode_id TEXT PRIMARY KEY, game TEXT, path TEXT,
    meta TEXT, started REAL);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, episode_id TEXT, bot_id TEXT,
    t REAL, goal_id TEXT, choice TEXT, acted INTEGER, margin REAL, p_top REAL, latency_ms REAL,
    model TEXT, line INTEGER);
CREATE TABLE IF NOT EXISTS escalations (id INTEGER PRIMARY KEY, episode_id TEXT, bot_id TEXT,
    t REAL, reason TEXT, fallback TEXT, line INTEGER);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, episode_id TEXT, bot_id TEXT, t REAL,
    kind TEXT, salience REAL, description TEXT, line INTEGER);
CREATE TABLE IF NOT EXISTS goals (id INTEGER PRIMARY KEY, episode_id TEXT, bot_id TEXT, t REAL,
    goal_id TEXT, intent TEXT, line INTEGER);
CREATE INDEX IF NOT EXISTS dec_ep ON decisions(episode_id);
"""


class ExperienceLog:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.sqlite")
        self.db.executescript(_SCHEMA)
        self._files: dict[str, tuple] = {}

    def start_episode(self, episode_id: str, game: str, meta: dict | None = None) -> None:
        import time
        path = self.root / f"{episode_id}.jsonl"
        self.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?)",
                        (episode_id, game, str(path), json.dumps(meta or {}), time.time()))
        self._open(episode_id)

    def _open(self, episode_id: str):
        if episode_id not in self._files:
            path = self.root / f"{episode_id}.jsonl"
            n = sum(1 for _ in open(path)) if path.exists() else 0
            self._files[episode_id] = [open(path, "a"), n]
        return self._files[episode_id]

    def write(self, msg: Message) -> None:
        entry = self._open(msg.episode_id)
        f, line = entry
        kind = type(msg).__name__.lower()
        f.write(json.dumps({"type": kind, "msg": msg.model_dump()}) + "\n")
        entry[1] += 1
        if isinstance(msg, Decision):
            p_top = max(msg.distribution.values()) if msg.distribution else 0.0
            self.db.execute(
                "INSERT INTO decisions (episode_id,bot_id,t,goal_id,choice,acted,margin,p_top,"
                "latency_ms,model,line) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (msg.episode_id, msg.bot_id, msg.t, msg.goal_id, msg.choice, int(msg.acted),
                 msg.margin, p_top, msg.latency_ms, msg.model, line))
        elif isinstance(msg, Escalation):
            self.db.execute("INSERT INTO escalations (episode_id,bot_id,t,reason,fallback,line) "
                            "VALUES (?,?,?,?,?,?)", (msg.episode_id, msg.bot_id, msg.t, msg.reason,
                                                     msg.fallback_action, line))
        elif isinstance(msg, Event):
            self.db.execute("INSERT INTO events (episode_id,bot_id,t,kind,salience,description,line)"
                            " VALUES (?,?,?,?,?,?,?)", (msg.episode_id, msg.bot_id, msg.t, msg.kind,
                                                       msg.salience, msg.description, line))
        elif isinstance(msg, Goal):
            self.db.execute("INSERT INTO goals (episode_id,bot_id,t,goal_id,intent,line) "
                            "VALUES (?,?,?,?,?,?)", (msg.episode_id, msg.bot_id, msg.t,
                                                     msg.goal_id, msg.intent, line))

    def flush(self) -> None:
        for f, _ in self._files.values():
            f.flush()
        self.db.commit()

    def close(self) -> None:
        self.flush()
        for f, _ in self._files.values():
            f.close()
        self._files.clear()
        self.db.close()

    def read(self, episode_id: str) -> Iterator[dict]:
        self.flush()
        with open(self.root / f"{episode_id}.jsonl") as f:
            for line in f:
                yield json.loads(line)


def read_episode(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)
