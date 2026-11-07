"""Episodic and skill memory (Architecture §7.3). Working memory lives in __init__."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


class EpisodicMemory:
    """Episode summaries tagged by game and situation type, so lessons cross games (§13)."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, "
                        "episode_id TEXT, game TEXT, situation TEXT, summary TEXT, outcome TEXT, "
                        "reward REAL, wall_t REAL)")

    def add(self, episode_id: str, game: str, situation: str, summary: str, outcome: str,
            reward: float = 0.0) -> None:
        self.db.execute("INSERT INTO episodes (episode_id,game,situation,summary,outcome,reward,"
                        "wall_t) VALUES (?,?,?,?,?,?,?)",
                        (episode_id, game, situation, summary, outcome, reward, time.time()))
        self.db.commit()

    def recall(self, situation: str, game: str | None = None, k: int = 3) -> list[dict]:
        """Most recent episodes of this situation type; any game unless `game` is given."""
        q = "SELECT episode_id,game,situation,summary,outcome,reward FROM episodes WHERE situation=?"
        args: list = [situation]
        if game:
            q += " AND game=?"
            args.append(game)
        rows = self.db.execute(q + " ORDER BY id DESC LIMIT ?", (*args, k)).fetchall()
        keys = ("episode_id", "game", "situation", "summary", "outcome", "reward")
        return [dict(zip(keys, r)) for r in rows]


@dataclass
class Skill:
    name: str                         # usually a vocabulary option, e.g. "gather:wood"
    game: str
    params: dict = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


class SkillLibrary:
    """Named behaviours with success rates; the brain plans with them (§7.3)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.skills: dict[str, Skill] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text()):
                s = Skill(**d)
                self.skills[f"{s.game}/{s.name}"] = s

    def get(self, game: str, name: str) -> Skill:
        return self.skills.setdefault(f"{game}/{name}", Skill(name=name, game=game))

    def record(self, game: str, name: str, success: bool) -> None:
        s = self.get(game, name)
        s.attempts += 1
        s.successes += int(success)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(s) for s in self.skills.values()], indent=1))

    def summary(self, game: str) -> str:
        items = [s for s in self.skills.values() if s.game == game and s.attempts]
        return "; ".join(f"{s.name} {s.success_rate:.0%} of {s.attempts}" for s in
                         sorted(items, key=lambda s: -s.attempts)) or "none"
