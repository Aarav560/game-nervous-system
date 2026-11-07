"""Message models mirroring protocol/*.schema.json (Architecture §10)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Header:
    bot_id: str
    game: str
    episode_id: str
    t: float  # game time, seconds


@dataclass
class Goal(Header):
    goal_id: str
    intent: str                     # shared vocabulary verb (Architecture §9)
    expires_at: float
    on_fail: str
    target: str | None = None
    constraints: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    valid_while: list[str] = field(default_factory=list)


@dataclass
class Decision(Header):
    goal_id: str
    state_text: str
    options: list[str]
    choice: str
    distribution: dict[str, float]
    acted: bool
    margin: float = 0.0
    latency_ms: float = 0.0
    model: str = ""


EscalationReason = Literal[
    "low_confidence", "small_margin", "chose_escalate",
    "goal_impossible", "goal_suspended", "drive_conflict", "surprise",
]


@dataclass
class Escalation(Header):
    goal_id: str
    reason: EscalationReason
    summary: str
    fallback_action: str
    distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class Event(Header):
    kind: Literal["event", "surprise", "outcome", "correction", "override", "kill"]
    salience: float
    description: str = ""
    data: dict = field(default_factory=dict)
