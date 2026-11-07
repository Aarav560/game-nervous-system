"""Message models mirroring protocol/*.schema.json (Architecture §10).

Protocol v0 (frozen). Changing any field here or in `protocol/*.schema.json` needs a
`SCHEMA_VERSION` bump and a note in the commit message. `tests/test_protocol.py`
checks that these models and the JSON schemas agree.
"""
from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0"

Prob = Annotated[float, Field(ge=0.0, le=1.0)]


class Message(BaseModel):
    """Header carried by every message so the experience log can be joined later."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0"] = SCHEMA_VERSION
    bot_id: str
    game: str
    episode_id: str
    t: float  # game time, seconds
    wall_t: float = Field(default_factory=time.time)  # wall-clock unix time, seconds

    def to_json(self) -> str:
        return self.model_dump_json()


class Goal(Message):
    """Brain -> Laya (Architecture §4, §11)."""

    goal_id: str
    intent: str                     # shared vocabulary verb (Architecture §9)
    expires_at: float               # game time after which the goal is stale
    on_fail: str                    # vocabulary verb to fall back to
    target: str | None = None
    constraints: list[str] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=list)   # highest first
    valid_while: list[str] = Field(default_factory=list)


class Decision(Message):
    """Laya -> reflex layer + log (Architecture §5, §12.1)."""

    goal_id: str
    state_text: str
    options: list[str] = Field(min_length=2)
    choice: str
    distribution: dict[str, Prob]
    acted: bool                     # False when the decision was escalated
    margin: float = 0.0
    latency_ms: float = 0.0
    model: str = ""                 # checkpoint id


EscalationReason = Literal[
    "low_confidence", "small_margin", "chose_escalate",
    "goal_impossible", "goal_suspended", "drive_conflict", "surprise",
]


class Escalation(Message):
    """Laya -> Brain (Architecture §5, §11)."""

    goal_id: str
    reason: EscalationReason
    summary: str                    # world summary for the brain
    fallback_action: str            # what Laya is doing while waiting
    distribution: dict[str, float] = Field(default_factory=dict)


EventKind = Literal["event", "surprise", "outcome", "correction", "override", "kill"]


class Event(Message):
    """Salience / forward model / adapter -> Brain, Laya or log (Architecture §7, §10)."""

    kind: EventKind
    salience: Prob
    description: str = ""
    data: dict = Field(default_factory=dict)


MESSAGE_TYPES: dict[str, type[Message]] = {
    "goal": Goal, "decision": Decision, "escalation": Escalation, "event": Event,
}
