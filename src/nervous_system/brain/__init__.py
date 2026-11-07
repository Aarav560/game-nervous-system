"""Tier 1 — brain. LLM planner that emits Goals and answers Escalations (Architecture §4).

The brain sees only text: world summaries and Laya's own state text + options. It never
emits inputs — only Goals (down to Laya) and option labels (for escalations and distillation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..protocol import Escalation, Event, Goal


@dataclass
class BrainUsage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, text_in: str, text_out: str) -> None:
        """Rough token count (chars / 4) when the backend does not report usage."""
        self.calls += 1
        self.tokens_in += max(1, len(text_in) // 4)
        self.tokens_out += max(1, len(text_out) // 4)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Brain(ABC):
    name: str = "brain"

    def __init__(self) -> None:
        self.usage = BrainUsage()

    @abstractmethod
    def plan(self, mission: str, world_summary: str, header: dict) -> list[Goal]:
        """Turn a mission into a goal stack (first = current)."""

    @abstractmethod
    def on_escalation(self, esc: Escalation, state_text: str, options: list[str]) -> str:
        """Answer Laya with the option the brain would take now (logged as a Correction)."""

    @abstractmethod
    def on_event(self, ev: Event, world_summary: str, goal: Goal | None, header: dict) -> Goal | None:
        """React to a high-salience event / heartbeat; None means keep the current goal."""

    @abstractmethod
    def label(self, state_text: str, options: list[str]) -> str:
        """Choose an option for distillation labelling (Architecture §12.2)."""


def fill_goal(d: dict, header: dict, default_ttl: float = 60.0) -> Goal:
    """Goal from a brain's dict output, filling header fields and a mandatory expiry."""
    import uuid
    d = dict(d)
    d.setdefault("goal_id", f"g-{uuid.uuid4().hex[:8]}")
    d.setdefault("on_fail", "disengage")
    d.setdefault("expires_at", header["t"] + float(d.pop("ttl_s", default_ttl)))
    d.pop("ttl_s", None)
    return Goal(**header, **{k: v for k, v in d.items() if k in Goal.model_fields})
