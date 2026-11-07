"""Tier 1 — brain. LLM planner that emits Goals and answers Escalations (Architecture §4)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..protocol import Goal, Escalation, Event


class Brain(ABC):
    @abstractmethod
    def plan(self, mission: str, world_summary: str) -> list[Goal]:
        """Turn a mission into a goal stack."""

    @abstractmethod
    def on_escalation(self, esc: Escalation) -> Goal | None:
        """Answer Laya. Return None if the situation has resolved (stale reply)."""

    @abstractmethod
    def on_event(self, ev: Event) -> Goal | None:
        """React to a high-salience event; None means keep the current goal."""

    @abstractmethod
    def label(self, state_text: str, options: list[str]) -> str:
        """Choose an option for distillation labelling (Architecture §12.2)."""
