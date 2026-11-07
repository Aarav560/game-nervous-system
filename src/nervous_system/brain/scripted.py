"""Scripted brain: a rule-based stand-in for Claude, driven by a game playbook.

Used when no LLM is configured (CI, offline benchmarks, this repo's recorded results). The
playbook is injected at wiring time, so this module imports no adapter code. It sees exactly
what an LLM brain would see — world summaries and Laya's state text + options — and its
usage is counted with the same rough token estimate.
"""
from __future__ import annotations

from typing import Protocol

from ..protocol import Escalation, Event, Goal
from . import Brain, fill_goal


class Playbook(Protocol):
    def plan(self, mission: str, world_summary: str) -> list[dict]: ...
    def label(self, state_text: str, options: list[str]) -> str: ...
    def on_event(self, description: str, world_summary: str, goal_intent: str | None) -> dict | None: ...


class ScriptedBrain(Brain):
    name = "scripted"

    def __init__(self, playbook: Playbook, mission_ttl_s: float = 600.0):
        super().__init__()
        self.playbook = playbook
        self.mission_ttl_s = mission_ttl_s

    def plan(self, mission: str, world_summary: str, header: dict) -> list[Goal]:
        goals = self.playbook.plan(mission, world_summary)
        self.usage.add(mission + world_summary, repr(goals))
        return [fill_goal(g, header, self.mission_ttl_s) for g in goals]

    def label(self, state_text: str, options: list[str]) -> str:
        choice = self.playbook.label(state_text, options)
        if choice not in options:
            raise ValueError(f"playbook chose {choice!r}, not in {options}")
        self.usage.add(state_text + " ".join(options), choice)
        return choice

    def on_escalation(self, esc: Escalation, state_text: str, options: list[str]) -> str:
        return self.label(state_text, options)

    def on_event(self, ev: Event, world_summary: str, goal: Goal | None, header: dict) -> Goal | None:
        d = self.playbook.on_event(ev.description, world_summary, goal.intent if goal else None)
        self.usage.add(ev.description + world_summary, repr(d))
        return fill_goal(d, header, self.mission_ttl_s) if d else None
