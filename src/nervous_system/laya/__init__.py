"""Tier 2 — Laya decision layer (Architecture §5)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EscalationPolicy:
    act_threshold: float = 0.75
    min_margin: float = 0.15

    def should_act(self, dist: dict[str, float]) -> tuple[bool, str, float]:
        """Return (act, top_choice, margin). Escalate on low confidence, small margin, or 'escalate'."""
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        top, p_top = ranked[0]
        margin = p_top - (ranked[1][1] if len(ranked) > 1 else 0.0)
        act = p_top >= self.act_threshold and margin >= self.min_margin and top != "escalate"
        return act, top, margin


class DecisionModel(ABC):
    """Wraps Laya (or Jev, for baselines). Returns a calibrated distribution over options."""

    @abstractmethod
    def decide(self, state_text: str, options: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def decide_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, float]]:
        """Batch across bots in one cycle (Architecture §5 Throughput)."""


class Calibrator(ABC):
    """Temperature scaling fitted on held-out logged decisions (Architecture §12.3)."""

    @abstractmethod
    def fit(self, logits: list[list[float]], labels: list[int]) -> None: ...

    @abstractmethod
    def apply(self, dist: dict[str, float]) -> dict[str, float]: ...
