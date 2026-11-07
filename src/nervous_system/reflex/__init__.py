"""Tier 3 — reflex contract (Architecture §6). Learned or scripted; same interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Literal

Status = Literal["running", "succeeded", "failed", "stuck"]


class Reflex(ABC):
    name: str

    @abstractmethod
    def start(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def tick(self, state: Any) -> Any:
        """Called every game tick/frame; returns game inputs for the adapter's actuator."""

    @abstractmethod
    def status(self) -> Status: ...

    @abstractmethod
    def cancel(self) -> None:
        """Must stop within one tick (inhibition, Architecture §7.6)."""
