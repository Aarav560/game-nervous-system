"""Supporting modules (Architecture §7)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Any
from ..protocol import Event


class ForwardModel(ABC):
    """Cerebellum: predict near-future state; emit surprise on large error (§7.1)."""

    @abstractmethod
    def predict(self, state: Any, horizon_ms: int) -> dict[str, float]: ...

    @abstractmethod
    def surprise(self, predicted: dict[str, float], actual: Any) -> float: ...


class SalienceFilter(ABC):
    """Thalamus: score state changes, pass only salient ones up (§7.2)."""

    @abstractmethod
    def score(self, prev: Any, state: Any) -> list[Event]: ...


class WorkingMemory:
    """Last few seconds of actions/events for the WM line (§7.3)."""

    def __init__(self, maxlen: int = 50):
        self.items: deque = deque(maxlen=maxlen)

    def push(self, item: Any) -> None:
        self.items.append(item)

    def recent(self, n: int = 5) -> list[Any]:
        return list(self.items)[-n:]


class Drives(ABC):
    """Limbic system: scalar drives in [0, 1] (§7.4)."""

    @abstractmethod
    def compute(self, state: Any) -> dict[str, float]: ...


class KillSwitch:
    """Hard stop independent of every other component (§7.11)."""

    def __init__(self) -> None:
        self.tripped = False
        self.reason = ""

    def trip(self, reason: str) -> None:
        self.tripped, self.reason = True, reason
