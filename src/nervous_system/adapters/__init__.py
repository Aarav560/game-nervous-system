"""Game adapters: everything game-specific lives here (Architecture §8)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..reflex import Reflex

Profile = Literal["turn_based", "realtime_strategic", "realtime_twitch"]
Context = Literal["singleplayer", "own_server", "permitted_server"]


@dataclass
class PlayRules:
    allowed: list[Context]
    permitted_servers: list[dict] = field(default_factory=list)

    def check(self, context: Context, address: str | None = None) -> None:
        """Raise unless connecting here is allowed (Architecture §14)."""
        if context not in self.allowed:
            raise PermissionError(f"context {context!r} not allowed")
        if context == "permitted_server":
            if not any(s.get("address") == address for s in self.permitted_servers):
                raise PermissionError(f"no recorded owner permission for {address!r}")


class GameAdapter(ABC):
    profile: Profile
    rules: PlayRules

    @abstractmethod
    def connect(self, context: Context, address: str | None = None) -> None:
        """Must call self.rules.check(...) before connecting."""

    @abstractmethod
    def read_state(self) -> Any: ...

    @abstractmethod
    def describe(self, state: Any, goal: Any) -> str:
        """State -> Laya text (fixed order, quantised, under ~350 tokens)."""

    @abstractmethod
    def summarise(self, state: Any) -> str:
        """World summary for the brain (under ~600 tokens)."""

    @abstractmethod
    def options(self, state: Any, goal: Any) -> list[str]:
        """Context-valid options from the shared vocabulary (Architecture §9)."""

    @abstractmethod
    def reflexes(self) -> dict[str, Reflex]: ...

    @abstractmethod
    def act(self, inputs: Any) -> None: ...

    @abstractmethod
    def reward(self, prev: Any, state: Any) -> dict[str, float]: ...

    @abstractmethod
    def release_all(self) -> None:
        """Release every input; called by the kill switch."""
