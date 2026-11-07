"""Game adapters: everything game-specific lives here (Architecture §8).

Tiers 1–2 (brain, laya, modules, runtime) never import from this package; the runtime is
handed an adapter instance at wiring time (scripts/, tests/).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..protocol import Goal
from ..reflex import Reflex
from ..state import Facts

Profile = Literal["turn_based", "realtime_strategic", "realtime_twitch"]
Context = Literal["singleplayer", "own_server", "permitted_server"]
GoalStatus = Literal["active", "succeeded", "failed"]


@dataclass
class PlayRules:
    allowed: list[Context]
    permitted_servers: list[dict] = field(default_factory=list)
    own_servers: list[str] = field(default_factory=list)

    def check(self, context: Context, address: str | None = None) -> None:
        """Raise unless connecting here is allowed (Architecture §14)."""
        if context not in self.allowed:
            raise PermissionError(f"context {context!r} not allowed")
        if context == "permitted_server":
            rec = next((s for s in self.permitted_servers if s.get("address") == address), None)
            if rec is None:
                raise PermissionError(f"no recorded owner permission for {address!r}")
            if not rec.get("owner") or not rec.get("agreed_on"):
                raise PermissionError(f"permission record for {address!r} needs owner and agreed_on")
        if context == "own_server":
            if address is None:
                raise PermissionError("own_server needs an address")
            host = address.rsplit(":", 1)[0]
            if address not in self.own_servers and host not in ("127.0.0.1", "localhost", "::1"):
                raise PermissionError(f"{address!r} is not a listed own server")

    @classmethod
    def from_config(cls, cfg: dict) -> "PlayRules":
        return cls(allowed=list(cfg.get("allowed", [])),
                   permitted_servers=list(cfg.get("permitted_servers") or []),
                   own_servers=list(cfg.get("own_servers") or []))


class GameAdapter(ABC):
    game: str
    profile: Profile
    rules: PlayRules
    tick_hz: float = 20.0
    stepped: bool = True         # True: runtime advances the game via tick(); False: real time

    @abstractmethod
    def connect(self, context: Context, address: str | None = None) -> None:
        """Must call self.rules.check(...) before connecting."""

    @abstractmethod
    def read_state(self) -> Any: ...

    @abstractmethod
    def facts(self, state: Any) -> Facts:
        """Game-agnostic self/entity facts for tier-2 modules (Architecture §7.7)."""

    @abstractmethod
    def describe(self, state: Any, goal: Goal | None, tier2: dict) -> str:
        """State -> Laya text (fixed order, quantised, `none` for missing, < ~350 tokens).
        `tier2` carries `pred`, `drives` and `wm` from the nervous-system modules."""

    @abstractmethod
    def summarise(self, state: Any, goal: Goal | None) -> str:
        """World summary for the brain (under ~600 tokens)."""

    @abstractmethod
    def context(self, state: Any, goal: Goal | None) -> str:
        """Name of the current option set (e.g. combat, gathering)."""

    @abstractmethod
    def options(self, state: Any, goal: Goal | None) -> list[str]:
        """Context-valid options from the shared vocabulary (Architecture §9)."""

    @abstractmethod
    def safe_default(self, state: Any, goal: Goal | None) -> str:
        """What to do while escalated (goal's on_fail, or disengage in combat)."""

    @abstractmethod
    def execute(self, choice: str, goal: Goal | None, state: Any) -> None:
        """Start or keep the reflex for a chosen option (Laya picks which; reflex picks how)."""

    @abstractmethod
    def goal_status(self, state: Any, goal: Goal) -> GoalStatus: ...

    @abstractmethod
    def check(self, condition: str, state: Any) -> bool:
        """Evaluate one `valid_while` condition."""

    @abstractmethod
    def tick(self) -> bool:
        """Advance one game tick (stepped games) or service I/O; return False when the episode ends."""

    @abstractmethod
    def reflexes(self) -> dict[str, Reflex]: ...

    @abstractmethod
    def reward(self, prev: Any, state: Any) -> dict[str, float]: ...

    @abstractmethod
    def release_all(self) -> None:
        """Release every input and cancel every reflex; called by the kill switch."""

    def disconnect(self) -> None:
        """Optional: drop the game connection (kill switch calls it after release_all)."""
