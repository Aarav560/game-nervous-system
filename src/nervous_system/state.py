"""Game-agnostic state facts (Architecture §7.7 proprioception).

Adapters turn their raw state into `Facts`; every tier-2 module (forward model, salience,
drives, memory) reads only this, never adapter types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Entity:
    id: str
    kind: str                      # e.g. "zombie", "player", "item"
    hostile: bool
    x: float
    y: float
    vx: float = 0.0                # units per second
    vy: float = 0.0
    hp: float | None = None
    hp_max: float | None = None
    facing_self: bool = False
    attack_ready: bool = False     # cooldown ready (for crit/aim-lock style predictions)

    def dist(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class Facts:
    t: float                       # game time, seconds
    x: float
    y: float
    hp: float
    hp_max: float
    vx: float = 0.0
    vy: float = 0.0
    hunger: float = 20.0
    hunger_max: float = 20.0
    entities: list[Entity] = field(default_factory=list)
    inventory: dict[str, int] = field(default_factory=dict)
    durability: float = 1.0        # worst tool durability fraction
    allies: int = 0
    reflex: str = "none"           # currently running reflex
    reflex_status: str = "none"
    explored_frac: float = 0.0
    idle_s: float = 0.0
    alive: bool = True
    extras: dict = field(default_factory=dict)

    @property
    def hp_frac(self) -> float:
        return max(0.0, min(1.0, self.hp / self.hp_max)) if self.hp_max else 0.0

    @property
    def hunger_frac(self) -> float:
        return max(0.0, min(1.0, self.hunger / self.hunger_max)) if self.hunger_max else 1.0

    def hostiles(self, radius: float = math.inf) -> list[Entity]:
        return sorted((e for e in self.entities if e.hostile and e.dist(self.x, self.y) <= radius),
                      key=lambda e: e.dist(self.x, self.y))

    def nearest_hostile(self) -> Entity | None:
        h = self.hostiles()
        return h[0] if h else None
