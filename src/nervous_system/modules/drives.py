"""Drives (Architecture §7.4): scalars in [0, 1] computed from Facts each tick."""
from __future__ import annotations

from ..state import Facts
from . import Drives


def _clip(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 2)


class BasicDrives(Drives):
    def __init__(self, threat_radius: float = 8.0):
        self.threat_radius = threat_radius

    def compute(self, f: Facts) -> dict[str, float]:
        threats = f.hostiles(self.threat_radius)
        outnumbered = max(0, len(threats) - f.allies - 1)
        fear = (1 - f.hp_frac) * 0.7 + 0.15 * len(threats) + 0.15 * outnumbered
        if threats and f.hp_frac < 0.35:
            fear += 0.3
        return {
            "fear": _clip(fear if threats or f.hp_frac < 0.5 else fear * 0.5),
            "hunger": _clip(1 - f.hunger_frac),
            "fatigue": _clip(1 - f.durability),
            "curiosity": _clip((1 - f.explored_frac) * min(1.0, f.idle_s / 10.0)),
        }

    @staticmethod
    def conflict(d: dict[str, float]) -> bool:
        """Two strong drives pulling different ways -> worth escalating (§5)."""
        strong = [k for k, v in d.items() if v >= 0.7]
        return len(strong) >= 2
