"""Rule-based salience filter (Architecture §7.2): only high-salience changes reach the brain."""
from __future__ import annotations

from ..protocol import Event
from ..state import Facts
from . import SalienceFilter


class RuleSalience(SalienceFilter):
    """Scores state changes. Events with salience >= `brain_threshold` should wake the brain."""

    def __init__(self, header: dict, brain_threshold: float = 0.6, near: float = 10.0,
                 damage_spike: float = 4.0, window_s: float = 1.0):
        self.header = header
        self.brain_threshold = brain_threshold
        self.near = near
        self.damage_spike = damage_spike
        self.window_s = window_s
        self._hp_hist: list[tuple[float, float]] = []
        self._low_hp_reported = False

    def _ev(self, f: Facts, salience: float, desc: str, **data) -> Event:
        return Event(**self.header, t=f.t, kind="event", salience=round(min(1.0, salience), 3),
                     description=desc, data=data)

    def score(self, prev: Facts | None, f: Facts) -> list[Event]:
        out: list[Event] = []
        if prev is None:
            return out
        prev_ids = {e.id for e in prev.entities}
        for e in f.hostiles(self.near):
            if e.id not in prev_ids:
                s = 0.5 + 0.3 * (e.kind == "player") + 0.2 * (1 - e.dist(f.x, f.y) / self.near)
                out.append(self._ev(f, s, f"new hostile {e.kind} at {e.dist(f.x, f.y):.0f}",
                                    entity=e.id, kind_=e.kind))
        self._hp_hist = [(t, hp) for t, hp in self._hp_hist if f.t - t <= self.window_s]
        self._hp_hist.append((f.t, f.hp))
        peak = max(hp for _, hp in self._hp_hist)
        if peak - f.hp >= self.damage_spike:
            out.append(self._ev(f, 0.6 + 0.05 * (peak - f.hp), f"damage spike {peak - f.hp:.0f}",
                                damage=peak - f.hp))
            self._hp_hist = [(f.t, f.hp)]
        if f.hp_frac < 0.3 and not self._low_hp_reported:
            out.append(self._ev(f, 0.8, "health critical", hp=f.hp))
            self._low_hp_reported = True
        elif f.hp_frac > 0.6:
            self._low_hp_reported = False
        if prev.alive and not f.alive:
            out.append(self._ev(f, 1.0, "died"))
        for item, n in f.inventory.items():
            if n > prev.inventory.get(item, 0) and prev.inventory.get(item, 0) == 0:
                out.append(self._ev(f, 0.3, f"first {item}", item=item))
        return out
