"""Forward model v1 (Architecture §7.1): physics extrapolation + analytic combat predictions."""
from __future__ import annotations

import math
from collections import deque

from ..state import Facts
from . import ForwardModel


class PhysicsForwardModel(ForwardModel):
    """Constant-velocity extrapolation of every entity, plus `PRED` features:

    - `opp_aim_lock`: nearest hostile is facing us and close enough to hit soon
    - `crit_incoming`: nearest hostile has its attack ready, is closing in, and is in reach
    - `hp_loss_1s`: expected self hp loss over the next second from hostiles in reach

    `surprise()` compares the prediction made `horizon` ago with what actually happened.
    """

    def __init__(self, horizon_ms: int = 300, reach: float = 3.0, threat_dps: float = 3.0):
        self.horizon_ms = horizon_ms
        self.reach = reach
        self.threat_dps = threat_dps
        self._pending: deque = deque()   # (due_t, predicted positions, predicted hp)

    def predict(self, f: Facts, horizon_ms: int | None = None) -> dict[str, float]:
        h = (horizon_ms or self.horizon_ms) / 1000.0
        out: dict[str, float] = {}
        opp = f.nearest_hostile()
        if opp is None:
            out.update(opp_aim_lock=0.0, crit_incoming=0.0, hp_loss_1s=0.0)
        else:
            d_now = opp.dist(f.x, f.y)
            px, py = opp.x + opp.vx * h, opp.y + opp.vy * h
            sx, sy = f.x + f.vx * h, f.y + f.vy * h
            d_next = math.hypot(px - sx, py - sy)
            closing = max(0.0, (d_now - d_next) / max(h, 1e-6))
            in_reach = 1.0 / (1.0 + math.exp(4.0 * (d_next - self.reach)))
            out["opp_aim_lock"] = round(in_reach * (0.95 if opp.facing_self else 0.3), 2)
            out["crit_incoming"] = round(in_reach * (0.9 if opp.attack_ready else 0.2)
                                         * min(1.0, 0.4 + closing / 4.0), 2)
            n_reach = sum(1 for e in f.hostiles(self.reach + 1.0))
            out["hp_loss_1s"] = round(self.threat_dps * n_reach * in_reach, 1)
        due = f.t + h
        self._pending.append((due, {e.id: (e.x + e.vx * h, e.y + e.vy * h) for e in f.entities},
                              f.hp - out["hp_loss_1s"] * h))
        return out

    def surprise(self, predicted: dict[str, float], actual: Facts) -> float:
        """Prediction error in [0, 1] for predictions now due. `predicted` is unused here
        (the model keeps its own queue) but kept for the ABC signature."""
        score = 0.0
        while self._pending and self._pending[0][0] <= actual.t + 1e-9:
            _, pos, hp_pred = self._pending.popleft()
            ents = {e.id: e for e in actual.entities}
            errs = [math.hypot(ents[i].x - x, ents[i].y - y) for i, (x, y) in pos.items() if i in ents]
            pos_err = max(errs) if errs else 0.0
            new_hostiles = sum(1 for e in actual.entities if e.hostile and e.id not in pos
                               and e.dist(actual.x, actual.y) < 8)
            hp_err = max(0.0, hp_pred - actual.hp)        # took more damage than predicted
            score = max(score, min(1.0, pos_err / 4.0 + hp_err / 8.0 + 0.5 * new_hostiles))
        return round(score, 3)
