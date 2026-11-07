"""Opponent play styles + an online style tracker (GridCraft PvP).

Styles give the league and the PvP dataset a varied population of opponents; the tracker
summarises what an opponent has been doing over the last few seconds so a policy (and the
game-sense models) can adapt to it.
"""
from __future__ import annotations

import math
import random

import numpy as np

from .world import Agent, World, scripted_pvp_ai

STYLES = ["practice", "aggro", "kiter", "healer", "counter", "erratic"]


def _vec(a: Agent, t: Agent):
    d = a.dist(t)
    return d, (t.x - a.x) / max(d, 1e-6), (t.y - a.y) / max(d, 1e-6)


def styled_ai(style: str, target_id: str, rng: random.Random | None = None):
    """Scripted opponents with distinct, recognisable habits."""
    rng = rng or random.Random(0)
    if style == "practice":
        return scripted_pvp_ai(target_id, rng=rng)

    def ai(w: World, a: Agent) -> None:
        a.clear_controls()
        t = w.agents.get(target_id)
        if t is None or not t.alive:
            return
        if a.offhand is None and a.count("totem"):
            a.offhand = "totem"
        d, dx, dy = _vec(a, t)
        side = 1 if (w.tick_n // 25) % 2 else -1
        if a.eating:
            a.move = (-dx, -dy)
            return
        a.attack = t.id
        if style == "aggro":                     # rush, never retreats, eats only when safe
            if a.hp <= 6 and a.count("golden_apple") and d > 6:
                w.start_eating(a, "golden_apple")
            a.move, a.sprint = (dx, dy), True
        elif style == "kiter":                   # stay at the edge of reach, dart in on cooldown
            if a.cd <= 1 and t.cd > 3:
                a.move, a.sprint = (dx, dy), True
            elif d < 3.6:
                a.move, a.sprint = (-dx + 0.4 * -dy, -dy + 0.4 * dx), True
            else:
                a.move = (side * -dy, side * dx)
        elif style == "healer":                  # disengages early and eats a lot
            if a.hp <= 13 and a.count("golden_apple"):
                if d > 4.5:
                    w.start_eating(a, "golden_apple")
                else:
                    a.move, a.sprint = (-dx, -dy), True
                    a.attack = None
                return
            a.move, a.sprint = ((dx, dy), True) if a.cd <= 2 else ((-dx - 0.5 * dy, -dy + 0.5 * dx), False)
        elif style == "counter":                 # waits for the opponent to swing, then punishes
            if t.cd > 4 and d < 5:
                a.move, a.sprint = (dx, dy), True
            elif d < 3.4:
                a.move = (-dx, -dy)
            else:
                a.move = (side * -dy + 0.2 * dx, side * dx + 0.2 * dy)
        elif style == "erratic":                 # unpredictable movement, swings whenever in reach
            ang = rng.uniform(0, 2 * math.pi)
            a.move = (dx + 1.2 * math.cos(ang), dy + 1.2 * math.sin(ang))
            a.sprint = rng.random() < 0.5
            if a.hp <= 8 and a.count("golden_apple") and d > 4:
                w.start_eating(a, "golden_apple")
    return ai


TRACK_DIM = 8


class StyleTracker:
    """Exponential moving averages (~2 s) of the opponent's observable habits."""

    def __init__(self, tau_s: float = 2.0, dt: float = 0.05):
        self.k = dt / tau_s
        self.f = np.zeros(TRACK_DIM, dtype=np.float32)
        self.f[1] = 1.0
        self._opp_cd = 0
        self._eat_hp = 1.0
        self.n = 0

    def update(self, me: Agent, opp: Agent) -> np.ndarray:
        d, ux, uy = _vec(opp, me)
        approach = opp.vx * ux + opp.vy * uy
        swung = opp.cd > self._opp_cd                  # cooldown just reset -> a swing happened
        self._opp_cd = opp.cd
        if opp.eating:
            self._eat_hp = opp.hp / 20
        x = np.array([
            approach / 6, min(d, 6) / 6, float(opp.sprint), float(approach < -0.5),
            float(swung) * 20 / 2, float(swung and me.cd > 0), float(bool(opp.eating)), self._eat_hp,
        ], dtype=np.float32)
        self.f += self.k * (x - self.f)
        self.n += 1
        return self.f.copy()
