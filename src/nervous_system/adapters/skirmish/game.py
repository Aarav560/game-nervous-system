"""Skirmish: a small top-down arena shooter (our own design) — the Phase 7 second game.

Different mechanics from GridCraft on purpose: ranged hit chance by distance, line of sight and
cover, magazines and reloads, medkits, supply crates. 20 Hz ticks, 2D, cover blocks LoS.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

DT = 0.05


@dataclass
class Unit:
    id: str
    kind: str                          # "player" | "drone" | "gunner"
    team: str
    x: float
    y: float
    hp: float = 100.0
    hp_max: float = 100.0
    speed: float = 5.0
    mag: int = 8
    mag_size: int = 8
    reserve: int = 24
    fire_cd: int = 0
    fire_every: int = 6                # ticks between shots
    reload_t: int = 0                  # ticks of reload left
    reload_ticks: int = 30
    damage: float = 15.0
    medkits: int = 1
    heal_t: int = 0
    accuracy: float = 0.9
    ai: Callable | None = None
    vx: float = 0.0
    vy: float = 0.0
    fx: float = 1.0
    fy: float = 0.0
    alive: bool = True
    move: tuple = (0.0, 0.0)
    shoot: str | None = None
    crates: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0

    def dist(self, o: "Unit") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def clear(self) -> None:
        self.move, self.shoot = (0.0, 0.0), None


class Arena:
    def __init__(self, size: float = 60.0, seed: int = 0):
        self.size = size
        self.rng = random.Random(seed)
        self.tick_n = 0
        self.units: dict[str, Unit] = {}
        self.cover: set[tuple[int, int]] = set()
        self.crates: set[tuple[int, int]] = set()
        self.medpacks: set[tuple[int, int]] = set()
        self.places: dict[str, tuple[float, float]] = {}
        self.waves: dict | None = None
        self._n = 0

    @property
    def t(self) -> float:
        return self.tick_n * DT

    def add(self, u: Unit) -> Unit:
        self.units[u.id] = u
        return u

    def los(self, a: Unit, b: Unit) -> bool:
        d = a.dist(b)
        steps = max(1, int(d / 0.5))
        for i in range(1, steps):
            x = a.x + (b.x - a.x) * i / steps
            y = a.y + (b.y - a.y) * i / steps
            if (int(x), int(y)) in self.cover:
                return False
        return True

    def hit_chance(self, a: Unit, b: Unit) -> float:
        if not self.los(a, b):
            return 0.0
        d = a.dist(b)
        moving = math.hypot(b.vx, b.vy) > 1.0
        return max(0.05, min(0.95, a.accuracy - 0.025 * d - (0.15 if moving else 0.0)))

    def nearest_cover(self, x: float, y: float, away_from: Unit | None = None):
        best, bd = None, 1e9
        for cx, cy in self.cover:
            # stand on the far side of the cover block from the threat
            if away_from is not None:
                ux, uy = cx + 0.5 - away_from.x, cy + 0.5 - away_from.y
                n = math.hypot(ux, uy) or 1
                px, py = cx + 0.5 + ux / n * 1.2, cy + 0.5 + uy / n * 1.2
            else:
                px, py = cx + 0.5, cy + 0.5
            d = math.hypot(px - x, py - y)
            if d < bd and (int(px), int(py)) not in self.cover:
                best, bd = (px, py), d
        return best, bd

    def step(self) -> None:
        for u in list(self.units.values()):
            if u.alive and u.ai:
                u.ai(self, u)
        for u in self.units.values():
            if not u.alive:
                continue
            mx, my = u.move
            n = math.hypot(mx, my)
            sp = u.speed * (0.4 if u.heal_t or u.reload_t else 1.0)
            if n > 1e-6:
                u.vx, u.vy = mx / n * sp, my / n * sp
                u.fx, u.fy = mx / n, my / n
            else:
                u.vx = u.vy = 0.0
            nx = min(self.size - 0.5, max(0.5, u.x + u.vx * DT))
            ny = min(self.size - 0.5, max(0.5, u.y + u.vy * DT))
            if (int(nx), int(ny)) not in self.cover:
                u.x, u.y = nx, ny
            elif (int(nx), int(u.y)) not in self.cover:          # slide along cover
                u.x = nx
                u.y = min(self.size - 0.5, max(0.5, u.y + math.copysign(sp * DT, u.vy or 1.0)))
            elif (int(u.x), int(ny)) not in self.cover:
                u.y = ny
                u.x = min(self.size - 0.5, max(0.5, u.x + math.copysign(sp * DT, u.vx or 1.0)))
            if u.fire_cd:
                u.fire_cd -= 1
            if u.reload_t:
                u.reload_t -= 1
                if u.reload_t == 0:
                    take = min(u.mag_size - u.mag, u.reserve)
                    u.mag += take
                    u.reserve -= take
            if u.heal_t:
                u.heal_t -= 1
                u.hp = min(u.hp_max, u.hp + 30.0 / 40)
            cell = (int(u.x), int(u.y))
            if u.kind == "player":
                if cell in self.crates:
                    self.crates.discard(cell)
                    u.crates += 1
                    u.reserve += 16
                if cell in self.medpacks:
                    self.medpacks.discard(cell)
                    u.medkits += 1
        for u in list(self.units.values()):
            if u.alive and u.shoot:
                self._fire(u)
        self._waves()
        self.tick_n += 1

    def _fire(self, u: Unit) -> None:
        t = self.units.get(u.shoot)
        if t is None or not t.alive or u.fire_cd or u.reload_t or u.heal_t or u.mag <= 0:
            return
        if not self.los(u, t):                  # nobody fires blind
            return
        u.mag -= 1
        u.fire_cd = u.fire_every
        u.fx, u.fy = (t.x - u.x) / max(u.dist(t), 1e-6), (t.y - u.y) / max(u.dist(t), 1e-6)
        if self.rng.random() < self.hit_chance(u, t):
            t.hp -= u.damage
            t.damage_taken += u.damage
            u.damage_dealt += u.damage
            if t.heal_t:
                t.heal_t = 0
            if t.hp <= 0:
                t.hp, t.alive = 0.0, False

    def start_reload(self, u: Unit) -> bool:
        if u.reload_t or u.reserve <= 0 or u.mag >= u.mag_size:
            return False
        u.reload_t = u.reload_ticks
        return True

    def start_heal(self, u: Unit) -> bool:
        if u.heal_t or u.medkits <= 0 or u.hp >= u.hp_max:
            return False
        u.medkits -= 1
        u.heal_t = 40
        return True

    def _waves(self) -> None:
        w = self.waves
        if not w:
            return
        w["clock"] = w.get("clock", 0.0) + DT
        if w["clock"] < w["every_s"]:
            return
        w["clock"] = 0.0
        if sum(1 for u in self.units.values() if u.kind == "drone" and u.alive) >= w["max"]:
            return
        p = self.units.get("bot")
        if p is None:
            return
        ang = self.rng.uniform(0, 2 * math.pi)
        x = min(self.size - 1, max(1, p.x + 22 * math.cos(ang)))
        y = min(self.size - 1, max(1, p.y + 22 * math.sin(ang)))
        if (int(x), int(y)) in self.cover:
            return
        self._n += 1
        self.add(make_drone(f"d{self._n}", x, y))


def shooter_ai(target: str, keep: float = 12.0, rng: random.Random | None = None):
    rng = rng or random.Random(0)

    def ai(arena: Arena, u: Unit) -> None:
        u.clear()
        t = arena.units.get(target)
        if t is None or not t.alive:
            return
        d = u.dist(t)
        if u.mag == 0:
            arena.start_reload(u)
        dx, dy = (t.x - u.x) / max(d, 1e-6), (t.y - u.y) / max(d, 1e-6)
        if u.hp < 40 and u.medkits and not arena.los(t, u):
            arena.start_heal(u)
        if not arena.los(u, t) or d > keep + 2:
            u.move = (dx + rng.uniform(-0.3, 0.3), dy + rng.uniform(-0.3, 0.3))
        elif d < keep - 3:
            u.move = (-dx, -dy)
        else:
            side = 1 if (arena.tick_n // 30) % 2 else -1
            u.move = (-dy * side, dx * side)
        u.shoot = t.id
    return ai


def make_drone(id_: str, x: float, y: float) -> Unit:
    return Unit(id=id_, kind="drone", team="red", x=x, y=y, hp=45, hp_max=45, speed=3.5, damage=8,
                accuracy=0.6, fire_every=10, reserve=999, medkits=0, ai=shooter_ai("bot", 10.0))


# ------------------------------------------------------------------------------ scenarios
@dataclass
class Scenario:
    name: str
    mission: str
    time_limit_s: float
    setup: Callable[[Arena], None]
    success: Callable[[Arena], bool]
    failed: Callable[[Arena], bool]


def _arena(a: Arena) -> None:
    rng = a.rng
    for _ in range(28):
        x, y = rng.randint(4, int(a.size) - 6), rng.randint(4, int(a.size) - 6)
        for dx in range(rng.randint(1, 3)):
            a.cover.add((x + dx, y))
    a.places["spawn"] = (30.0, 30.0)
    for c in list(a.cover):
        if math.hypot(c[0] - 30, c[1] - 30) < 3:
            a.cover.discard(c)


def _player(a: Arena, **kw) -> Unit:
    return a.add(Unit(id="bot", kind="player", team="blue", x=30, y=30, **kw))


def hold_out() -> Scenario:
    def setup(a: Arena) -> None:
        _arena(a)
        _player(a, medkits=2, reserve=48)
        a.waves = {"every_s": 6.0, "max": 3}
        for _ in range(3):
            a.medpacks.add((a.rng.randint(5, 55), a.rng.randint(5, 55)))
    return Scenario("hold_out", "hold out for 2 minutes", 120.0, setup,
                    lambda a: a.units["bot"].alive and a.t >= 120.0 - 1e-6, lambda a: not a.units["bot"].alive)


def gunfight() -> Scenario:
    def setup(a: Arena) -> None:
        _arena(a)
        _player(a, medkits=1, reserve=32)
        a.add(Unit(id="opp", kind="gunner", team="red", x=48, y=30, medkits=1, reserve=32,
                   ai=shooter_ai("bot", 12.0, a.rng)))
    return Scenario("gunfight", "win the 1v1 against gunner opp", 120.0, setup,
                    lambda a: not a.units["opp"].alive and a.units["bot"].alive,
                    lambda a: not a.units["bot"].alive)


def supply_run() -> Scenario:
    def setup(a: Arena) -> None:
        _arena(a)
        _player(a, medkits=1, reserve=16)
        while len(a.crates) < 6:
            c = (a.rng.randint(4, 56), a.rng.randint(4, 56))
            if c not in a.cover and math.hypot(c[0] - 30, c[1] - 30) > 10:
                a.crates.add(c)
        a.waves = {"every_s": 15.0, "max": 2}

    def success(a: Arena) -> bool:
        b = a.units["bot"]
        return b.crates >= 4 and math.hypot(b.x - 30, b.y - 30) <= 3.0
    return Scenario("supply_run", "collect 4 supply crates and return to spawn", 180.0, setup, success,
                    lambda a: not a.units["bot"].alive)


SCENARIOS = {"hold_out": hold_out, "gunfight": gunfight, "supply_run": supply_run}
