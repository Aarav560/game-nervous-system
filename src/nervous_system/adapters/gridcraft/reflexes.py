"""GridCraft tier-3 reflexes (Architecture §6): scripted controllers + autonomic rules.

Each reflex sets the bot's controls every tick; Laya only chooses which one runs.
"""
from __future__ import annotations

import math
from typing import Any

from ...reflex import Reflex, Status
from .world import Agent, World


class _Base(Reflex):
    name = "base"

    def __init__(self, world: World, bot: Agent):
        self.w, self.bot = world, bot
        self._status: Status = "running"
        self.params: dict = {}

    def start(self, params: dict[str, Any]) -> None:
        self.params = dict(params)
        self._status = "running"

    def status(self) -> Status:
        return self._status

    def cancel(self) -> None:
        self.bot.clear_controls()
        self.bot.mine_cell = None
        self._status = "failed" if self._status == "running" else self._status

    def _goto(self, x: float, y: float, sprint: bool = False, stop: float = 1.0) -> bool:
        b = self.bot
        d = math.hypot(x - b.x, y - b.y)
        if d <= stop:
            b.move = (0.0, 0.0)
            return True
        b.move, b.sprint = (x - b.x, y - b.y), sprint and b.hunger > 6
        return False

    def _hostiles(self, radius: float = 1e9) -> list[Agent]:
        b = self.bot
        return sorted((a for a in self.w.agents.values()
                       if a.alive and a.team != b.team and b.dist(a) <= radius), key=b.dist)


class Idle(_Base):
    name = "wait"

    def tick(self, state: Any = None) -> None:
        self.bot.clear_controls()


class Goto(_Base):
    name = "goto"

    def tick(self, state: Any = None) -> None:
        self.bot.clear_controls()
        x, y = self.params["x"], self.params["y"]
        if self._goto(x, y, sprint=self.params.get("sprint", False), stop=self.params.get("stop", 1.5)):
            self._status = "succeeded"


class Gather(_Base):
    """Walk to the nearest block of a kind and mine/chop it until the block is gone."""
    name = "gather"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.clear_controls()
        kind = self.params["block"]
        if b.mine_cell is None or b.mine_cell not in self.w.blocks:
            cell, _ = self.w.nearest_block(b.x, b.y, kind, radius=self.params.get("radius", 1e9))
            if cell is None:
                self._status = "failed"
                return
            b.mine_cell, b.mine_progress = cell, 0
        cx, cy = b.mine_cell
        self._goto(cx + 0.5, cy + 0.5, stop=1.5)


class Engage(_Base):
    """Scripted melee: sprint in for a crit when the cooldown is ready, back off while it recharges."""
    name = "engage"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.clear_controls()
        b.mine_cell = None
        tid = self.params.get("target")
        t = self.w.agents.get(tid) if tid else None
        if t is None or not t.alive:
            hs = self._hostiles(16)
            if not hs:
                self._status = "succeeded"
                return
            t = hs[0]
            self.params["target"] = t.id
        d = b.dist(t)
        dx, dy = (t.x - b.x) / max(d, 1e-6), (t.y - b.y) / max(d, 1e-6)
        b.attack = t.id
        if b.cd <= 2:
            b.move, b.sprint = (dx, dy), b.hunger > 6
        elif d < 2.6:
            b.move = (-dx + 0.5 * -dy, -dy + 0.5 * dx)
        else:
            side = 1 if (self.w.tick_n // 20) % 2 else -1
            b.move = (side * -dy, side * dx)


class Reposition(_Base):
    """Circle the nearest threat at ~3.5 blocks without committing (waiting out its cooldown)."""
    name = "reposition"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.clear_controls()
        hs = self._hostiles(16)
        if not hs:
            self._status = "succeeded"
            return
        t = hs[0]
        d = b.dist(t)
        dx, dy = (t.x - b.x) / max(d, 1e-6), (t.y - b.y) / max(d, 1e-6)
        radial = (d - 3.5) * 0.8
        b.move = (dx * radial - dy, dy * radial + dx)
        b.attack = t.id if b.cd == 0 else None


class Disengage(_Base):
    """Sprint directly away from the threat centroid (biased toward base if known)."""
    name = "disengage"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.clear_controls()
        b.mine_cell = None
        hs = self._hostiles(20)
        if not hs:
            self._status = "succeeded"
            return
        cx = sum(h.x for h in hs) / len(hs)
        cy = sum(h.y for h in hs) / len(hs)
        ax, ay = b.x - cx, b.y - cy
        n = math.hypot(ax, ay) or 1.0
        ax, ay = ax / n, ay / n
        # steer away from walls
        m = 6.0
        if b.x < m: ax += 1.0
        if b.x > self.w.size - m: ax -= 1.0
        if b.y < m: ay += 1.0
        if b.y > self.w.size - m: ay -= 1.0
        b.move, b.sprint = (ax, ay), b.hunger > 6


class Heal(_Base):
    """Eat a golden apple if hurt (else bread), backing away from threats while eating."""
    name = "heal"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.clear_controls()
        b.mine_cell = None
        if not b.eating:
            item = ("golden_apple" if b.count("golden_apple") and b.hp < b.hp_max * 0.75
                    else "bread" if b.count("bread") else
                    "golden_apple" if b.count("golden_apple") else None)
            if item is None or (item == "bread" and b.hunger >= 20):
                self._status = "failed"
                return
            if self.params.get("started"):
                self._status = "succeeded"
                return
            self.w.start_eating(b, item)
            self.params["started"] = True
        hs = self._hostiles(8)
        if hs:
            t = hs[0]
            b.move = (b.x - t.x, b.y - t.y)


class TotemSwap(_Base):
    name = "use_ability:totem_swap"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        if b.offhand != "totem" and b.count("totem") > 0:
            b.offhand = "totem"
            self._status = "succeeded"
        else:
            self._status = "failed"


class Explore(_Base):
    """Head for the least-visited region of the map."""
    name = "explore"

    def start(self, params: dict[str, Any]) -> None:
        super().start(params)
        b = self.bot
        best, bs = None, -math.inf
        for i in range(12):
            ang = 2 * math.pi * i / 12
            x = min(self.w.size - 4, max(4, b.x + 25 * math.cos(ang)))
            y = min(self.w.size - 4, max(4, b.y + 25 * math.sin(ang)))
            score = -self.params.get("visits", {}).get((int(x) // 16, int(y) // 16), 0) + self.w.rng.random()
            if score > bs:
                best, bs = (x, y), score
        self.params["dest"] = best

    def tick(self, state: Any = None) -> None:
        self.bot.clear_controls()
        x, y = self.params["dest"]
        if self._goto(x, y, stop=2.0):
            self._status = "succeeded"


class Craft(_Base):
    name = "build"

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.clear_controls()
        bx, by = self.w.places["base"]
        if self._goto(bx, by, stop=2.0):
            self._status = "succeeded" if self.w.craft(b, self.params["item"]) else "failed"


def autonomic(world: World, bot: Agent, cfg: dict) -> str | None:
    """Rules that never need thought (§6, §7.8): eat when hungry and safe, keep a totem in offhand."""
    acted = None
    if cfg.get("totem_offhand", True) and bot.offhand is None and bot.count("totem") > 0:
        bot.offhand = "totem"
        acted = "totem_offhand"
    hungry_below = cfg.get("eat_below_hunger", 14)
    if (bot.hunger < hungry_below and not bot.eating and bot.count("bread") > 0
            and not any(a.alive and a.team != bot.team and bot.dist(a) < 8 for a in world.agents.values())):
        world.start_eating(bot, "bread")
        acted = "eat"
    return acted
