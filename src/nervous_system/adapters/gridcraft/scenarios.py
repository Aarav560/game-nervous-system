"""GridCraft benchmark scenarios (Architecture §12.5; Roadmap Phase 2 'benchmark harness v1').

Mirrors the four Minecraft examples: gather 64 logs, survive 5 min in a mob-dense area,
win a 1v1 vs a fixed bot, mine to diamonds.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .world import Agent, World, scripted_pvp_ai

BOT = "bot"


@dataclass
class Scenario:
    name: str
    mission: str
    time_limit_s: float
    setup: Callable[[World], None]
    success: Callable[[World], bool]
    failed: Callable[[World], bool]


def _player(world: World, x: float, y: float, kit: dict, id_: str = BOT, team: str = "bot") -> Agent:
    a = Agent(id=id_, kind="player", team=team, x=x, y=y, damage=kit.get("damage", 6.0),
              armor=kit.get("armor", 0.0), inventory=dict(kit.get("items", {})),
              offhand=kit.get("offhand"))
    return world.add(a)


def _forest(world: World, cx: float, cy: float, n: int, rmin: float, rmax: float) -> None:
    rng = world.rng
    placed = 0
    while placed < n:
        ang, r = rng.uniform(0, 2 * math.pi), rng.uniform(rmin, rmax)
        x, y = int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))
        if 1 < x < world.size - 2 and 1 < y < world.size - 2 and (x, y) not in world.blocks:
            world.place_block(x, y, "tree", rng.randint(4, 6))
            placed += 1


def _bot(world: World) -> Agent:
    return world.agents[BOT]


# ---------------------------------------------------------------------------- scenarios
def gather_logs() -> Scenario:
    def setup(w: World) -> None:
        w.places["base"] = (48.0, 48.0)
        _player(w, 48, 48, {"damage": 5.0, "armor": 0.2, "items": {"bread": 6, "golden_apple": 1}})
        _forest(w, 48, 48, 22, 12, 34)
        w.spawner = {"kind": "zombie", "every_s": 20.0, "max": 2, "near": BOT, "rmin": 16, "rmax": 24}

    def success(w: World) -> bool:
        b = _bot(w)
        bx, by = w.places["base"]
        return b.count("log") >= 64 and math.hypot(b.x - bx, b.y - by) <= 3.0

    return Scenario("gather_logs", "gather 64 logs and return to base", 300.0, setup, success,
                    lambda w: not _bot(w).alive)


def survive() -> Scenario:
    def setup(w: World) -> None:
        w.places["base"] = (48.0, 48.0)
        _player(w, 48, 48, {"damage": 6.0, "armor": 0.4, "offhand": "totem",
                            "items": {"bread": 8, "golden_apple": 3, "totem": 1}})
        w.spawner = {"kind": "zombie", "every_s": 4.0, "max": 6, "near": BOT, "rmin": 12, "rmax": 20}

    return Scenario("survive", "survive for 5 minutes", 300.0, setup,
                    lambda w: _bot(w).alive and w.t >= 300.0 - 1e-6, lambda w: not _bot(w).alive)


def duel(opponent_ai=None) -> Scenario:
    kit = {"damage": 7.0, "armor": 0.5, "offhand": "totem", "items": {"golden_apple": 2, "totem": 1}}

    def setup(w: World) -> None:
        w.places["base"] = (48.0, 48.0)
        _player(w, 40, 48, kit)
        opp = _player(w, 56, 48, kit, id_="opp", team="red")
        opp.ai = (opponent_ai or scripted_pvp_ai)(BOT, rng=random.Random(w.rng.random()))

    def success(w: World) -> bool:
        return not w.agents["opp"].alive and _bot(w).alive

    return Scenario("duel", "win the 1v1 against player opp", 180.0, setup, success,
                    lambda w: not _bot(w).alive)


def diamonds() -> Scenario:
    def setup(w: World) -> None:
        w.places["base"] = (20.0, 48.0)
        _player(w, 20, 48, {"damage": 5.0, "armor": 0.2,
                            "items": {"stone_pickaxe": 1, "bread": 6, "golden_apple": 1, "log": 4}})
        rng = w.rng
        for _ in range(6):
            w.place_block(int(rng.uniform(32, 44)), int(rng.uniform(36, 60)), "iron_ore", 1)
        for _ in range(3):
            w.place_block(int(rng.uniform(70, 86)), int(rng.uniform(30, 66)), "diamond_ore", 1)
        w.spawner = {"kind": "zombie", "every_s": 12.0, "max": 3, "near": BOT, "rmin": 14, "rmax": 22}

    return Scenario("diamonds", "mine a diamond", 400.0, setup,
                    lambda w: _bot(w).count("diamond") >= 1, lambda w: not _bot(w).alive)


SCENARIOS = {"gather_logs": gather_logs, "survive": survive, "duel": duel, "diamonds": diamonds}
