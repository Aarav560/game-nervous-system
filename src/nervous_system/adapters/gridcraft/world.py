"""GridCraft: a small, fast, headless survival/PvP simulator (our own design).

Stands in for a sped-up private game server so benchmarks and training loops run thousands of
episodes on a CPU. Mechanics are deliberately Minecraft-like in shape (20 Hz ticks, melee
cooldowns, sprint-crits, knockback, hunger, golden apples, totems, ores needing better tools)
but it is not Minecraft and results here say "gridcraft", never "minecraft".

Top-down 2D, continuous positions, no collision except map bounds. Blocks live on an integer grid.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

DT = 0.05            # seconds per tick (20 Hz)

# block -> (ticks to break with the right tool, tool needed, item yielded)
BLOCKS = {
    "tree": (15, None, "log"),
    "stone": (20, "pickaxe", "cobblestone"),
    "iron_ore": (30, "stone_pickaxe", "iron_ore"),
    "diamond_ore": (40, "iron_pickaxe", "diamond"),
}
PICK_TIER = {"pickaxe": 0, "wooden_pickaxe": 0, "stone_pickaxe": 1, "iron_pickaxe": 2}
RECIPES = {"iron_pickaxe": {"iron_ore": 3, "log": 2}, "stone_pickaxe": {"cobblestone": 3, "log": 2}}
FOOD = {"bread": (6, 0.0), "golden_apple": (4, 4.0)}   # hunger restored, hp restored
EAT_TICKS = 32


@dataclass
class Agent:
    id: str
    kind: str                      # "player" or mob kind
    team: str
    x: float
    y: float
    hp: float = 20.0
    hp_max: float = 20.0
    damage: float = 6.0
    armor: float = 0.0             # fraction of damage absorbed
    reach: float = 3.0
    cooldown: int = 12             # ticks between full-strength attacks
    speed: float = 4.3             # blocks/s walking
    sprint_mult: float = 1.3
    hunger: float = 20.0
    inventory: dict = field(default_factory=dict)
    offhand: str | None = None
    ai: Callable | None = None     # mobs / opponent: ai(world, agent) sets controls
    # dynamic
    vx: float = 0.0
    vy: float = 0.0
    fx: float = 1.0                # facing unit vector
    fy: float = 0.0
    cd: int = 0
    kb_x: float = 0.0
    kb_y: float = 0.0
    eating: str | None = None
    eat_t: int = 0
    regen_t: int = 0               # ticks of golden-apple / totem regen left
    mine_cell: tuple | None = None
    mine_progress: int = 0
    alive: bool = True
    exhaustion: float = 0.0
    regen_clock: float = 0.0
    starve_clock: float = 0.0
    # controls (set each tick by reflexes / ai)
    move: tuple = (0.0, 0.0)
    sprint: bool = False
    attack: str | None = None
    # stats
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    hits: int = 0
    crits: int = 0
    totems_popped: int = 0

    def dist(self, o: "Agent") -> float:
        return math.hypot(self.x - o.x, self.y - o.y)

    def clear_controls(self) -> None:
        self.move, self.sprint, self.attack = (0.0, 0.0), False, None

    def count(self, item: str) -> int:
        return self.inventory.get(item, 0)

    def best_pick(self) -> int:
        return max((PICK_TIER[k] for k in self.inventory if k in PICK_TIER and self.inventory[k] > 0),
                   default=-1)


class World:
    def __init__(self, size: float = 96.0, seed: int = 0):
        self.size = size
        self.rng = random.Random(seed)
        self.tick_n = 0
        self.agents: dict[str, Agent] = {}
        self.blocks: dict[tuple[int, int], list] = {}       # cell -> [kind, remaining units]
        self.places: dict[str, tuple[float, float]] = {}
        self.spawner: dict | None = None
        self.events: list[tuple] = []                       # (t, kind, data) since last drain
        self._next_id = 0

    @property
    def t(self) -> float:
        return self.tick_n * DT

    # ------------------------------------------------------------------ setup
    def add(self, agent: Agent) -> Agent:
        self.agents[agent.id] = agent
        return agent

    def new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}{self._next_id}"

    def place_block(self, x: int, y: int, kind: str, units: int = 1) -> None:
        self.blocks[(x, y)] = [kind, units]

    def nearest_block(self, x: float, y: float, kind: str, radius: float = 1e9):
        best, bd = None, radius
        for (cx, cy), (k, _) in self.blocks.items():
            if k == kind:
                d = math.hypot(cx + 0.5 - x, cy + 0.5 - y)
                if d < bd:
                    best, bd = (cx, cy), d
        return best, bd

    # ------------------------------------------------------------------ combat
    def hurt(self, target: Agent, amount: float, src: Agent | None) -> None:
        if not target.alive:
            return
        dmg = amount * (1 - target.armor)
        target.damage_taken += dmg
        if src:
            src.damage_dealt += dmg
        target.hp -= dmg
        if target.eating:                 # damage interrupts eating
            target.eating, target.eat_t = None, 0
        if target.hp <= 0:
            if target.offhand == "totem":
                target.offhand = None
                target.inventory["totem"] = max(0, target.count("totem") - 1)
                target.hp, target.regen_t = 1.0, 80
                target.totems_popped += 1
                self.events.append((self.t, "totem_pop", target.id))
            else:
                target.hp, target.alive = 0.0, False
                self.events.append((self.t, "death", target.id, src.id if src else None))

    def _attack(self, a: Agent) -> None:
        tgt = self.agents.get(a.attack) if a.attack else None
        if tgt is None or not tgt.alive or a.cd > 0 or a.eating:
            return
        d = a.dist(tgt)
        if d > a.reach:
            return
        dx, dy = (tgt.x - a.x) / max(d, 1e-6), (tgt.y - a.y) / max(d, 1e-6)
        approach = a.vx * dx + a.vy * dy
        crit = a.sprint and approach > a.speed * 0.9
        a.hits += 1
        a.crits += int(crit)
        self.hurt(tgt, a.damage * (1.5 if crit else 1.0), a)
        tgt.kb_x += dx * (0.9 if crit else 0.6)
        tgt.kb_y += dy * (0.9 if crit else 0.6)
        a.cd = a.cooldown
        a.fx, a.fy = dx, dy

    # ------------------------------------------------------------------ tick
    def step(self) -> None:
        for a in list(self.agents.values()):
            if a.alive and a.ai:
                a.ai(self, a)
        for a in list(self.agents.values()):
            if a.alive:
                self._physics(a)
        for a in list(self.agents.values()):
            if a.alive:
                self._attack(a)
                self._mine(a)
                self._eat(a)
                self._metabolism(a)
        self._spawn()
        self.agents = {k: v for k, v in self.agents.items() if v.alive or v.kind == "player"}
        self.tick_n += 1

    def _physics(self, a: Agent) -> None:
        mx, my = a.move
        n = math.hypot(mx, my)
        sp = a.speed * (a.sprint_mult if a.sprint and a.hunger > 6 else 1.0) * (0.35 if a.eating else 1.0)
        if n > 1e-6:
            a.vx, a.vy = mx / n * sp, my / n * sp
            a.fx, a.fy = mx / n, my / n
        else:
            a.vx = a.vy = 0.0
        a.x += a.vx * DT + a.kb_x
        a.y += a.vy * DT + a.kb_y
        a.kb_x *= 0.0
        a.kb_y *= 0.0
        a.x = min(self.size - 0.5, max(0.5, a.x))
        a.y = min(self.size - 0.5, max(0.5, a.y))
        if a.cd > 0:
            a.cd -= 1
        if a.kind == "player":
            a.exhaustion += DT * (1 / 30 + (1 / 8 if a.sprint and n > 1e-6 else 0.0))

    def _mine(self, a: Agent) -> None:
        if a.mine_cell is None:
            a.mine_progress = 0
            return
        blk = self.blocks.get(a.mine_cell)
        cx, cy = a.mine_cell
        if blk is None or math.hypot(cx + 0.5 - a.x, cy + 0.5 - a.y) > 2.0:
            a.mine_progress = 0
            return
        kind = blk[0]
        ticks, tool, item = BLOCKS[kind]
        if tool is not None and a.best_pick() < PICK_TIER[tool]:
            return
        a.mine_progress += 1
        if a.mine_progress >= ticks:
            a.mine_progress = 0
            a.inventory[item] = a.count(item) + 1
            blk[1] -= 1
            if blk[1] <= 0:
                del self.blocks[a.mine_cell]
                a.mine_cell = None
            self.events.append((self.t, "mined", a.id, item))

    def _eat(self, a: Agent) -> None:
        if not a.eating:
            return
        a.eat_t += 1
        if a.eat_t >= EAT_TICKS:
            item = a.eating
            if a.count(item) > 0:
                a.inventory[item] -= 1
                hunger, hp = FOOD[item]
                a.hunger = min(20.0, a.hunger + hunger)
                a.hp = min(a.hp_max, a.hp + hp)
                if item == "golden_apple":
                    a.regen_t = max(a.regen_t, 100)
            a.eating, a.eat_t = None, 0

    def start_eating(self, a: Agent, item: str) -> bool:
        if a.eating or a.count(item) <= 0:
            return False
        a.eating, a.eat_t = item, 0
        return True

    def craft(self, a: Agent, item: str) -> bool:
        need = RECIPES.get(item)
        base = self.places.get("base")
        if need is None or base is None or math.hypot(a.x - base[0], a.y - base[1]) > 3.0:
            return False
        if any(a.count(k) < n for k, n in need.items()):
            return False
        for k, n in need.items():
            a.inventory[k] -= n
        a.inventory[item] = a.count(item) + 1
        self.events.append((self.t, "crafted", a.id, item))
        return True

    def _metabolism(self, a: Agent) -> None:
        if a.regen_t > 0:
            a.regen_t -= 1
            a.hp = min(a.hp_max, a.hp + 1.0 * DT * 2)        # 2 hp/s while regen lasts
        if a.kind != "player":
            return
        while a.exhaustion >= 1.0:
            a.exhaustion -= 1.0
            a.hunger = max(0.0, a.hunger - 1)
        if a.hunger >= 18 and a.hp < a.hp_max:
            a.regen_clock += DT
            if a.regen_clock >= 2.0:
                a.regen_clock = 0.0
                a.hp = min(a.hp_max, a.hp + 1)
        if a.hunger <= 0:
            a.starve_clock += DT
            if a.starve_clock >= 4.0:
                a.starve_clock = 0.0
                self.hurt(a, 1.0 / (1 - a.armor) if a.armor < 1 else 1.0, None)

    def _spawn(self) -> None:
        s = self.spawner
        if not s:
            return
        s["clock"] = s.get("clock", 0.0) + DT
        if s["clock"] < s["every_s"]:
            return
        s["clock"] = 0.0
        mobs = [a for a in self.agents.values() if a.kind == s["kind"] and a.alive]
        if len(mobs) >= s["max"]:
            return
        anchor = self.agents.get(s["near"])
        if anchor is None or not anchor.alive:
            return
        ang = self.rng.uniform(0, 2 * math.pi)
        r = self.rng.uniform(s.get("rmin", 14), s.get("rmax", 22))
        x = min(self.size - 1, max(1, anchor.x + r * math.cos(ang)))
        y = min(self.size - 1, max(1, anchor.y + r * math.sin(ang)))
        self.add(make_zombie(self.new_id("z"), x, y, s.get("aggro", 16.0)))

    def drain_events(self) -> list[tuple]:
        ev, self.events = self.events, []
        return ev


# ---------------------------------------------------------------------- mobs & bots
def zombie_ai(aggro: float):
    def ai(world: World, z: Agent) -> None:
        z.clear_controls()
        players = [p for p in world.agents.values() if p.kind == "player" and p.alive and p.team != z.team]
        if not players:
            return
        p = min(players, key=z.dist)
        d = z.dist(p)
        if d <= aggro:
            if d > z.reach * 0.8:
                z.move = (p.x - z.x, p.y - z.y)
            z.attack = p.id
    return ai


def make_zombie(id_: str, x: float, y: float, aggro: float = 16.0) -> Agent:
    return Agent(id=id_, kind="zombie", team="mobs", x=x, y=y, hp=20, hp_max=20, damage=4.0,
                 armor=0.0, reach=2.2, cooldown=20, speed=3.6, ai=zombie_ai(aggro))


def scripted_pvp_ai(target_id: str, heal_below: float = 8.0, rng: random.Random | None = None):
    """The fixed practice opponent: sprint in, crit on cooldown, back off (w-tap), eat when low."""
    rng = rng or random.Random(0)

    def ai(world: World, a: Agent) -> None:
        a.clear_controls()
        t = world.agents.get(target_id)
        if t is None or not t.alive:
            return
        d = a.dist(t)
        dx, dy = (t.x - a.x) / max(d, 1e-6), (t.y - a.y) / max(d, 1e-6)
        if a.offhand is None and a.count("totem") > 0:
            a.offhand = "totem"
        if a.hp <= heal_below and a.count("golden_apple") > 0 and not a.eating and d > 4.0:
            world.start_eating(a, "golden_apple")
        if a.eating:
            a.move = (-dx, -dy)
            return
        if a.hp <= heal_below and a.count("golden_apple") > 0:
            a.move, a.sprint = (-dx + 0.3 * -dy, -dy + 0.3 * dx), True        # retreat to eat
            return
        a.attack = t.id
        if a.cd <= 2:
            a.move, a.sprint = (dx, dy), True
        elif d < 2.6:
            a.move = (-dx + 0.5 * -dy, -dy + 0.5 * dx)                         # back off / strafe
        else:
            side = 1 if (world.tick_n // 20) % 2 else -1
            a.move = (side * -dy, side * dx)
        if rng.random() < 0.05:
            a.move = (a.move[0] + rng.uniform(-0.5, 0.5), a.move[1] + rng.uniform(-0.5, 0.5))
    return ai
