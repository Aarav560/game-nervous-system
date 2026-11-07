"""GridCraft adapter: serializer, option sets, reflex mapping, rewards (Architecture §8)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from ...protocol import Goal
from ...state import Entity, Facts
from .. import Context, GameAdapter, PlayRules
from . import reflexes as R
from .scenarios import BOT, SCENARIOS, Scenario
from .world import World

CONFIG = Path(__file__).resolve().parents[4] / "configs/games/gridcraft.yaml"
RESOURCE_BLOCK = {"wood": "tree", "log": "tree", "logs": "tree", "iron_ore": "iron_ore", "cobblestone": "stone",
                  "iron": "iron_ore", "diamond": "diamond_ore", "diamonds": "diamond_ore",
                  "stone": "stone"}
RESOURCE_ITEM = {"tree": "log", "iron_ore": "iron_ore", "diamond_ore": "diamond", "stone": "cobblestone"}


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _q(v: float, step: float) -> str:
    return f"{round(v / step) * step:.1f}" if step < 1 else str(int(round(v / step) * step))


class GridCraftAdapter(GameAdapter):
    game = "gridcraft"
    stepped = True

    def __init__(self, scenario: Scenario | str, seed: int = 0, config: dict | None = None,
                 world: World | None = None, bot_id: str = BOT, owns_world: bool = True,
                 combat_policy: str | None = None):
        self.cfg = config or load_config()
        self.profile = self.cfg["profile"]
        self.rules = PlayRules.from_config(self.cfg["play_rules"])
        self.tick_hz = self.cfg["tick_hz"]
        self.scenario = SCENARIOS[scenario]() if isinstance(scenario, str) else scenario
        self.seed = seed
        self.bot_id = bot_id
        self.w = world
        self.connected = False
        self.current: R._Base | None = None
        self.current_choice = ""
        self.autonomic_log: list[str] = []
        self.visits: dict = {}
        self.engaged_target: str | None = None
        self._last_hp = 20.0
        self.owns_world = owns_world               # hive: the HiveRuntime steps a shared world
        self.markers: list[tuple[float, float]] = []   # hive danger markers to avoid
        self.deposited = 0
        self.combat_policy = None
        path = combat_policy or (self.cfg.get("reflexes") or {}).get("combat_policy")
        if path:
            from .combat import load_policy
            p = Path(path) if Path(path).is_absolute() else CONFIG.parents[2] / path
            if p.exists():
                self.combat_policy = load_policy(p)

    # ------------------------------------------------------------------ connection
    def connect(self, context: Context, address: str | None = None) -> None:
        self.rules.check(context, address)
        if self.w is None:
            self.w = World(seed=self.seed)
            self.scenario.setup(self.w)
        self.connected = True

    @property
    def bot(self):
        return self.w.agents[self.bot_id]

    # ------------------------------------------------------------------ state
    def read_state(self) -> World:
        if not self.connected:
            raise RuntimeError("adapter not connected (play rules are checked in connect())")
        return self.w

    def facts(self, w: World) -> Facts:
        b = self.bot
        ents = []
        for a in w.agents.values():
            if a.id == b.id or not a.alive:
                continue
            d = b.dist(a)
            if d > 24:
                continue
            fx, fy = (b.x - a.x) / max(d, 1e-6), (b.y - a.y) / max(d, 1e-6)
            ents.append(Entity(id=a.id, kind=a.kind, hostile=a.team != b.team, x=a.x, y=a.y,
                               vx=a.vx, vy=a.vy, hp=a.hp, hp_max=a.hp_max,
                               facing_self=(a.fx * fx + a.fy * fy) > 0.7, attack_ready=a.cd == 0))
        cell = (int(b.x) // 16, int(b.y) // 16)
        explored = len(self.visits) / max(1, (int(w.size) // 16) ** 2)
        return Facts(t=w.t, x=b.x, y=b.y, vx=b.vx, vy=b.vy, hp=b.hp, hp_max=b.hp_max,
                     hunger=b.hunger, entities=ents, inventory=dict(b.inventory),
                     allies=sum(1 for a in w.agents.values() if a.alive and a.team == b.team and a.id != b.id),
                     reflex=self.current_choice or "none",
                     reflex_status=self.current.status() if self.current else "none",
                     explored_frac=explored, idle_s=0.0, alive=b.alive,
                     extras={"cell": cell})

    # ------------------------------------------------------------------ goals
    def _goal_parts(self, goal: Goal | None) -> tuple[str, str | None]:
        if goal is None:
            return "none", None
        verb, _, arg = goal.intent.partition(":")
        return verb, arg or goal.target

    def _goal_progress(self, goal: Goal | None) -> str:
        verb, arg = self._goal_parts(goal)
        if verb == "gather" and arg:
            item = RESOURCE_ITEM.get(RESOURCE_BLOCK.get(arg, arg), arg)
            need = self._goal_amount(goal)
            return f"{self.bot.count(item)}/{need}"
        return "none"

    @staticmethod
    def _goal_amount(goal: Goal | None) -> int:
        for c in (goal.constraints if goal else []):
            if c.startswith("amount "):
                return int(c.split()[1])
        return 1

    def goal_status(self, w: World, goal: Goal):
        b = self.bot
        if not b.alive:
            return "failed"
        verb, arg = self._goal_parts(goal)
        if verb == "gather" and arg:
            item = RESOURCE_ITEM.get(RESOURCE_BLOCK.get(arg, arg), arg)
            if b.count(item) >= self._goal_amount(goal):
                return "succeeded"
            kind = RESOURCE_BLOCK.get(arg, arg)
            if not any(k == kind for k, _ in w.blocks.values()):
                return "failed"                       # resource exhausted everywhere
            return "active"
        if verb == "return" and arg:
            px, py = w.places.get(arg, (b.x, b.y))
            return "succeeded" if math.hypot(b.x - px, b.y - py) <= 3.0 else "active"
        if verb == "build" and arg:
            st = w.structures.get(arg)
            if st is not None:
                carrying = any(b.count(k) > 0 and st["have"].get(k, 0) < n for k, n in st["need"].items())
                return "succeeded" if st["built"] or not carrying else "active"
            return "succeeded" if b.count(arg) > 0 else "active"
        if verb == "engage":
            tgt = w.agents.get(goal.target or "")
            if goal.target and (tgt is None or not tgt.alive):
                return "succeeded"
            return "active"
        return "active"

    def check(self, condition: str, w: World) -> bool:
        parts = condition.split()
        if parts[0] == "no_enemy_within":
            r = float(parts[1])
            return not any(a.alive and a.team != self.bot.team and self.bot.dist(a) <= r
                           for a in w.agents.values())
        if parts[0] == "hp_above":
            return self.bot.hp > float(parts[1])
        if parts[0] == "has":
            return self.bot.count(parts[1]) >= int(parts[2] if len(parts) > 2 else 1)
        raise ValueError(f"unknown condition {condition!r}")

    # ------------------------------------------------------------------ options
    def _threat(self, w: World):
        b = self.bot
        hs = sorted((a for a in w.agents.values() if a.alive and a.team != b.team), key=b.dist)
        return hs

    def context(self, w: World, goal: Goal | None) -> str:
        hs = self._threat(w)
        verb, _ = self._goal_parts(goal)
        if hs and (self.bot.dist(hs[0]) <= self.cfg["combat_radius"] or verb == "engage"):
            return "combat"
        if verb == "gather":
            return "gathering"
        return "travel"

    def options(self, w: World, goal: Goal | None) -> list[str]:
        return list(self.cfg["option_sets"][self.context(w, goal)])

    def safe_default(self, w: World, goal: Goal | None) -> str:
        ctx = self.context(w, goal)
        if ctx == "combat":
            return "disengage"
        if goal is not None and goal.on_fail in self.cfg["option_sets"][ctx]:
            return goal.on_fail
        return "continue"

    # ------------------------------------------------------------------ serializers
    def describe(self, w: World, goal: Goal | None, tier2: dict) -> str:
        b = self.bot
        g = goal
        goal_line = ("GOAL none" if g is None else
                     f"GOAL {g.intent}" + (f" target:{g.target}" if g.target else "")
                     + f" | CONSTRAINT {','.join(g.constraints) or 'none'}"
                     + f" | PRIORITY {'>'.join(g.priorities) or 'none'}")
        cooldown = "ready" if b.cd == 0 else ("eating" if b.eating else "charging")
        self_line = (f"SELF hp {int(math.ceil(b.hp))}/20 hunger {int(b.hunger)} armor {b.armor:.1f} "
                     f"totem offhand {'yes' if b.offhand == 'totem' else 'no'} spare {b.count('totem')} "
                     f"gapples {b.count('golden_apple')} food {b.count('bread')} cooldown {cooldown}")
        hs = self._threat(w)
        if hs:
            o = hs[0]
            d = b.dist(o)
            fx, fy = (b.x - o.x) / max(d, 1e-6), (b.y - o.y) / max(d, 1e-6)
            facing = (o.fx * fx + o.fy * fy) > 0.7
            crit = "high" if (o.cd == 0 and d < 4) else "low"
            opp_line = (f"OPP {o.kind} dist {_q(d, 0.1) if d < 10 else _q(d, 1)} hp ~{int(math.ceil(o.hp))}/20 "
                        f"facing_me {'yes' if facing else 'no'} crit_threat {crit}")
        else:
            opp_line = "OPP none"
        near = sum(1 for a in hs if b.dist(a) <= 8)
        base = w.places.get("base")
        base_d = _q(math.hypot(b.x - base[0], b.y - base[1]), 1) if base else "none"
        verb, arg = self._goal_parts(g)
        res = "none"
        if verb == "gather" and arg:
            _, rd = w.nearest_block(b.x, b.y, RESOURCE_BLOCK.get(arg, arg), radius=40)
            res = _q(rd, 1) if rd < 40 else "none"
        env_line = (f"ENV enemies {len(hs)} near {near} allies {self.facts(w).allies} "
                    f"base_dist {base_d} resource_dist {res} progress {self._goal_progress(g)}")
        pred = tier2.get("pred") or {}
        pred_line = ("PRED " + " ".join(f"{k} {v:.1f}" for k, v in sorted(pred.items()))) if pred else "PRED none"
        drv = tier2.get("drives") or {}
        drv_line = ("DRIVES " + " ".join(f"{k} {v:.1f}" for k, v in sorted(drv.items()))) if drv else "DRIVES none"
        wm = tier2.get("wm") or []
        wm_line = f"WM last_actions {','.join(wm) if wm else 'none'}"
        return "\n".join([f"GAME {self.game}", goal_line, self_line, opp_line, env_line, pred_line,
                          drv_line, wm_line])

    def summarise(self, w: World, goal: Goal | None) -> str:
        b = self.bot
        hs = self._threat(w)
        inv = ", ".join(f"{k} x{v}" for k, v in sorted(b.inventory.items()) if v) or "empty"
        threats = ", ".join(f"{a.kind} at {b.dist(a):.0f}" for a in hs[:4]) or "none"
        base = w.places.get("base")
        return (f"[t={w.t:.1f}s] [bot {self.bot_id}] game: gridcraft scenario {self.scenario.name}\n"
                f"status: health {b.hp:.0f}/20, hunger {b.hunger:.0f}/20, offhand {b.offhand or 'empty'}\n"
                f"inventory: {inv}\n"
                f"position: ({b.x:.0f},{b.y:.0f}); base {('at (%.0f,%.0f)' % base) if base else 'none'}\n"
                f"threats: {threats}\n"
                f"active goal: {goal.intent if goal else 'none'}; reflex {self.current_choice.split('|')[0] or 'none'}"
                + "".join(f"\nblueprint {n}: " + ("built" if st["built"] else "need " + " ".join(
                    f"{k} {v - st['have'].get(k, 0)}" for k, v in st["need"].items()))
                    for n, st in w.structures.items()))

    # ------------------------------------------------------------------ acting
    def _make_reflex(self, choice: str, goal: Goal | None) -> R._Base:
        w, b = self.w, self.bot
        verb, arg = choice.partition(":")[0], choice.partition(":")[2] or None
        if verb == "continue":
                gverb, garg = self._goal_parts(goal)
                if gverb == "gather" and garg:
                    r = R.Gather(w, b); r.start({"block": RESOURCE_BLOCK.get(garg, garg), "avoid": self.markers}); return r
                if gverb == "return" and garg and garg in w.places:
                    x, y = w.places[garg]; r = R.Goto(w, b); r.start({"x": x, "y": y, "stop": 2.0}); return r
                if gverb == "build" and garg:
                    return self._build(garg)
                if gverb == "engage":
                    return self._engage(goal.target)
                r = R.Idle(w, b); r.start({}); return r
        if verb == "engage":
            return self._engage(goal.target if goal and goal.intent == "engage" else None)
        if verb == "disengage":
            r = R.Disengage(w, b); r.start({}); return r
        if verb == "heal":
            r = R.Heal(w, b); r.start({}); return r
        if verb == "reposition":
            r = R.Reposition(w, b); r.start({}); return r
        if verb == "use_ability" and arg == "totem_swap":
            r = R.TotemSwap(w, b); r.start({}); return r
        if verb == "explore":
            r = R.Explore(w, b); r.start({"visits": self.visits}); return r
        if verb == "return" and arg in w.places:
            x, y = w.places[arg]; r = R.Goto(w, b); r.start({"x": x, "y": y, "stop": 2.0}); return r
        if verb == "gather" and arg:
            r = R.Gather(w, b); r.start({"block": RESOURCE_BLOCK.get(arg, arg)}); return r
        if verb == "build" and arg:
            return self._build(arg)
        r = R.Idle(w, b); r.start({}); return r

    def _build(self, arg: str) -> R._Base:
        if arg in self.w.structures:
            r = R.Deposit(self.w, self.bot); r.start({"structure": arg}); return r
        r = R.Craft(self.w, self.bot); r.start({"item": arg}); return r

    def _engage(self, target: str | None) -> R._Base:
        if self.combat_policy is not None:
            from .combat import LearnedEngage
            r = LearnedEngage(self.w, self.bot, self.combat_policy)
        else:
            r = R.Engage(self.w, self.bot)
        r.start({"target": target})
        return r

    def execute(self, choice: str, goal: Goal | None, w: World) -> None:
        key = choice + "|" + (goal.goal_id if goal else "")
        if self.current is not None and key == self.current_choice and self.current.status() == "running":
            return
        if self.current is not None:
            self.current.cancel()
        self.current = self._make_reflex(choice, goal)
        self.current_choice = key

    def tick(self) -> bool:
        w = self.w
        b = self.bot
        if self.current is not None and b.alive:
            self.current.tick()
            if self.current.status() != "running":
                self.current_choice = ""     # finished reflex: next decision starts a fresh one
        if b.alive:
            a = R.autonomic(w, b, self.cfg["autonomic"])
            if a:
                self.autonomic_log.append(a)
        cell = (int(b.x) // 16, int(b.y) // 16)
        self.visits[cell] = self.visits.get(cell, 0) + 1
        if self.owns_world:
            w.step()
        return not self.episode_over()

    def episode_over(self) -> bool:
        s, w = self.scenario, self.w
        return s.success(w) or s.failed(w) or w.t >= s.time_limit_s

    def episode_result(self) -> dict:
        s, w, b = self.scenario, self.w, self.bot
        res = {"scenario": s.name, "success": bool(s.success(w)), "t": round(w.t, 2),
               "deaths": int(not b.alive), "hp": round(b.hp, 1),
               "damage_dealt": round(b.damage_dealt, 1), "damage_taken": round(b.damage_taken, 1),
               "totems_popped": b.totems_popped}
        if "opp" in w.agents:
            res["opp_hp"] = round(w.agents["opp"].hp, 1)
        return res

    def reflexes(self) -> dict:
        return {"engage": R.Engage, "disengage": R.Disengage, "heal": R.Heal, "reposition": R.Reposition,
                "use_ability:totem_swap": R.TotemSwap, "explore": R.Explore, "gather": R.Gather,
                "return": R.Goto, "build": R.Craft, "wait": R.Idle}

    def reward(self, prev: World, w: World) -> dict[str, float]:
        b = self.bot
        rw = self.cfg["reward"]
        out = {}
        dmg_taken = b.damage_taken - getattr(self, "_dt", 0.0)
        dmg_dealt = b.damage_dealt - getattr(self, "_dd", 0.0)
        self._dt, self._dd = b.damage_taken, b.damage_dealt
        if dmg_taken:
            out["damage_taken"] = rw["damage_taken"] * dmg_taken
        if dmg_dealt:
            out["damage_dealt"] = rw["damage_dealt"] * dmg_dealt
        if not b.alive and not getattr(self, "_dead", False):
            self._dead = True
            out["death"] = rw["death"]
        return out

    def release_all(self) -> None:
        if self.current is not None:
            self.current.cancel()
        self.current, self.current_choice = None, ""
        if self.w is not None and self.bot_id in self.w.agents:
            self.bot.clear_controls()
            self.bot.mine_cell = None

    def disconnect(self) -> None:
        self.connected = False
