"""Minecraft adapter (Architecture §8): Mineflayer bridge over a localhost socket.

Tier-3 reflexes (pathing, gathering, combat, eating) run in bridge/minecraft/bridge.js; this
class serialises state for Laya, maps vocabulary options to bridge reflexes, and enforces the
play rules before any connection is made.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import yaml

from ...protocol import Goal
from ...state import Entity, Facts
from .. import Context, GameAdapter, PlayRules
from .client import BridgeClient

CONFIG = Path(__file__).resolve().parents[4] / "configs/games/minecraft.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _q(v: float, step: float) -> str:
    return f"{round(v / step) * step:.1f}" if step < 1 else str(int(round(v / step) * step))


class Scenario:
    def __init__(self, name: str, mission: str, time_limit_s: float, target_logs: int = 64):
        self.name, self.mission, self.time_limit_s, self.target_logs = name, mission, time_limit_s, target_logs


SCENARIOS = {"gather_logs": lambda: Scenario("gather_logs", "gather 64 logs and return to base", 900.0)}


class MinecraftAdapter(GameAdapter):
    game = "minecraft"
    stepped = False

    def __init__(self, scenario: str = "gather_logs", bridge_port: int = 47100, token: str = "",
                 config: dict | None = None, seed: int = 0):
        self.cfg = config or load_config()
        self.profile = self.cfg["profile"]
        self.rules = PlayRules.from_config(self.cfg["play_rules"])
        self.tick_hz = self.cfg["tick_hz"]
        self.scenario = SCENARIOS[scenario]()
        self.bridge_port, self.token = bridge_port, token
        self.client: BridgeClient | None = None
        self.state: dict | None = None
        self.current_choice = ""
        self._next_tick = 0.0
        self._dead = False
        self._hp_prev = None

    # ------------------------------------------------------------------ connection
    def connect(self, context: Context, address: str | None = None) -> None:
        self.rules.check(context, address)          # refuse before touching the network
        self.client = BridgeClient(self.bridge_port, self.token)
        self.client.send({"cmd": "config", "autonomic": {"eat_below_food": self.cfg["autonomic"]["eat_below_hunger"]}})
        self.state = self.client.wait_state()
        self._next_tick = time.monotonic()

    def read_state(self) -> dict:
        if self.client is None:
            raise RuntimeError("not connected")
        self.client.poll()
        self.state = self.client.state
        return self.state

    def facts(self, s: dict) -> Facts:
        ents = [Entity(id=e["id"], kind=e.get("kind", "unknown"), hostile=bool(e["hostile"]), x=e["x"], y=e["z"],
                       vx=e.get("vx", 0.0), vy=e.get("vz", 0.0), hp=e.get("hp"))
                for e in s.get("entities", [])]
        return Facts(t=s["t"], x=s["x"], y=s["z"], vx=s.get("vx", 0.0), vy=s.get("vz", 0.0),
                     hp=float(s["health"]), hp_max=20.0, hunger=float(s["food"]), entities=ents,
                     inventory=dict(s.get("inv", {})), reflex=s["reflex"]["name"],
                     reflex_status=s["reflex"]["status"], alive=float(s["health"]) > 0)

    # ------------------------------------------------------------------ helpers
    def logs(self, s: dict) -> int:
        return sum(n for k, n in s.get("inv", {}).items() if k.endswith("_log"))

    def base_dist(self, s: dict) -> float:
        b = s["base"]
        return math.hypot(s["x"] - b["x"], s["z"] - b["z"])

    def _hostiles(self, s: dict) -> list[dict]:
        return sorted((e for e in s.get("entities", []) if e["hostile"]),
                      key=lambda e: math.hypot(e["x"] - s["x"], e["z"] - s["z"]))

    @staticmethod
    def _amount(goal: Goal | None) -> int:
        for c in (goal.constraints if goal else []):
            if c.startswith("amount "):
                return int(c.split()[1])
        return 1

    # ------------------------------------------------------------------ goals
    def goal_status(self, s: dict, goal: Goal):
        if float(s["health"]) <= 0:
            return "failed"
        verb, _, arg = goal.intent.partition(":")
        if verb == "gather":
            return "succeeded" if self.logs(s) >= self._amount(goal) else "active"
        if verb == "return":
            return "succeeded" if self.base_dist(s) <= 3.0 else "active"
        return "active"

    def check(self, condition: str, s: dict) -> bool:
        parts = condition.split()
        if parts[0] == "no_enemy_within":
            h = self._hostiles(s)
            return not h or math.hypot(h[0]["x"] - s["x"], h[0]["z"] - s["z"]) > float(parts[1])
        if parts[0] == "hp_above":
            return float(s["health"]) > float(parts[1])
        raise ValueError(f"unknown condition {condition!r}")

    def context(self, s: dict, goal: Goal | None) -> str:
        h = self._hostiles(s)
        if h and math.hypot(h[0]["x"] - s["x"], h[0]["z"] - s["z"]) <= self.cfg.get("combat_radius", 6.0):
            return "combat"
        if goal is not None and goal.intent.startswith("gather"):
            return "gathering"
        return "travel"

    def options(self, s: dict, goal: Goal | None) -> list[str]:
        return list(self.cfg["option_sets"][self.context(s, goal)])

    def safe_default(self, s: dict, goal: Goal | None) -> str:
        ctx = self.context(s, goal)
        if ctx == "combat":
            return "disengage"
        if goal is not None and goal.on_fail in self.cfg["option_sets"][ctx]:
            return goal.on_fail
        return "continue"

    # ------------------------------------------------------------------ serializers
    def describe(self, s: dict, goal: Goal | None, tier2: dict) -> str:
        g = goal
        goal_line = ("GOAL none" if g is None else
                     f"GOAL {g.intent}" + (f" target:{g.target}" if g.target else "")
                     + f" | CONSTRAINT {','.join(g.constraints) or 'none'}"
                     + f" | PRIORITY {'>'.join(g.priorities) or 'none'}")
        inv = s.get("inv", {})
        self_line = (f"SELF hp {int(math.ceil(s['health']))}/20 hunger {int(s['food'])} armor none "
                     f"totem offhand no spare {inv.get('totem_of_undying', 0)} "
                     f"gapples {inv.get('golden_apple', 0)} food {inv.get('bread', 0)} cooldown ready")
        h = self._hostiles(s)
        if h:
            o = h[0]
            d = math.hypot(o["x"] - s["x"], o["z"] - s["z"])
            opp_line = (f"OPP {o['kind']} dist {_q(d, 0.1) if d < 10 else _q(d, 1)} "
                        f"hp ~{int(o['hp']) if o.get('hp') else 20}/20 facing_me none crit_threat none")
        else:
            opp_line = "OPP none"
        near = sum(1 for e in h if math.hypot(e["x"] - s["x"], e["z"] - s["z"]) <= 8)
        res = _q(s["nearest_log"], 1) if s.get("nearest_log") is not None else "none"
        prog = f"{self.logs(s)}/{self._amount(g)}" if g and g.intent.startswith("gather") else "none"
        env_line = (f"ENV enemies {len(h)} near {near} allies 0 base_dist {_q(self.base_dist(s), 1)} "
                    f"resource_dist {res} progress {prog}")
        pred = tier2.get("pred") or {}
        pred_line = ("PRED " + " ".join(f"{k} {v:.1f}" for k, v in sorted(pred.items()))) if pred else "PRED none"
        drv = tier2.get("drives") or {}
        drv_line = ("DRIVES " + " ".join(f"{k} {v:.1f}" for k, v in sorted(drv.items()))) if drv else "DRIVES none"
        wm = tier2.get("wm") or []
        return "\n".join([f"GAME {self.game}", goal_line, self_line, opp_line, env_line, pred_line, drv_line,
                          f"WM last_actions {','.join(wm) if wm else 'none'}"])

    def summarise(self, s: dict, goal: Goal | None) -> str:
        inv = ", ".join(f"{k} x{v}" for k, v in sorted(s.get("inv", {}).items())) or "empty"
        h = self._hostiles(s)
        threats = ", ".join(f"{e['kind']} at {math.hypot(e['x'] - s['x'], e['z'] - s['z']):.0f}" for e in h[:4]) or "none"
        return (f"[t={s['t']:.1f}s] game: minecraft scenario {self.scenario.name}\n"
                f"status: health {s['health']:.0f}/20, hunger {s['food']:.0f}/20\n"
                f"inventory: {inv}\nposition: ({s['x']:.0f},{s['y']:.0f},{s['z']:.0f}); base {self.base_dist(s):.0f} away\n"
                f"threats: {threats}\nactive goal: {goal.intent if goal else 'none'}; reflex {s['reflex']['name']} "
                f"({s['reflex']['status']})")

    # ------------------------------------------------------------------ acting
    def _reflex_for(self, choice: str, goal: Goal | None) -> tuple[str, dict]:
        verb, _, arg = choice.partition(":")
        if verb == "continue" and goal is not None:
            gverb, _, garg = goal.intent.partition(":")
            if gverb == "gather":
                return "gather", {"block": "log"}
            if gverb == "return":
                return "goto", {"place": garg or "base"}
            if gverb == "engage":
                return "engage", {"target": goal.target}
            return "wait", {}
        return {"engage": ("engage", {}), "disengage": ("disengage", {}), "heal": ("heal", {}),
                "reposition": ("disengage", {}), "explore": ("explore", {}), "wait": ("wait", {}),
                "return": ("goto", {"place": arg or "base"}), "gather": ("gather", {"block": "log"}),
                }.get(verb, ("wait", {}))

    def execute(self, choice: str, goal: Goal | None, s: dict) -> None:
        name, params = self._reflex_for(choice, goal)
        key = f"{name}|{sorted(params.items())}|{goal.goal_id if goal else ''}"
        running = s["reflex"]["status"] == "running"
        if key == self.current_choice and running:
            return
        self.client.send({"cmd": "reflex", "name": name, "params": params})
        self.current_choice = key

    def tick(self) -> bool:
        self._next_tick += 1.0 / self.tick_hz
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = time.monotonic()
        s = self.read_state()
        if self.client.closed:
            return False
        if s["reflex"]["status"] in ("succeeded", "failed"):
            self.current_choice = ""
        return not self.episode_over()

    def episode_over(self) -> bool:
        s = self.state
        return (self.success() or float(s["health"]) <= 0 or s["t"] >= self.scenario.time_limit_s
                or self.client.closed)

    def success(self) -> bool:
        s = self.state
        return self.logs(s) >= self.scenario.target_logs and self.base_dist(s) <= 3.0

    def episode_result(self) -> dict:
        s = self.state
        return {"scenario": self.scenario.name, "success": self.success(), "t": round(s["t"], 2),
                "deaths": int(float(s["health"]) <= 0), "logs": self.logs(s),
                "base_dist": round(self.base_dist(s), 2), "hp": s["health"], "food": s["food"]}

    def reflexes(self) -> dict:
        return {"gather": "bridge", "goto": "bridge", "engage": "bridge", "disengage": "bridge",
                "heal": "bridge", "explore": "bridge", "wait": "bridge"}

    def reward(self, prev: dict, s: dict) -> dict[str, float]:
        rw = self.cfg["reward"]
        out = {}
        dh = float(s["health"]) - float(prev["health"])
        if dh < 0:
            out["damage_taken"] = rw["damage_taken"] * -dh
        if float(s["health"]) <= 0 and not self._dead:
            self._dead = True
            out["death"] = rw["death"]
        return out

    def release_all(self) -> None:
        self.current_choice = ""
        if self.client is not None and not self.client.closed:
            self.client.send({"cmd": "release_all"})

    def disconnect(self) -> None:
        if self.client is not None and not self.client.closed:
            self.client.send({"cmd": "quit"})
            self.client.wait_ack("quit", timeout=2.0)
            self.client.close()
