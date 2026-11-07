"""Skirmish adapter (Phase 7). Written against the unchanged GameAdapter interface."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from ...protocol import Goal
from ...reflex import Reflex, Status
from ...state import Entity, Facts
from .. import Context, GameAdapter, PlayRules
from .game import SCENARIOS, Arena, Scenario, Unit

CONFIG = Path(__file__).resolve().parents[4] / "configs/games/skirmish.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _q(v: float, step: float) -> str:
    return f"{round(v / step) * step:.1f}" if step < 1 else str(int(round(v / step) * step))


class _R(Reflex):
    name = "base"

    def __init__(self, a: Arena, me: Unit, params: dict | None = None):
        self.a, self.me, self.p = a, me, dict(params or {})
        self._s: Status = "running"

    def start(self, params): self.p.update(params)
    def status(self) -> Status: return self._s
    def cancel(self) -> None: self.me.clear()

    def threats(self) -> list[Unit]:
        return sorted((u for u in self.a.units.values() if u.alive and u.team != self.me.team), key=self.me.dist)

    def goto(self, x, y, stop=1.0) -> bool:
        d = math.hypot(x - self.me.x, y - self.me.y)
        if d <= stop:
            self.me.move = (0.0, 0.0)
            return True
        self.me.move = (x - self.me.x, y - self.me.y)
        return False


class Engage(_R):
    def tick(self, state=None):
        m = self.me
        m.clear()
        ts = self.threats()
        if not ts:
            self._s = "succeeded"; return
        t = ts[0]
        if m.mag == 0:
            self.a.start_reload(m)
        d = m.dist(t)
        dx, dy = (t.x - m.x) / max(d, 1e-6), (t.y - m.y) / max(d, 1e-6)
        if not self.a.los(m, t) or d > 16:
            m.move = (dx, dy)
        else:
            side = 1 if (self.a.tick_n // 25) % 2 else -1
            m.move = (-dy * side + (0.4 * dx if d > 10 else -0.3 * dx), dx * side + (0.4 * dy if d > 10 else -0.3 * dy))
        m.shoot = t.id


class Disengage(_R):
    def tick(self, state=None):
        m = self.me
        m.clear()
        ts = self.threats()
        if not ts:
            self._s = "succeeded"; return
        cx = sum(t.x for t in ts[:3]) / min(3, len(ts)); cy = sum(t.y for t in ts[:3]) / min(3, len(ts))
        ax, ay = m.x - cx, m.y - cy
        if m.x < 5: ax += 2
        if m.x > self.a.size - 5: ax -= 2
        if m.y < 5: ay += 2
        if m.y > self.a.size - 5: ay -= 2
        m.move = (ax, ay)


class Reposition(_R):
    """Move to the far side of the nearest cover from the main threat."""
    def tick(self, state=None):
        m = self.me
        m.clear()
        ts = self.threats()
        spot, _ = self.a.nearest_cover(m.x, m.y, ts[0] if ts else None)
        if spot is None or self.goto(*spot, stop=0.6):
            self._s = "succeeded"
        if ts and self.a.los(m, ts[0]) and m.mag:
            m.shoot = ts[0].id


class Heal(_R):
    def tick(self, state=None):
        m = self.me
        m.clear()
        if not m.heal_t:
            if self.p.get("started") or not self.a.start_heal(m):
                self._s = "succeeded" if self.p.get("started") else "failed"
                return
            self.p["started"] = True


class Reload(_R):
    def tick(self, state=None):
        m = self.me
        m.clear()
        if not m.reload_t:
            if self.p.get("started") or not self.a.start_reload(m):
                self._s = "succeeded" if self.p.get("started") else "failed"
                return
            self.p["started"] = True


class Collect(_R):
    def tick(self, state=None):
        m = self.me
        m.clear()
        if not self.a.crates:
            self._s = "failed"; return
        c = min(self.a.crates, key=lambda c: math.hypot(c[0] + 0.5 - m.x, c[1] + 0.5 - m.y))
        self.goto(c[0] + 0.5, c[1] + 0.5, stop=0.3)


class Goto(_R):
    def tick(self, state=None):
        self.me.clear()
        if self.goto(self.p["x"], self.p["y"], stop=1.5):
            self._s = "succeeded"


class Explore(_R):
    def tick(self, state=None):
        self.me.clear()
        if "dest" not in self.p:
            self.p["dest"] = (self.a.rng.uniform(6, self.a.size - 6), self.a.rng.uniform(6, self.a.size - 6))
        if self.goto(*self.p["dest"], stop=2.0):
            self._s = "succeeded"


class Wait(_R):
    def tick(self, state=None):
        self.me.clear()


class SkirmishAdapter(GameAdapter):
    game = "skirmish"
    stepped = True

    def __init__(self, scenario: Scenario | str, seed: int = 0, config: dict | None = None):
        self.cfg = config or load_config()
        self.profile = self.cfg["profile"]
        self.rules = PlayRules.from_config(self.cfg["play_rules"])
        self.tick_hz = self.cfg["tick_hz"]
        self.scenario = SCENARIOS[scenario]() if isinstance(scenario, str) else scenario
        self.seed = seed
        self.a: Arena | None = None
        self.connected = False
        self.current: _R | None = None
        self.key = ""
        self._dt = self._dd = 0.0
        self._dead = False

    def connect(self, context: Context, address: str | None = None) -> None:
        self.rules.check(context, address)
        self.a = Arena(seed=self.seed)
        self.scenario.setup(self.a)
        self.connected = True

    @property
    def me(self) -> Unit:
        return self.a.units["bot"]

    def read_state(self) -> Arena:
        if not self.connected:
            raise RuntimeError("not connected")
        return self.a

    def facts(self, a: Arena) -> Facts:
        m = self.me
        ents = [Entity(u.id, u.kind, u.team != m.team, u.x, u.y, u.vx, u.vy, u.hp, u.hp_max,
                       facing_self=a.los(u, m), attack_ready=u.fire_cd == 0 and u.mag > 0)
                for u in a.units.values() if u.alive and u.id != m.id and m.dist(u) < 30]
        return Facts(t=a.t, x=m.x, y=m.y, vx=m.vx, vy=m.vy, hp=m.hp, hp_max=m.hp_max, entities=ents,
                     inventory={"ammo": m.mag + m.reserve, "medkit": m.medkits, "crate": m.crates},
                     alive=m.alive)

    def _threats(self, a: Arena) -> list[Unit]:
        return sorted((u for u in a.units.values() if u.alive and u.team != self.me.team), key=self.me.dist)

    def _amount(self, goal: Goal | None) -> int:
        for c in (goal.constraints if goal else []):
            if c.startswith("amount "):
                return int(c.split()[1])
        return 1

    def goal_status(self, a: Arena, goal: Goal):
        m = self.me
        if not m.alive:
            return "failed"
        verb, _, arg = goal.intent.partition(":")
        if verb == "gather":
            if m.crates >= self._amount(goal):
                return "succeeded"
            return "failed" if not a.crates else "active"
        if verb == "return":
            x, y = a.places.get(arg or "spawn", (m.x, m.y))
            return "succeeded" if math.hypot(m.x - x, m.y - y) <= 3.0 else "active"
        if verb == "engage" and goal.target:
            t = a.units.get(goal.target)
            return "succeeded" if t is None or not t.alive else "active"
        return "active"

    def check(self, condition: str, a: Arena) -> bool:
        parts = condition.split()
        if parts[0] == "hp_above":
            return self.me.hp > float(parts[1])
        raise ValueError(condition)

    def context(self, a: Arena, goal: Goal | None) -> str:
        ts = self._threats(a)
        if ts and (self.me.dist(ts[0]) <= self.cfg["combat_radius"] or (goal and goal.intent == "engage")):
            return "combat"
        if goal is not None and goal.intent.startswith("gather"):
            return "gathering"
        return "travel"

    def options(self, a: Arena, goal: Goal | None) -> list[str]:
        return list(self.cfg["option_sets"][self.context(a, goal)])

    def safe_default(self, a: Arena, goal: Goal | None) -> str:
        ctx = self.context(a, goal)
        if ctx == "combat":
            return "reposition"
        if goal is not None and goal.on_fail in self.cfg["option_sets"][ctx]:
            return goal.on_fail
        return "continue"

    def describe(self, a: Arena, goal: Goal | None, tier2: dict) -> str:
        m, g = self.me, goal
        goal_line = ("GOAL none" if g is None else
                     f"GOAL {g.intent}" + (f" target:{g.target}" if g.target else "")
                     + f" | CONSTRAINT {','.join(g.constraints) or 'none'}"
                     + f" | PRIORITY {'>'.join(g.priorities) or 'none'}")
        state = "reloading" if m.reload_t else "healing" if m.heal_t else "ready" if m.mag else "empty"
        self_line = (f"SELF hp {int(math.ceil(m.hp))}/100 ammo {m.mag}/{m.mag_size} reserve {m.reserve} "
                     f"medkits {m.medkits} weapon {state}")
        ts = self._threats(a)
        if ts:
            t = ts[0]
            d = m.dist(t)
            opp_line = (f"OPP {t.kind} dist {_q(d, 0.1) if d < 10 else _q(d, 1)} hp ~{int(math.ceil(t.hp))}/{int(t.hp_max)} "
                        f"los {'yes' if a.los(t, m) else 'no'} hit_chance {a.hit_chance(m, t):.1f}")
        else:
            opp_line = "OPP none"
        near = sum(1 for t in ts if m.dist(t) <= 15)
        spawn = a.places["spawn"]
        cov = a.nearest_cover(m.x, m.y, ts[0] if ts else None)[1]
        crate = min((math.hypot(c[0] - m.x, c[1] - m.y) for c in a.crates), default=None)
        prog = f"{m.crates}/{self._amount(g)}" if g and g.intent.startswith("gather") else "none"
        env_line = (f"ENV enemies {len(ts)} near {near} cover_dist {_q(cov, 1)} spawn_dist "
                    f"{_q(math.hypot(m.x - spawn[0], m.y - spawn[1]), 1)} resource_dist "
                    f"{_q(crate, 1) if crate is not None else 'none'} progress {prog}")
        pred = tier2.get("pred") or {}
        drv = tier2.get("drives") or {}
        wm = tier2.get("wm") or []
        return "\n".join([
            f"GAME {self.game}", goal_line, self_line, opp_line, env_line,
            ("PRED " + " ".join(f"{k} {v:.1f}" for k, v in sorted(pred.items()))) if pred else "PRED none",
            ("DRIVES " + " ".join(f"{k} {v:.1f}" for k, v in sorted(drv.items()))) if drv else "DRIVES none",
            f"WM last_actions {','.join(wm) if wm else 'none'}"])

    def summarise(self, a: Arena, goal: Goal | None) -> str:
        m = self.me
        ts = self._threats(a)
        return (f"[t={a.t:.1f}s] game: skirmish scenario {self.scenario.name}\n"
                f"status: health {m.hp:.0f}/100, ammo {m.mag}+{m.reserve}, medkits {m.medkits}\n"
                f"inventory: crate x{m.crates}\nthreats: "
                + (", ".join(f"{t.kind} at {m.dist(t):.0f}" for t in ts[:4]) or "none")
                + f"\nactive goal: {goal.intent if goal else 'none'}")

    def _reflex(self, choice: str, goal: Goal | None) -> _R:
        a, m = self.a, self.me
        verb, _, arg = choice.partition(":")
        if verb == "continue" and goal is not None:
            verb, _, arg = goal.intent.partition(":")
            if verb == "gather":
                return Collect(a, m)
            if verb == "wait":
                return Wait(a, m)
        if verb == "engage":
            return Engage(a, m)
        if verb == "disengage":
            return Disengage(a, m)
        if verb == "heal":
            return Heal(a, m)
        if verb == "reposition":
            return Reposition(a, m)
        if verb == "use_ability" and arg == "reload":
            return Reload(a, m)
        if verb == "gather":
            return Collect(a, m)
        if verb == "return":
            x, y = a.places.get(arg or "spawn", a.places["spawn"])
            return Goto(a, m, {"x": x, "y": y})
        if verb == "explore":
            return Explore(a, m)
        return Wait(a, m)

    def execute(self, choice: str, goal: Goal | None, a: Arena) -> None:
        key = choice + "|" + (goal.goal_id if goal else "")
        if self.current is not None and key == self.key and self.current.status() == "running":
            return
        if self.current is not None:
            self.current.cancel()
        self.current, self.key = self._reflex(choice, goal), key

    def tick(self) -> bool:
        if self.current is not None and self.me.alive:
            self.current.tick()
            if self.current.status() != "running":
                self.key = ""
        # autonomic: reload an empty magazine when nothing is in line of sight
        m = self.me
        if m.alive and m.mag == 0 and not any(self.a.los(t, m) for t in self._threats(self.a)):
            self.a.start_reload(m)
        self.a.step()
        s = self.scenario
        return not (s.success(self.a) or s.failed(self.a) or self.a.t >= s.time_limit_s)

    def episode_result(self) -> dict:
        s, m = self.scenario, self.me
        return {"scenario": s.name, "success": bool(s.success(self.a)), "t": round(self.a.t, 2),
                "deaths": int(not m.alive), "hp": round(m.hp, 1), "damage_dealt": m.damage_dealt,
                "damage_taken": m.damage_taken}

    def reflexes(self) -> dict:
        return {"engage": Engage, "disengage": Disengage, "heal": Heal, "reposition": Reposition,
                "use_ability:reload": Reload, "gather": Collect, "return": Goto, "explore": Explore, "wait": Wait}

    def reward(self, prev: Arena, a: Arena) -> dict[str, float]:
        m, rw, out = self.me, self.cfg["reward"], {}
        dt, dd = m.damage_taken - self._dt, m.damage_dealt - self._dd
        self._dt, self._dd = m.damage_taken, m.damage_dealt
        if dt:
            out["damage_taken"] = rw["damage_taken"] * dt
        if dd:
            out["damage_dealt"] = rw["damage_dealt"] * dd
        if not m.alive and not self._dead:
            self._dead = True
            out["death"] = rw["death"]
        return out

    def release_all(self) -> None:
        if self.current is not None:
            self.current.cancel()
        self.current, self.key = None, ""
        if self.a is not None:
            self.me.clear()

    def disconnect(self) -> None:
        self.connected = False
