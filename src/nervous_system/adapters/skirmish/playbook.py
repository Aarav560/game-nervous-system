"""Scripted expert for Skirmish (stand-in brain); reads only Laya's state text."""
from __future__ import annotations

import re


def parse(text: str) -> dict:
    g = lambda p, cast=str, d=None: (cast(m.group(1)) if (m := re.search(p, text)) else d)  # noqa: E731
    return {"hp": g(r"SELF hp (\d+)/", int, 100), "mag": g(r"ammo (\d+)/", int, 0),
            "reserve": g(r"reserve (\d+)", int, 0), "medkits": g(r"medkits (\d+)", int, 0),
            "weapon": g(r"weapon (\w+)", str, "ready"), "opp": g(r"OPP (\w+)", str, "none"),
            "dist": g(r"OPP \w+ dist ([\d.]+)", float, 99.0), "los": g(r"los (\w+)", str, "no"),
            "hit": g(r"hit_chance ([\d.]+)", float, 0.0), "near": g(r"near (\d+)", int, 0),
            "progress": g(r"progress (\S+)", str, "none"), "res": g(r"resource_dist (\w+)", str, "none"),
            "goal": g(r"GOAL (\S+)", str, "none")}


class SkirmishPlaybook:
    def plan(self, mission: str, world_summary: str) -> list[dict]:
        m = mission.lower()
        if "crate" in m:
            crates = int(re.search(r"crate x(\d+)", world_summary).group(1)) if "crate x" in world_summary else 0
            steps = [] if crates >= 4 else [{"intent": "gather:crates", "constraints": ["amount 4"],
                                             "on_fail": "return:spawn"}]
            return steps + [{"intent": "return:spawn", "on_fail": "disengage"}]
        if "1v1" in m:
            return [{"intent": "engage", "target": "opp", "on_fail": "reposition"}]
        return [{"intent": "wait", "priorities": ["survive"], "on_fail": "reposition"}]

    def label(self, text: str, options: list[str]) -> str:
        f = parse(text)
        if "engage" in options:
            if f["weapon"] in ("reloading", "healing"):
                return "reposition" if f["los"] == "yes" else "wait" if "wait" in options else "reposition"
            if f["mag"] == 0 and f["reserve"] > 0:
                return "use_ability:reload" if f["los"] == "no" or f["dist"] > 14 else "reposition"
            if f["hp"] <= 45 and f["medkits"] > 0:
                return "heal" if f["los"] == "no" else "reposition"
            if f["near"] >= 3 and f["hp"] <= 60:
                return "disengage"
            if f["mag"] == 0 and f["reserve"] == 0:
                return "disengage"
            return "engage"
        if "explore" in options:
            if f["hp"] <= 45 and f["medkits"] > 0:
                return "heal"
            if f["res"] == "none":
                return "explore"
            return "continue"
        return "continue"

    def on_event(self, description: str, world_summary: str, goal_intent: str | None) -> dict | None:
        return None
