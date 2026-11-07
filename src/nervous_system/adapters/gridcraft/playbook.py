"""Scripted expert for GridCraft: the stand-in brain's game knowledge (no LLM available here).

`label()` reads only Laya's state text, exactly what an LLM labeller would see, so the
distillation data is the same shape whichever brain produced it.
"""
from __future__ import annotations

import re


def parse(text: str) -> dict:
    f: dict = {}
    for key, pat in {
        "hp": r"SELF hp (\d+)/", "hunger": r"hunger (\d+)", "offhand": r"totem offhand (\w+)",
        "spare": r"spare (\d+)", "gapples": r"gapples (\d+)", "food": r"food (\d+)",
        "cooldown": r"cooldown (\w+)", "opp_kind": r"OPP (\w+)", "opp_dist": r"OPP \w+ dist ([\d.]+)",
        "opp_hp": r"hp ~(\d+)/", "crit": r"crit_threat (\w+)", "enemies": r"ENV enemies (\d+)",
        "near": r"near (\d+)", "res": r"resource_dist (\w+)", "progress": r"progress (\S+)",
        "goal": r"GOAL (\S+)", "fear": r"fear ([\d.]+)", "hp_loss": r"hp_loss_1s ([\d.]+)",
    }.items():
        m = re.search(pat, text)
        f[key] = m.group(1) if m else None
    for k in ("hp", "hunger", "spare", "gapples", "food", "enemies", "near", "opp_hp"):
        f[k] = int(f[k]) if f[k] is not None else 0
    f["opp_dist"] = float(f["opp_dist"]) if f["opp_dist"] else None
    f["fear"] = float(f["fear"]) if f["fear"] else 0.0
    done = f["progress"] not in (None, "none") and "/" in f["progress"]
    if done:
        have, need = f["progress"].split("/")
        f["progress_done"] = int(have) >= int(need)
    else:
        f["progress_done"] = False
    return f


class GridCraftPlaybook:
    def plan(self, mission: str, world_summary: str) -> list[dict]:
        inv = dict((m.group(1), int(m.group(2))) for m in re.finditer(r"(\w+) x(\d+)", world_summary))
        m = mission.lower()
        if "logs" in m:
            if inv.get("log", 0) >= 64:
                return [{"intent": "return:base", "on_fail": "disengage"}]
            return [{"intent": "gather:wood", "constraints": ["amount 64"], "priorities": ["survive", "finish_task"],
                     "on_fail": "return:base"},
                    {"intent": "return:base", "on_fail": "disengage"}]
        if "survive" in m:
            return [{"intent": "wait", "priorities": ["survive"], "on_fail": "disengage"}]
        if "1v1" in m or "duel" in m:
            return [{"intent": "engage", "target": "opp", "priorities": ["survive", "win"], "on_fail": "disengage"}]
        if "diamond" in m:
            steps = []
            if not inv.get("iron_pickaxe"):
                if inv.get("iron_ore", 0) < 3:
                    steps.append({"intent": "gather:iron_ore", "constraints": ["amount 3"], "on_fail": "return:base"})
                steps += [{"intent": "return:base", "on_fail": "disengage"},
                          {"intent": "build:iron_pickaxe", "on_fail": "return:base"}]
            steps.append({"intent": "gather:diamond", "constraints": ["amount 1"], "on_fail": "return:base"})
            return steps
        if "base" in m:
            return [{"intent": "return:base", "on_fail": "disengage"}]
        return [{"intent": "wait", "on_fail": "disengage"}]

    def label(self, text: str, options: list[str]) -> str:
        f = parse(text)
        if "engage" in options:                       # combat
            return self._combat(f, options)
        if "explore" in options:                      # gathering
            if f["hp"] <= 8 and (f["food"] or f["gapples"]):
                return "heal"
            if f["progress_done"]:
                return "return:base"
            if f["res"] in (None, "none"):
                return "explore"
            return "continue"
        # travel
        return "continue"

    def _combat(self, f: dict, options: list[str]) -> str:
        if f["offhand"] == "no" and f["spare"] > 0:
            return "use_ability:totem_swap"
        dist = f["opp_dist"] if f["opp_dist"] is not None else 99.0
        player = f["opp_kind"] == "player"
        fighting_goal = (f["goal"] or "").startswith("engage") or (f["goal"] or "") == "wait"
        if f["hp"] <= 8 and f["gapples"] > 0:
            return "heal" if dist > 4.0 else "disengage"     # get space, then eat
        if not player and f["hp"] <= 5 and f["offhand"] == "no":
            return "disengage"                          # vs a player, fleeing just gives up a free crit
        if f["near"] >= 3 and f["hp"] <= 12:
            return "disengage"
        if not fighting_goal and not player and f["hp"] <= 10:
            return "disengage"
        return "engage"

    def on_event(self, description: str, world_summary: str, goal_intent: str | None) -> dict | None:
        return None
