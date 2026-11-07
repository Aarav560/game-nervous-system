"""Learned combat reflex for GridCraft (Architecture §6, §12.4; Roadmap Phase 4).

One observation in, one action out, every 50 ms tick. Small actor-critic MLP with auxiliary
prediction heads (`will_hit`, `will_be_hit` over the next 0.5 s) whose outputs can be fed up
to Laya as `PRED` features — the design lesson taken from the Minecraft Combat AI reference.

Trained by scripts/train_combat.py: behaviour cloning on the scripted engage reflex ->
PPO vs the scripted practice bot -> league (past snapshots + an exploiter).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ...reflex import Status
from .reflexes import _Base
from .world import Agent, World

OBS_DIM = 24
MOVES = 9                     # stop + 8 directions relative to the opponent
HEADS = {"move": MOVES, "sprint": 2, "attack": 2, "eat": 2}


def _dirs() -> list[tuple[float, float]]:
    # index 0 = stop; 1..8 = angle k*45deg from "toward the opponent", counter-clockwise
    return [(0.0, 0.0)] + [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]


DIRS = _dirs()


def observe(w: World, me: Agent, opp: Agent) -> np.ndarray:
    d = me.dist(opp)
    ux, uy = (opp.x - me.x) / max(d, 1e-6), (opp.y - me.y) / max(d, 1e-6)
    px, py = -uy, ux
    proj = lambda vx, vy: (vx * ux + vy * uy, vx * px + vy * py)  # noqa: E731
    mv, mp = proj(me.vx, me.vy)
    ov, op = proj(opp.vx, opp.vy)
    of = opp.fx * -ux + opp.fy * -uy                  # opponent facing me
    wall = min(me.x, me.y, w.size - me.x, w.size - me.y)
    return np.array([
        min(d, 20) / 10, min(d, 6) / 6, float(d <= me.reach), float(d <= opp.reach + 0.6),
        mv / 6, mp / 6, ov / 6, op / 6,
        me.cd / me.cooldown, opp.cd / max(1, opp.cooldown),
        me.hp / 20, opp.hp / 20, float(bool(me.eating)), float(bool(opp.eating)),
        me.count("golden_apple") / 2, opp.count("golden_apple") / 2,
        float(me.offhand == "totem"), float(opp.offhand == "totem"),
        of, float(me.sprint), min(wall, 10) / 10, me.regen_t / 100, opp.regen_t / 100,
        (me.hp - opp.hp) / 20,
    ], dtype=np.float32)


def apply_action(w: World, me: Agent, opp: Agent, a: dict[str, int]) -> None:
    me.clear_controls()
    d = me.dist(opp)
    ux, uy = (opp.x - me.x) / max(d, 1e-6), (opp.y - me.y) / max(d, 1e-6)
    cx, sy = DIRS[a["move"]]
    me.move = (cx * ux - sy * uy, cx * uy + sy * ux)
    me.sprint = bool(a["sprint"]) and me.hunger > 6
    if a["attack"]:
        me.attack = opp.id
    if a["eat"] and not me.eating and me.count("golden_apple") > 0 and me.hp < me.hp_max:
        w.start_eating(me, "golden_apple")


def expert_action(w: World, me: Agent, opp: Agent) -> dict[str, int]:
    """Labels from the scripted engage reflex (+ the playbook's heal rule) for behaviour cloning."""
    d = me.dist(opp)
    ux, uy = (opp.x - me.x) / max(d, 1e-6), (opp.y - me.y) / max(d, 1e-6)
    eat = int(me.hp <= 8 and me.count("golden_apple") > 0 and d > 4.0 and not me.eating)
    if me.hp <= 8 and me.count("golden_apple") > 0 and d <= 4.0:
        mx, my, sprint = -ux, -uy, 1                               # get space to eat
    elif me.cd <= 2:
        mx, my, sprint = ux, uy, 1
    elif d < 2.6:
        mx, my, sprint = -ux - 0.5 * uy, -uy + 0.5 * ux, 0
    else:
        side = 1 if (w.tick_n // 20) % 2 else -1
        mx, my, sprint = side * -uy, side * ux, 0
    # nearest relative direction
    n = math.hypot(mx, my)
    if n < 1e-6:
        mv = 0
    else:
        rel = math.atan2(-uy * mx + ux * my, ux * mx + uy * my)        # angle relative to "toward"
        mv = 1 + int(round(rel / (math.pi / 4))) % 8
    return {"move": mv, "sprint": sprint, "attack": 1, "eat": eat}


class CombatPolicy(nn.Module):
    def __init__(self, hidden: int = 128, obs_dim: int = OBS_DIM):
        super().__init__()
        self.obs_dim = obs_dim
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in HEADS.items()})
        self.value = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, 2)            # will_hit, will_be_hit (next 0.5 s), logits

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return {k: head(h) for k, head in self.heads.items()}, self.value(h).squeeze(-1), self.aux(h)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[list[dict[str, int]], dict]:
        o = torch.as_tensor(obs, dtype=torch.float32)
        single = o.dim() == 1
        if single:
            o = o[None]
        logits, v, aux = self(o)
        acts = {}
        logp = torch.zeros(o.shape[0])
        for k, lg in logits.items():
            dist = torch.distributions.Categorical(logits=lg)
            a = lg.argmax(-1) if greedy else dist.sample()
            acts[k] = a
            logp += dist.log_prob(a)
        out = [{k: int(acts[k][i]) for k in HEADS} for i in range(o.shape[0])]
        return out, {"logp": logp, "value": v, "aux": torch.sigmoid(aux), "acts": acts}


def save_policy(policy: CombatPolicy, path: str | Path, meta: dict | None = None) -> None:
    torch.save({"state_dict": policy.state_dict(), "obs_dim": policy.obs_dim, "heads": HEADS,
                "meta": meta or {}}, str(path))


def load_policy(path: str | Path) -> CombatPolicy:
    ck = torch.load(str(path), weights_only=True, map_location="cpu")   # CLAUDE.md: weights only
    w0 = ck["state_dict"]["trunk.0.weight"]
    p = CombatPolicy(hidden=w0.shape[0], obs_dim=w0.shape[1])
    p.load_state_dict(ck["state_dict"])
    p.eval()
    return p


class LearnedEngage(_Base):
    """Drop-in replacement for the scripted `engage` reflex."""
    name = "engage"

    def __init__(self, world: World, bot: Agent, policy: CombatPolicy, features=None):
        super().__init__(world, bot)
        self.policy = policy
        self.features = features          # AdaptiveFeatures for style-adaptive policies
        self.last_aux: dict[str, float] = {}

    def tick(self, state: Any = None) -> None:
        b = self.bot
        b.mine_cell = None
        tid = self.params.get("target")
        t = self.w.agents.get(tid) if tid else None
        if t is None or not t.alive:
            hs = self._hostiles(16)
            if not hs:
                b.clear_controls()
                self._status = "succeeded"
                return
            t = hs[0]
            self.params["target"] = t.id
        obs = observe(self.w, b, t)
        if self.policy.obs_dim > OBS_DIM:
            obs = np.concatenate([obs, self.features(b, t)])
        acts, info = self.policy.act(obs, greedy=True)
        apply_action(self.w, b, t, acts[0])
        aux = info["aux"][0]
        self.last_aux = {"will_hit": float(aux[0]), "will_be_hit": float(aux[1])}

    def status(self) -> Status:
        return self._status
