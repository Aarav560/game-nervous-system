"""Game sense for GridCraft PvP: opponent-style recognition + win probability.

- OpponentModel: StyleTracker features (last ~2 s of the opponent's habits) -> style posterior.
- WinModel: current fight state + tracker -> P(I win this fight). Exposed to Laya as the
  `PRED win_prob` feature and used by the brain playbook to decide when fighting on beats
  retreating ("game sense").
Both are tiny MLPs trained on the PvP replay dataset (scripts/pvp_dataset.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .combat import OBS_DIM, observe
from .styles import STYLES, TRACK_DIM, StyleTracker
from .world import Agent, World


class OpponentModel(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(TRACK_DIM, hidden), nn.ReLU(), nn.Linear(hidden, len(STYLES)))

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def probs(self, track: np.ndarray) -> np.ndarray:
        return torch.softmax(self(torch.as_tensor(track)[None]), -1)[0].numpy()


class WinModel(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM + TRACK_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def prob(self, x: np.ndarray) -> float:
        return float(torch.sigmoid(self(torch.as_tensor(x)[None]))[0])


def save(model: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "kind": type(model).__name__, "meta": meta or {}}, str(path))


def load(path: str | Path) -> nn.Module:
    ck = torch.load(str(path), weights_only=True, map_location="cpu")
    m = {"OpponentModel": OpponentModel, "WinModel": WinModel}[ck["kind"]]()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


class AdaptiveFeatures:
    """Per-opponent tracker + style posterior, appended to the combat observation."""

    EXTRA = TRACK_DIM + len(STYLES)

    def __init__(self, opponent_model: OpponentModel | None):
        self.om = opponent_model
        self.trackers: dict[str, StyleTracker] = {}

    def __call__(self, me: Agent, opp: Agent) -> np.ndarray:
        tr = self.trackers.setdefault(opp.id, StyleTracker())
        f = tr.update(me, opp)
        p = self.om.probs(f) if self.om is not None else np.full(len(STYLES), 1 / len(STYLES), np.float32)
        return np.concatenate([f, p]).astype(np.float32)

    def style(self, opp_id: str) -> tuple[str, float] | None:
        tr = self.trackers.get(opp_id)
        if tr is None or self.om is None or tr.n < 20:
            return None
        p = self.om.probs(tr.f)
        return STYLES[int(p.argmax())], float(p.max())


def full_obs(w: World, me: Agent, opp: Agent, feats: AdaptiveFeatures) -> np.ndarray:
    return np.concatenate([observe(w, me, opp), feats(me, opp)])
