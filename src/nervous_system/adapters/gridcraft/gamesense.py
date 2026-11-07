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


class StyleSelector:
    """Adaptation by counter-picking: identify the opponent's style, then switch to the policy
    with the best record against that style in the replay dataset (card.json win table)."""

    def __init__(self, opponent_model: OpponentModel, players: dict, win_table: dict,
                 default: str, min_conf: float = 0.6):
        self.feats = AdaptiveFeatures(opponent_model)
        self.players = players                      # name -> act(w, me, opp, obs) -> action
        self.best = {}
        for style in STYLES:
            cands = {p: win_table.get(f"{p}/{style}", 0.0) for p in players}
            self.best[style] = max(cands, key=cands.get)
        self.default, self.min_conf = default, min_conf
        self.current = default
        self.table = np.array([[win_table.get(f"{p}/{s}", 0.0) for s in STYLES] for p in players])
        self.names = list(players)

    def act(self, w: World, me: Agent, opp: Agent) -> dict:
        from .combat import observe
        self.feats(me, opp)
        tr = self.feats.trackers.get(opp.id)
        if tr is not None and tr.n >= 20 and self.feats.om is not None:
            p = self.feats.om.probs(tr.f)
            # expected win rate of each policy under the style posterior (robust to misreads)
            self.current = self.names[int((self.table @ p).argmax())]
        return self.players[self.current](w, me, opp, observe(w, me, opp))
