"""Evaluate counter-pick adaptation (StyleSelector) per opponent style vs fixed policies."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
torch.set_num_threads(1)

from nervous_system.adapters.gridcraft import gamesense as gs  # noqa: E402
from nervous_system.adapters.gridcraft.combat import expert_action, load_policy  # noqa: E402
from nervous_system.adapters.gridcraft.styles import STYLES  # noqa: E402
from pvp_dataset import pure_engage  # noqa: E402
from train_adaptive import EVAL_SEED, StyleDuel  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
pol = load_policy(ROOT / "checkpoints/combat-reflex.pt")
players = {"reflex": lambda w, me, opp, o: pol.act(o, greedy=True)[0][0],
           "pure_scripted": lambda w, me, opp, o: pure_engage(w, me, opp),
           "expert_heal": lambda w, me, opp, o: expert_action(w, me, opp)}
card = json.loads((ROOT / "data/pvp-v1/card.json").read_text())
om = gs.load(ROOT / "checkpoints/opponent-model.pt")
res = {}
for style in STYLES:
    row = {}
    for name in list(players) + ["selector"]:
        wins, picks = 0, {}
        for i in range(ROUNDS):
            e = StyleDuel(EVAL_SEED + i, style, lambda: (lambda me, opp: np.zeros(0, np.float32)), max_s=90.0)
            sel = gs.StyleSelector(om, players, card["win_rate"], default="reflex") if name == "selector" else None
            while not e.done:
                if sel:
                    a = sel.act(e.w, e.me, e.opp)
                else:
                    from nervous_system.adapters.gridcraft.combat import observe
                    a = players[name](e.w, e.me, e.opp, observe(e.w, e.me, e.opp))
                e.step(a)
            wins += e.won
            if sel:
                picks[sel.current] = picks.get(sel.current, 0) + 1
        row[name] = wins / ROUNDS
        if picks:
            row["selector_final_pick"] = picks
    res[style] = row
    print(style, row, flush=True)
mean = {k: float(np.mean([res[s][k] for s in STYLES])) for k in list(players) + ["selector"]}
print("mean", mean)
out = ROOT / "docs/results/pvp-adaptive.json"
r = json.loads(out.read_text())
r["counter_pick_selector"] = {"rounds_per_style": ROUNDS, "per_style": res, "mean_win": mean,
                              "best_policy_per_style": gs.StyleSelector(om, players, card["win_rate"], "reflex").best}
out.write_text(json.dumps(r, indent=1, default=str) + "\n")
