"""Build a large GridCraft PvP replay dataset ("footage") and a best-rounds subset.

Every tick of every round is recorded from the bot's side: combat observation, opponent-habit
tracker features, the action taken, auxiliary labels (hit / got hit in the next 0.5 s), the
opponent's style, and the round outcome. "Best footage" = rounds won with top-half damage
efficiency; it is the behaviour-cloning set for the adaptive policy. The full set (wins and
losses) trains the game-sense models.

    python scripts/pvp_dataset.py --rounds 400 --out data/pvp-v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nervous_system.adapters.gridcraft.combat import HEADS, apply_action, expert_action, load_policy, observe  # noqa: E402
from nervous_system.adapters.gridcraft.scenarios import duel  # noqa: E402
from nervous_system.adapters.gridcraft.styles import STYLES, StyleTracker, styled_ai  # noqa: E402
from nervous_system.adapters.gridcraft.world import World  # noqa: E402
from train_combat import aux_labels  # noqa: E402

torch.set_num_threads(1)


def pure_engage(w, me, opp):
    a = expert_action(w, me, opp)
    if a["eat"] or (me.hp <= 8 and me.count("golden_apple") > 0):
        d = me.dist(opp)
        a = {"move": 1 if me.cd <= 2 else (5 if d < 2.6 else 3), "sprint": int(me.cd <= 2), "attack": 1, "eat": 0}
    return a


def make_round(seed: int, style: str):
    w = World(seed=seed)
    duel().setup(w)
    me, opp = w.agents["bot"], w.agents["opp"]
    r = random.Random(seed)
    for a in (me, opp):                              # kit variety
        a.inventory["golden_apple"] = r.randint(1, 3)
        if r.random() < 0.3:
            a.offhand, a.inventory["totem"] = None, 0
    me.x += r.uniform(-4, 4); me.y += r.uniform(-6, 6)
    opp.x += r.uniform(-4, 4); opp.y += r.uniform(-6, 6)
    opp.ai = styled_ai(style, "bot", random.Random(seed + 1))
    return w, me, opp


def play(seed: int, style: str, player, max_s: float = 90.0) -> dict:
    w, me, opp = make_round(seed, style)
    tr = StyleTracker()
    O, T, A = [], [], {k: [] for k in HEADS}
    hist = []
    dd = dt = 0.0
    while w.tick_n < max_s * 20 and me.alive and opp.alive:
        o = observe(w, me, opp)
        f = tr.update(me, opp)
        a = player(w, me, opp, o)
        O.append(o); T.append(f)
        for k in HEADS:
            A[k].append(a[k])
        apply_action(w, me, opp, a)
        if me.offhand is None and me.count("totem"):
            me.offhand = "totem"
        w.step()
        hist.append((me.damage_dealt > dd, me.damage_taken > dt))
        dd, dt = me.damage_dealt, me.damage_taken
    won = (not opp.alive) and me.alive
    return {"obs": np.array(O, np.float16), "track": np.array(T, np.float16),
            "act": np.stack([np.array(A[k], np.int8) for k in HEADS], 1), "aux": aux_labels(hist).astype(np.int8),
            "won": won, "lost": not me.alive, "t": w.t, "dealt": me.damage_dealt, "taken": me.damage_taken}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=400, help="rounds per (player, style) pair")
    ap.add_argument("--policy", default=str(ROOT / "checkpoints/combat-reflex.pt"))
    ap.add_argument("--out", default=str(ROOT / "data/pvp-v1"))
    a = ap.parse_args()
    pol = load_policy(a.policy)
    players = {
        "reflex": lambda w, me, opp, o: pol.act(o, greedy=True)[0][0],
        "reflex_sampled": lambda w, me, opp, o: pol.act(o, greedy=False)[0][0],
        "pure_scripted": lambda w, me, opp, o: pure_engage(w, me, opp),
        "expert_heal": lambda w, me, opp, o: expert_action(w, me, opp),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = {k: [] for k in ("obs", "track", "act", "aux")}
    meta = {k: [] for k in ("round", "style", "player", "won", "t_in_round")}
    rounds, t0, rid = [], time.time(), 0
    for pi, (pname, player) in enumerate(players.items()):
        for si, style in enumerate(STYLES):
            for i in range(a.rounds):
                seed = 1_000_000 + rid
                r = play(seed, style, player)
                n = len(r["obs"])
                for k in cols:
                    cols[k].append(r[k])
                meta["round"].append(np.full(n, rid, np.int32))
                meta["style"].append(np.full(n, si, np.int8))
                meta["player"].append(np.full(n, pi, np.int8))
                meta["won"].append(np.full(n, int(r["won"]), np.int8))
                meta["t_in_round"].append(np.arange(n, dtype=np.int16))
                rounds.append({"round": rid, "seed": seed, "player": pname, "style": style, "won": r["won"],
                               "lost": r["lost"], "t": round(r["t"], 2), "dealt": round(r["dealt"], 1),
                               "taken": round(r["taken"], 1), "frames": n})
                rid += 1
            wr = np.mean([x["won"] for x in rounds[-a.rounds:]])
            print(f"{pname:15s} vs {style:9s} win {wr:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    data = {k: np.concatenate(v) for k, v in cols.items()} | {k: np.concatenate(v) for k, v in meta.items()}
    np.savez_compressed(out / "all.npz", **data)
    # best footage: won rounds, top half by damage efficiency, within each style
    best = set()
    for style in STYLES:
        won = [r for r in rounds if r["style"] == style and r["won"]]
        won.sort(key=lambda r: -(r["dealt"] / (r["taken"] + 1.0)))
        best |= {r["round"] for r in won[: max(1, len(won) // 2)]}
    mask = np.isin(data["round"], list(best))
    np.savez_compressed(out / "best.npz", **{k: v[mask] for k, v in data.items()})
    (out / "rounds.json").write_text(json.dumps(rounds))
    card = {"rounds": len(rounds), "frames": int(len(data["obs"])), "best_rounds": len(best),
            "best_frames": int(mask.sum()), "players": list(players), "styles": STYLES,
            "win_rate": {f"{p}/{s}": float(np.mean([r["won"] for r in rounds if r["player"] == p and r["style"] == s]))
                         for p in players for s in STYLES},
            "best_by_player": {p: sum(1 for r in rounds if r["round"] in best and r["player"] == p) for p in players},
            "gen_s": round(time.time() - t0, 1)}
    (out / "card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({k: card[k] for k in ("rounds", "frames", "best_rounds", "best_frames", "best_by_player", "gen_s")}, indent=1))


if __name__ == "__main__":
    main()
