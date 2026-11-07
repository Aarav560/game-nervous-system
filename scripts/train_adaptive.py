"""Game sense + style adaptation for the GridCraft combat reflex.

1. OpponentModel: opponent-habit tracker -> play style (6 styles). Accuracy vs seconds observed.
2. WinModel ("game sense"): fight state + tracker -> P(win). AUC / Brier on held-out rounds.
3. Adaptive policy: obs = combat obs + tracker + style posterior. BC on the best footage, then PPO +
   league over all styles. Evaluated per style vs the non-adaptive reflex, plus an ablation with
   the adaptive inputs blanked (does it actually use them?).

    python scripts/train_adaptive.py --data data/pvp-v1
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nervous_system.adapters.gridcraft import gamesense as gs  # noqa: E402
from nervous_system.adapters.gridcraft.combat import (  # noqa: E402
    HEADS, OBS_DIM, CombatPolicy, apply_action, load_policy, observe, save_policy)
from nervous_system.adapters.gridcraft.styles import STYLES, TRACK_DIM  # noqa: E402
from pvp_dataset import make_round  # noqa: E402
from train_combat import aux_labels, ppo_update  # noqa: E402

torch.set_num_threads(1)
EXTRA = gs.AdaptiveFeatures.EXTRA
EVAL_SEED = 7_000_000


def split_rounds(d, frac=0.1, seed=0):
    rounds = np.unique(d["round"])
    rng = np.random.default_rng(seed)
    test = set(rng.choice(rounds, max(1, int(len(rounds) * frac)), replace=False).tolist())
    m = np.isin(d["round"], list(test))
    return ~m, m


def fit(model, X, y, loss_fn, epochs=6, bs=4096, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


# ------------------------------------------------------------------------------ env
class StyleDuel:
    def __init__(self, seed, style, feats_fn, opponent=None, max_s=60.0):
        self.w, self.me, self.opp = make_round(seed, style)
        self.style = style
        self.opponent = opponent              # None (scripted style) or (policy, feats) acting as opp
        if opponent is not None:
            self.opp.ai = None
        self.feats = feats_fn()
        self.max_ticks = int(max_s * 20)
        self.done = self.won = False
        self._dd = self._dt = 0.0
        self.hist = []

    def obs(self):
        return np.concatenate([observe(self.w, self.me, self.opp), self.feats(self.me, self.opp)])

    def step(self, a, opp_a=None):
        apply_action(self.w, self.me, self.opp, a)
        if opp_a is not None:
            apply_action(self.w, self.opp, self.me, opp_a)
        for x in (self.me, self.opp):
            if x.offhand is None and x.count("totem"):
                x.offhand = "totem"
        self.w.step()
        dealt, taken = self.me.damage_dealt - self._dd, self.me.damage_taken - self._dt
        self._dd, self._dt = self.me.damage_dealt, self.me.damage_taken
        self.hist.append((dealt > 0, taken > 0))
        r = 0.1 * (dealt - taken)
        if not self.opp.alive or not self.me.alive or self.w.tick_n >= self.max_ticks:
            self.done = True
            self.won = (not self.opp.alive) and self.me.alive
            r += 5.0 if self.won else (-5.0 if not self.me.alive else -1.0)
        return r


def rollout(policy, om, n_envs, seed, snapshots):
    rng = random.Random(seed)
    envs = []
    for i in range(n_envs):
        style = rng.choice(STYLES)
        opp = None
        if snapshots and rng.random() < 0.2:
            snap = rng.choice(snapshots)
            opp = (snap, gs.AdaptiveFeatures(om))
        envs.append(StyleDuel(seed * 1000 + i, style, lambda: gs.AdaptiveFeatures(om), opp))
    buf = [dict(obs=[], act=[], logp=[], val=[], rew=[]) for _ in envs]
    while not all(e.done for e in envs):
        live = [i for i, e in enumerate(envs) if not e.done]
        obs = np.stack([envs[i].obs() for i in live])
        acts, info = policy.act(obs)
        for j, i in enumerate(live):
            e = envs[i]
            oa = None
            if e.opponent is not None:
                pol, f = e.opponent
                oa = pol.act(np.concatenate([observe(e.w, e.opp, e.me), f(e.opp, e.me)]))[0][0]
            r = e.step(acts[j], oa)
            b = buf[i]
            b["obs"].append(obs[j]); b["act"].append(acts[j]); b["logp"].append(float(info["logp"][j]))
            b["val"].append(float(info["value"][j])); b["rew"].append(r)
    for b, e in zip(buf, envs):
        b["aux"] = aux_labels(e.hist)
    return buf, envs


def evaluate(act_fn, feats_fn, rounds, style, seed0=EVAL_SEED):
    wins = 0
    ident = []
    for i in range(rounds):
        e = StyleDuel(seed0 + i, style, feats_fn, max_s=90.0)
        first_correct = None
        while not e.done:
            o = e.obs()
            e.step(act_fn(o))
            if first_correct is None and hasattr(e.feats, "style"):
                s = e.feats.style(e.opp.id)
                if s and s[0] == style:
                    first_correct = e.w.t
        wins += e.won
        ident.append(first_correct)
    got = [t for t in ident if t is not None]
    return {"win_rate": wins / rounds, "identified_frac": len(got) / rounds,
            "median_s_to_identify": float(np.median(got)) if got else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/pvp-v1"))
    ap.add_argument("--ppo-iters", type=int, default=150)
    ap.add_argument("--envs", type=int, default=36)
    ap.add_argument("--eval-rounds", type=int, default=150)
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--results", default=str(ROOT / "docs/results/pvp-adaptive.json"))
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    t0 = time.time()
    rep: dict = {"data": json.loads((Path(a.data) / "card.json").read_text())}
    d = dict(np.load(Path(a.data) / "all.npz"))
    tr, te = split_rounds(d)

    # 1. opponent model -----------------------------------------------------------------
    warm = d["t_in_round"] >= 20                       # at least 1 s of observation
    Xt = torch.tensor(d["track"].astype(np.float32))
    yt = torch.tensor(d["style"].astype(np.int64))
    om = fit(gs.OpponentModel(), Xt[tr & warm], yt[tr & warm], nn.CrossEntropyLoss())
    with torch.no_grad():
        pred = om(Xt[te]).argmax(-1).numpy()
    ys, tt = d["style"][te], d["t_in_round"][te]
    rep["opponent_model"] = {
        "test_accuracy_after_1s": float((pred == ys)[tt >= 20].mean()),
        "accuracy_by_seconds_observed": {f"{lo // 20}-{hi // 20}s": float((pred == ys)[(tt >= lo) & (tt < hi)].mean())
                                         for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 160), (160, 10_000)]
                                         if ((tt >= lo) & (tt < hi)).any()},
        "per_style_accuracy": {s: float((pred == ys)[(ys == i) & (tt >= 20)].mean()) for i, s in enumerate(STYLES)},
        "chance": 1 / len(STYLES)}
    gs.save(om, Path(a.out_dir) / "opponent-model.pt", rep["opponent_model"])
    print("opponent model", rep["opponent_model"], flush=True)

    # 2. win model (game sense) ---------------------------------------------------------
    Xw = torch.tensor(np.concatenate([d["obs"], d["track"]], 1).astype(np.float32))
    yw = torch.tensor(d["won"].astype(np.float32))
    wm = fit(gs.WinModel(), Xw[tr], yw[tr], nn.BCEWithLogitsLoss())
    with torch.no_grad():
        p = torch.sigmoid(wm(Xw[te])).numpy()
    y = d["won"][te]
    base_rate = float(d["won"][tr].mean())
    rep["win_model"] = {"test_auc": auc(p, y), "test_brier": float(((p - y) ** 2).mean()),
                        "brier_of_base_rate": float(((base_rate - y) ** 2).mean()),
                        "auc_by_seconds_into_fight": {f">={s}s": auc(p[tt >= s * 20], y[tt >= s * 20])
                                                      for s in (0, 2, 5) if (tt >= s * 20).any()}}
    gs.save(wm, Path(a.out_dir) / "win-model.pt", rep["win_model"])
    print("win model", rep["win_model"], flush=True)
    del Xw, yw

    # 3. adaptive policy ----------------------------------------------------------------
    b = dict(np.load(Path(a.data) / "best.npz"))
    with torch.no_grad():
        sp = torch.softmax(om(torch.tensor(b["track"].astype(np.float32))), -1).numpy()
    Xb = torch.tensor(np.concatenate([b["obs"].astype(np.float32), b["track"].astype(np.float32), sp], 1))
    Ab = {k: torch.tensor(b["act"][:, i].astype(np.int64)) for i, k in enumerate(HEADS)}
    AX = torch.tensor(b["aux"].astype(np.float32))
    policy = CombatPolicy(obs_dim=OBS_DIM + EXTRA)
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    for ep in range(6):
        perm = torch.randperm(len(Xb))
        for i in range(0, len(Xb), 2048):
            idx = perm[i:i + 2048]
            lg, _, aux = policy(Xb[idx])
            loss = sum(ce(lg[k], Ab[k][idx]) for k in HEADS) + 0.5 * bce(aux, AX[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    rep["bc_frames"] = int(len(Xb))
    print(f"BC on {len(Xb)} best-footage frames done ({time.time() - t0:.0f}s)", flush=True)

    feats = lambda: gs.AdaptiveFeatures(om)  # noqa: E731
    greedy = lambda pol: (lambda o: pol.act(o, greedy=True)[0][0])  # noqa: E731
    rep["after_bc"] = {s: evaluate(greedy(policy), feats, a.eval_rounds // 3, s)["win_rate"] for s in STYLES}
    print("after BC", rep["after_bc"], flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    snapshots = []
    curve = []
    for it in range(a.ppo_iters):
        buf, envs = rollout(policy, om, a.envs, 50_000 + it, snapshots)
        ppo_update(policy, opt, buf)
        curve.append(float(np.mean([e.won for e in envs if e.opponent is None] or [0])))
        if it % 30 == 29:
            snapshots.append(copy.deepcopy(policy).eval())
        if it % 10 == 9:
            print(f"ppo {it + 1}: win vs styles {np.mean(curve[-10:]):.2f} ({time.time() - t0:.0f}s)", flush=True)
    rep["ppo_curve"] = curve
    save_policy(policy, Path(a.out_dir) / "combat-adaptive.pt", {"obs_dim": OBS_DIM + EXTRA, "styles": STYLES})

    # 4. evaluation ---------------------------------------------------------------------
    base = load_policy(ROOT / "checkpoints/combat-reflex.pt")
    blank = lambda: (lambda me, opp: np.concatenate([np.zeros(TRACK_DIM, np.float32),  # noqa: E731
                                                     np.full(len(STYLES), 1 / len(STYLES), np.float32)]))
    ev = {}
    for s in STYLES:
        ev[s] = {
            "non_adaptive_reflex": evaluate(lambda o: base.act(o[:OBS_DIM], greedy=True)[0][0], feats, a.eval_rounds, s)["win_rate"],
            "adaptive": evaluate(greedy(policy), feats, a.eval_rounds, s),
            "adaptive_inputs_blanked": evaluate(greedy(policy), blank, a.eval_rounds, s)["win_rate"],
        }
        print(s, ev[s], flush=True)
    rep["eval_per_style"] = ev
    rep["mean_win"] = {k: float(np.mean([ev[s][k]["win_rate"] if k == "adaptive" else ev[s][k] for s in STYLES]))
                       for k in ("non_adaptive_reflex", "adaptive", "adaptive_inputs_blanked")}
    rep["train_s"] = round(time.time() - t0, 1)
    Path(a.results).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"mean_win": rep["mean_win"], "opponent_acc": rep["opponent_model"]["test_accuracy_after_1s"],
                      "win_auc": rep["win_model"]["test_auc"]}, indent=1))


if __name__ == "__main__":
    main()
