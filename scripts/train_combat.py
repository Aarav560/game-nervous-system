"""Phase 4: train the GridCraft combat reflex (Architecture §12.4 recipe).

1. Behaviour cloning on the scripted engage reflex. Stand-in for "record own PvP play":
   there is no human play data in this environment.
2. PPO against the scripted practice bot.
3. League training: opponents are drawn from {scripted bot, past snapshots of the main policy,
   an exploiter trained only to beat the current main}.
4. Evaluation: N rounds of the reflex alone vs the scripted practice bot (exit: >= 70% wins).

    python scripts/train_combat.py --out checkpoints/combat-reflex.pt
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

from nervous_system.adapters.gridcraft.combat import (  # noqa: E402
    HEADS, CombatPolicy, apply_action, expert_action, observe, save_policy)
from nervous_system.adapters.gridcraft.scenarios import duel  # noqa: E402
from nervous_system.adapters.gridcraft.world import World, scripted_pvp_ai  # noqa: E402

torch.set_num_threads(1)


# ---------------------------------------------------------------------------------- envs
class Duel:
    """One duel; `bot` is driven by the learner, `opp` by a scripted AI or a policy."""

    def __init__(self, seed: int, opponent, max_s: float):
        self.w = World(seed=seed)
        duel(opponent_ai=None).setup(self.w)
        self.me, self.opp = self.w.agents["bot"], self.w.agents["opp"]
        # random start offsets so the policy doesn't overfit one opening
        r = random.Random(seed)
        self.me.x += r.uniform(-4, 4); self.me.y += r.uniform(-6, 6)
        self.opp.x += r.uniform(-4, 4); self.opp.y += r.uniform(-6, 6)
        self.opponent = opponent           # "scripted" | "mirror" | CombatPolicy
        if opponent == "scripted":
            self.opp.ai = scripted_pvp_ai("bot", rng=random.Random(seed + 7))
        else:
            self.opp.ai = None
        self.max_ticks = int(max_s * 20)
        self.done = False
        self.won = False
        self._dd = self._dt = 0.0
        self.hist: list = []                # (dealt, taken) per tick for aux labels

    def obs(self) -> np.ndarray:
        return observe(self.w, self.me, self.opp)

    def step(self, a: dict, opp_a: dict | None) -> float:
        if self.opponent == "mirror":           # the opponent runs our own expert logic
            opp_a = expert_action(self.w, self.opp, self.me)
        apply_action(self.w, self.me, self.opp, a)
        if opp_a is not None:
            apply_action(self.w, self.opp, self.me, opp_a)
        if self.opp.offhand is None and self.opp.count("totem"):
            self.opp.offhand = "totem"
        if self.me.offhand is None and self.me.count("totem"):
            self.me.offhand = "totem"
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


def aux_labels(hist: list, horizon: int = 10) -> np.ndarray:
    hit = np.array([h[0] for h in hist], dtype=np.float32)
    got = np.array([h[1] for h in hist], dtype=np.float32)
    out = np.zeros((len(hist), 2), dtype=np.float32)
    for i in range(len(hist)):
        out[i, 0] = hit[i:i + horizon].max(initial=0)
        out[i, 1] = got[i:i + horizon].max(initial=0)
    return out


# ---------------------------------------------------------------------------------- BC
def behaviour_cloning(policy: CombatPolicy, episodes: int, seed0: int) -> dict:
    X, Y, AUX = [], {k: [] for k in HEADS}, []
    for e in range(episodes):
        env = Duel(seed0 + e, "scripted", 90.0)
        ep_x = []
        while not env.done:
            o = env.obs()
            a = expert_action(env.w, env.me, env.opp)
            ep_x.append(o)
            for k in HEADS:
                Y[k].append(a[k])
            env.step(a, None)
        X += ep_x
        AUX.append(aux_labels(env.hist))
    X = torch.tensor(np.stack(X))
    Yt = {k: torch.tensor(v) for k, v in Y.items()}
    A = torch.tensor(np.concatenate(AUX))
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    n = len(X)
    for ep in range(8):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            logits, _, aux = policy(X[idx])
            loss = sum(ce(logits[k], Yt[k][idx]) for k in HEADS) + 0.5 * bce(aux, A[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
    with torch.no_grad():
        logits, _, _ = policy(X)
        acc = {k: float((logits[k].argmax(-1) == Yt[k]).float().mean()) for k in HEADS}
    return {"samples": n, "final_loss": tot / n, "train_accuracy": acc}


# ---------------------------------------------------------------------------------- PPO
def rollout(policy: CombatPolicy, opponents: list, n_envs: int, seed: int, max_s: float):
    rng = random.Random(seed)
    envs = [Duel(seed * 1000 + i, rng.choice(opponents), max_s) for i in range(n_envs)]
    buf = [dict(obs=[], act=[], logp=[], val=[], rew=[]) for _ in envs]
    while not all(e.done for e in envs):
        live = [i for i, e in enumerate(envs) if not e.done]
        obs = np.stack([envs[i].obs() for i in live])
        acts, info = policy.act(obs)
        # opponents that are policies act from their own perspective
        opp_acts: dict[int, dict] = {}
        by_pol: dict[int, list[int]] = {}
        for j, i in enumerate(live):
            if not isinstance(envs[i].opponent, str):
                by_pol.setdefault(id(envs[i].opponent), []).append(i)
        for _, idxs in by_pol.items():
            pol = envs[idxs[0]].opponent
            oo = np.stack([observe(envs[i].w, envs[i].opp, envs[i].me) for i in idxs])
            oa, _ = pol.act(oo)
            for i, a in zip(idxs, oa):
                opp_acts[i] = a
        for j, i in enumerate(live):
            r = envs[i].step(acts[j], opp_acts.get(i))
            b = buf[i]
            b["obs"].append(obs[j]); b["act"].append(acts[j]); b["logp"].append(float(info["logp"][j]))
            b["val"].append(float(info["value"][j])); b["rew"].append(r)
    for b, e in zip(buf, envs):
        b["aux"] = aux_labels(e.hist)
    return buf, envs


def gae(rew, val, gamma=0.995, lam=0.95):
    adv = np.zeros(len(rew), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rew))):
        nv = val[t + 1] if t + 1 < len(val) else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.array(val, dtype=np.float32)


def ppo_update(policy, opt, buf, epochs=4, mb=4096, clip=0.2, ent=0.01, bc_policy=None, bc_coef=0.0):
    obs, acts, logp, adv, ret, aux = [], {k: [] for k in HEADS}, [], [], [], []
    for b in buf:
        a, r = gae(b["rew"], b["val"])
        obs += b["obs"]; logp += b["logp"]; adv.append(a); ret.append(r); aux.append(b["aux"])
        for k in HEADS:
            acts[k] += [x[k] for x in b["act"]]
    O = torch.tensor(np.stack(obs)); LP = torch.tensor(logp); AD = torch.tensor(np.concatenate(adv))
    RT = torch.tensor(np.concatenate(ret)); AX = torch.tensor(np.concatenate(aux))
    AC = {k: torch.tensor(v) for k, v in acts.items()}
    AD = (AD - AD.mean()) / (AD.std() + 1e-8)
    n = len(O)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            logits, v, auxl = policy(O[idx])
            nlp = torch.zeros(len(idx)); entropy = 0.0
            for k in HEADS:
                d = torch.distributions.Categorical(logits=logits[k])
                nlp = nlp + d.log_prob(AC[k][idx]); entropy = entropy + d.entropy().mean()
            ratio = torch.exp(nlp - LP[idx])
            pl = -torch.min(ratio * AD[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * AD[idx]).mean()
            vl = 0.5 * ((v - RT[idx]) ** 2).mean()
            loss = pl + 0.5 * vl - ent * entropy + 0.3 * bce(auxl, AX[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
    return n


# ---------------------------------------------------------------------------------- eval
def evaluate(policy: CombatPolicy, rounds: int, seed0: int, opponent="scripted", greedy=True) -> dict:
    wins = deaths = 0
    t = []
    for i in range(rounds):
        env = Duel(seed0 + i, opponent, 180.0)
        while not env.done:
            acts, _ = policy.act(env.obs(), greedy=greedy)
            oa = None
            if not isinstance(opponent, str):
                oa = opponent.act(observe(env.w, env.opp, env.me), greedy=greedy)[0][0]
            env.step(acts[0], oa)
        wins += env.won
        deaths += int(not env.me.alive)
        t.append(env.w.t)
    return {"rounds": rounds, "win_rate": wins / rounds, "loss_rate": deaths / rounds,
            "mean_round_s": float(np.mean(t))}


def scripted_engage_baseline(rounds: int, seed0: int, opponent: str = "scripted") -> dict:
    """The scripted engage reflex (+ heal rule) alone vs the practice bot, for comparison."""
    wins = 0
    for i in range(rounds):
        env = Duel(seed0 + i, opponent, 180.0)
        while not env.done:
            env.step(expert_action(env.w, env.me, env.opp), None)
        wins += env.won
    return {"rounds": rounds, "win_rate": wins / rounds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "checkpoints/combat-reflex.pt"))
    ap.add_argument("--bc-episodes", type=int, default=60)
    ap.add_argument("--ppo-iters", type=int, default=60)
    ap.add_argument("--league-iters", type=int, default=40)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--eval-rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=str(ROOT / "docs/results/phase4-combat-reflex.json"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rep: dict = {"recipe": "BC (scripted engage) -> PPO vs scripted -> league (snapshots + exploiter)"}
    t0 = time.time()
    EVAL_SEED = 900_000
    rep["scripted_engage_baseline"] = scripted_engage_baseline(a.eval_rounds, EVAL_SEED)
    rep["scripted_engage_baseline_mirror"] = scripted_engage_baseline(a.eval_rounds, EVAL_SEED, "mirror")
    print("scripted engage baseline", rep["scripted_engage_baseline"], rep["scripted_engage_baseline_mirror"], flush=True)

    policy = CombatPolicy()
    rep["bc"] = behaviour_cloning(policy, a.bc_episodes, 10_000)
    rep["after_bc"] = evaluate(policy, a.eval_rounds // 2, EVAL_SEED)
    rep["after_bc_vs_mirror"] = evaluate(policy, a.eval_rounds // 2, EVAL_SEED, opponent="mirror")
    print("after BC", rep["bc"]["train_accuracy"], rep["after_bc"], flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    curve = []
    for it in range(a.ppo_iters):
        buf, envs = rollout(policy, ["scripted", "mirror", "mirror"], a.envs, 20_000 + it, 60.0)
        ppo_update(policy, opt, buf)
        wr = sum(e.won for e in envs) / len(envs)
        curve.append(wr)
        if it % 5 == 4:
            print(f"ppo {it + 1}: train win {np.mean(curve[-5:]):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    rep["ppo_curve"] = curve
    rep["after_ppo"] = evaluate(policy, a.eval_rounds // 2, EVAL_SEED)
    rep["after_ppo_vs_mirror"] = evaluate(policy, a.eval_rounds // 2, EVAL_SEED, opponent="mirror")
    print("after PPO", rep["after_ppo"], flush=True)

    # league: main vs {scripted, snapshots, exploiter}
    snapshots = [copy.deepcopy(policy).eval()]
    exploiter = copy.deepcopy(policy)
    ex_opt = torch.optim.Adam(exploiter.parameters(), lr=3e-4)
    league_curve = []
    for it in range(a.league_iters):
        # exploiter step: learns to beat the frozen current main
        frozen = copy.deepcopy(policy).eval()
        ebuf, _ = rollout(exploiter, [frozen], a.envs // 2, 40_000 + it, 60.0)
        ppo_update(exploiter, ex_opt, ebuf, epochs=2)
        # main step
        opps = ["scripted", "mirror", "mirror", random.choice(snapshots), copy.deepcopy(exploiter).eval()]
        buf, envs = rollout(policy, opps, a.envs, 30_000 + it, 60.0)
        ppo_update(policy, opt, buf)
        league_curve.append(sum(e.won for e in envs if e.opponent == "mirror") /
                            max(1, sum(e.opponent == "mirror" for e in envs)))
        if it % 10 == 9:
            snapshots.append(copy.deepcopy(policy).eval())
            print(f"league {it + 1}: win vs mirror bot {np.mean(league_curve[-10:]):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    rep["league_curve_vs_mirror"] = league_curve
    rep["final"] = evaluate(policy, a.eval_rounds, EVAL_SEED)
    rep["final_vs_mirror"] = evaluate(policy, a.eval_rounds, EVAL_SEED, opponent="mirror")
    rep["final_vs_exploiter"] = evaluate(policy, 50, EVAL_SEED + 5000, opponent=exploiter.eval())
    rep["train_s"] = round(time.time() - t0, 1)
    rep["exit_criterion"] = {"target_win_rate": 0.70, "win_rate": rep["final"]["win_rate"],
                             "passed": rep["final"]["win_rate"] >= 0.70}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_policy(policy, a.out, {"final": rep["final"]})
    Path(a.results).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("scripted_engage_baseline", "scripted_engage_baseline_mirror", "after_bc",
                                          "after_bc_vs_mirror", "after_ppo", "after_ppo_vs_mirror", "final", "final_vs_mirror",
                                          "final_vs_exploiter", "exit_criterion", "train_s")}, indent=1))


if __name__ == "__main__":
    main()
