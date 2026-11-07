"""Phase 5: automated sleep cycle (Architecture §7.9, §12.2).

review logs -> brain labels (escalations, spot checks + a DAgger sample of Laya's own decisions)
-> episodic memory + skill library -> fine-tune from the champion -> calibrate -> benchmark
-> promote only if no benchmark regression.

    python scripts/sleep.py --cycles 3 --champion checkpoints/laya-gc-v1 \
        --review logs/<last laya bench run> --baseline docs/results/phase2-brain-only.json
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nervous_system.laya.model import QUESTION_ID, build_question  # noqa: E402
from nervous_system.modules.memory import EpisodicMemory, SkillLibrary  # noqa: E402
from nervous_system.wiring import game_spec, scripted_brain  # noqa: E402

PY = sys.executable


def review(log_dir: Path, game: str, dagger: int, rng: random.Random, memory: EpisodicMemory,
           skills: SkillLibrary) -> tuple[list[dict], dict]:
    """Brain-label escalations/spot checks and a sample of acted decisions from one run's logs."""
    brain = scripted_brain(game_spec(game))
    db = sqlite3.connect(log_dir / "index.sqlite")
    rows, stats = [], {"corrections": 0, "dagger": 0, "disagree": 0}
    episodes = [r[0] for r in db.execute("SELECT episode_id FROM episodes")]
    acted_pool = []
    for ep in episodes:
        path = log_dir / f"{ep}.jsonl"
        if not path.exists():
            continue
        goals = {}
        for line in open(path):
            m = json.loads(line)
            msg = m["msg"]
            if m["type"] == "event" and msg["kind"] == "correction":
                d = msg["data"]
                rows.append({"state": d["state_text"], "options": d["options"], "label": d["label"], "ep": ep})
                stats["corrections"] += 1
                stats["disagree"] += int(not d["agree"])
            elif m["type"] == "decision" and msg["acted"]:
                acted_pool.append((msg["state_text"], msg["options"], ep))
            elif m["type"] == "goal":
                goals[msg["goal_id"]] = msg["intent"]
            elif m["type"] == "event" and msg["description"].startswith("goal "):
                st = msg["data"].get("status")
                gi = goals.get(msg["data"].get("goal_id"))
                if gi and st in ("succeeded", "failed"):
                    skills.record(game, gi, st == "succeeded")
        meta = db.execute("SELECT meta FROM episodes WHERE episode_id=?", (ep,)).fetchone()
        sc = json.loads(meta[0]).get("scenario", "?") if meta else "?"
        n_esc = db.execute("SELECT COUNT(*) FROM escalations WHERE episode_id=?", (ep,)).fetchone()[0]
        n_dec = db.execute("SELECT COUNT(*) FROM decisions WHERE episode_id=?", (ep,)).fetchone()[0]
        died = db.execute("SELECT COUNT(*) FROM events WHERE episode_id=? AND description='died'", (ep,)).fetchone()[0]
        memory.add(ep, game, sc, f"{n_dec} decisions, {n_esc} escalations", "death" if died else "survived")
    uniq = list({(s, tuple(o)): (s, o, ep) for s, o, ep in acted_pool}.values())
    for s, o, ep in rng.sample(uniq, min(dagger, len(uniq))):
        rows.append({"state": s, "options": list(o), "label": brain.label(s, list(o)), "ep": ep})
        stats["dagger"] += 1
    return rows, stats


def write_dataset(rows: list[dict], replay: list[dict], out: Path, rng: random.Random,
                  anchor_calib: list[dict] = (), anchor_test: list[dict] = (), cap: int = 300) -> dict:
    """New brain labels (capped per option set x label so one label cannot flood a night),
    plus replay; calib/test are pooled with a fixed anchor slice of the original held-out sets so a
    temperature is never fitted on a few dozen skewed examples (attempt 1 failure)."""
    out.mkdir(parents=True, exist_ok=True)
    seen, uniq, per = set(), [], {}
    rng.shuffle(rows)
    for r in rows:
        k = (r["state"], tuple(r["options"]))
        key = (tuple(r["options"]), r["label"])
        if k not in seen and per.get(key, 0) < cap:
            seen.add(k)
            per[key] = per.get(key, 0) + 1
            uniq.append(r)
    eps = sorted({r["ep"] for r in uniq})
    rng.shuffle(eps)
    test_eps = set(eps[: max(1, len(eps) // 7)])
    calib_eps = set(eps[max(1, len(eps) // 7): max(2, 2 * len(eps) // 7)])
    to_row = lambda r: {"state": r["state"], "questions": build_question(r["options"]),  # noqa: E731
                        "expected": {QUESTION_ID: r["label"]}}
    split = {"train": [], "calib": [], "test": []}
    for r in uniq:
        split["test" if r["ep"] in test_eps else "calib" if r["ep"] in calib_eps else "train"].append(to_row(r))
    split["train"] += replay
    split["calib"] += list(anchor_calib)
    split["test"] += list(anchor_test)
    rng.shuffle(split["train"])
    for k, v in split.items():
        (out / f"{k}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in v))
    return {k: len(v) for k, v in split.items()} | {"new_unique": len(uniq), "replay": len(replay),
                                                    "anchor_calib": len(anchor_calib), "anchor_test": len(anchor_test)}


def stratified_replay(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Half combat rows, half everything else, so combat skills are not forgotten."""
    combat = [r for r in pool if "engage" in r["questions"]["action"]["criteria"]]
    other = [r for r in pool if "engage" not in r["questions"]["action"]["criteria"]]
    return rng.sample(combat, min(n // 2, len(combat))) + rng.sample(other, min(n - n // 2, len(other)))


def run(cmd: list[str], log: Path) -> None:
    with open(log, "w") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="gridcraft")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--review", required=True, help="log dir of the champion's last benchmark run")
    ap.add_argument("--champion-bench", required=True, help="bench JSON of the champion")
    ap.add_argument("--baseline", required=True, help="brain-only bench JSON (regression gate)")
    ap.add_argument("--replay", default=str(ROOT / "data/gridcraft-v1/train.jsonl"))
    ap.add_argument("--replay-n", type=int, default=1200)
    ap.add_argument("--anchor", default=str(ROOT / "data/gridcraft-v1"), help="original calib/test to pool with")
    ap.add_argument("--anchor-n", type=int, default=300)
    ap.add_argument("--dagger", type=int, default=1200)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs/results/phase5-sleep-cycles.json"))
    a = ap.parse_args()
    rng = random.Random(7)
    memory = EpisodicMemory(ROOT / "memory/episodic.sqlite")
    skills = SkillLibrary(ROOT / "memory/skills.json")
    champion, review_dir = a.champion, Path(a.review)
    champ_bench = json.loads(Path(a.champion_bench).read_text())
    replay_pool = [json.loads(l) for l in open(a.replay)]
    arng = random.Random(11)
    anchor_calib = arng.sample([json.loads(l) for l in open(Path(a.anchor) / "calib.jsonl")], a.anchor_n)
    anchor_test = arng.sample([json.loads(l) for l in open(Path(a.anchor) / "test.jsonl")], a.anchor_n)
    history = [{"cycle": 0, "checkpoint": champion,
                "escalations_per_hour": champ_bench["overall"]["escalations_per_hour"],
                "acted_frac": champ_bench["overall"]["acted_frac"],
                "success_rate": champ_bench["overall"]["success_rate"]}]
    for c in range(1, a.cycles + 1):
        t0 = time.time()
        rows, rstats = review(review_dir, a.game, a.dagger, rng, memory, skills)
        skills.save()
        replay = stratified_replay(replay_pool, a.replay_n, rng)
        data = ROOT / f"data/sleep-{c}"
        dstats = write_dataset(rows, replay, data, rng, anchor_calib, anchor_test)
        replay_pool += [json.loads(l) for l in open(data / "train.jsonl")][: a.dagger]
        ckpt = str(ROOT / f"checkpoints/laya-gc-sleep{c}")
        run([PY, "scripts/train_laya.py", "--init", champion, "--data", str(data), "--out", ckpt,
             "--epochs", "1", "--threads", str(a.threads)], ROOT / f"logs/sleep-{c}-train.log")
        cal = json.loads((Path(ckpt) / "gns_calibration.json").read_text())
        bench_out = ROOT / f"docs/results/phase5-sleep{c}-bench.json"
        run([PY, "scripts/bench.py", "--game", a.game, "--mode", "laya", "--checkpoint", ckpt,
             "--episodes", str(a.episodes), "--label-fraction", "0.05", "--baseline", a.baseline,
             "--out", str(bench_out)], ROOT / f"logs/sleep-{c}-bench.log")
        b = json.loads(bench_out.read_text())
        # promote only against the CURRENT CHAMPION (attempt 1 compared with the previous cycle)
        vs_champ = __import__("bench").compare(b, champ_bench)
        champ_esc = champ_bench["overall"]["escalations_per_hour"]
        promote = (b["gate"]["passed"] and vs_champ["passed"]
                   and b["overall"]["escalations_per_hour"] <= champ_esc * 1.05)
        entry = {"cycle": c, "checkpoint": ckpt, "review": rstats, "dataset": dstats,
                 "temperature": cal["temperature"], "test_calibrated": cal["metrics"]["test_calibrated"],
                 "escalations_per_hour": b["overall"]["escalations_per_hour"],
                 "acted_frac": b["overall"]["acted_frac"], "success_rate": b["overall"]["success_rate"],
                 "gate": b["gate"], "gate_vs_champion": vs_champ, "champion_escalations_per_hour": champ_esc,
                 "promoted": promote, "wall_s": round(time.time() - t0)}
        history.append(entry)
        print(json.dumps(entry, indent=1, default=str), flush=True)
        if promote:
            champion, champ_bench = ckpt, b
            (ROOT / "checkpoints/champion.json").write_text(json.dumps({"checkpoint": ckpt, "cycle": c}))
        review_dir = ROOT / "logs" / b["run_id"]           # next night reviews today's play
        Path(a.out).write_text(json.dumps({"history": history, "skills": skills.summary(a.game)}, indent=1,
                                          default=str) + "\n")
    esc = [h["escalations_per_hour"] for h in history]
    champ = [h["escalations_per_hour"] for h in history if h["cycle"] == 0 or h.get("promoted")]
    print("escalations/hour by cycle:", [round(x) for x in esc], "champions:", [round(x) for x in champ])


if __name__ == "__main__":
    main()
