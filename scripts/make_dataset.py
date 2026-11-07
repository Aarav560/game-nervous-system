"""Phase 2: the brain labels every decision in slow runs to bootstrap Laya's dataset (§12.2).

Rollouts use the scripted brain as labeller. With probability `--explore` the executed action is
random instead of the brain's, so the dataset also covers recovery states (DAgger-style); the
label is always the brain's choice for the state Laya would see.

Outputs laya.train JSONL rows `{state, questions, expected, meta}` split by episode into
train / calib (temperature fitting) / test files.

    python scripts/make_dataset.py --game gridcraft --episodes 12 --out data/gridcraft-v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nervous_system.laya.model import QUESTION_ID, build_question  # noqa: E402
from nervous_system.runtime.log import ExperienceLog  # noqa: E402
from nervous_system.runtime.loop import RuntimeConfig  # noqa: E402
from nervous_system.wiring import build_runtime, game_spec, scripted_brain  # noqa: E402


class LabelRecorder:
    """DecisionModel that records (state, options, brain label) and executes eps-random actions."""

    name = "brain-label+explore"

    def __init__(self, label, explore: float, rng: random.Random):
        self.label, self.explore, self.rng = label, explore, rng
        self.rows: list[tuple[str, list[str], str]] = []
        self.last_latency_ms = 0.0

    def decide(self, text: str, options: list[str]) -> dict[str, float]:
        y = self.label(text, options)
        self.rows.append((text, list(options), y))
        act = y
        if self.rng.random() < self.explore:
            act = self.rng.choice([o for o in options if o != "escalate"])
        rest = 0.01 / max(1, len(options) - 1)
        return {o: (0.99 if o == act else rest) for o in options}

    def decide_batch(self, items):
        return [self.decide(s, o) for s, o in items]


def collect(game: str, scenarios: list[str], episodes: int, seed0: int, explore: float,
            log_root: Path) -> list[dict]:
    spec = game_spec(game)
    log = ExperienceLog(log_root)
    out = []
    for sc in scenarios:
        for i in range(episodes):
            seed = seed0 + i
            rng = random.Random(seed)
            brain = scripted_brain(spec)
            rec = LabelRecorder(brain.label, explore if i % 3 else 0.0, rng)   # 1/3 clean rollouts
            cfg = RuntimeConfig(seed=seed, brain_latency_s=spec.config["brain"].get("latency_s", 2.0))
            rt, adapter = build_runtime(spec, sc, seed, log, brain, rec, "dataset", cfg)
            rt.run()
            for text, opts, y in rec.rows:
                out.append({"state": text, "options": opts, "label": y,
                            "meta": {"game": game, "scenario": sc, "seed": seed}})
    log.close()
    return out


def to_rows(samples: list[dict]) -> list[dict]:
    return [{"state": s["state"], "questions": build_question(s["options"]),
             "expected": {QUESTION_ID: s["label"]}, "meta": s["meta"]} for s in samples]


def dedupe_and_balance(samples: list[dict], cap_per_label: int, rng: random.Random) -> list[dict]:
    seen, uniq = set(), []
    for s in samples:
        k = (s["state"], tuple(s["options"]))
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    rng.shuffle(uniq)
    by = Counter()
    kept = []
    for s in uniq:
        key = (tuple(s["options"]), s["label"])
        if by[key] < cap_per_label:
            by[key] += 1
            kept.append(s)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="gridcraft")
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--explore", type=float, default=0.2)
    ap.add_argument("--cap-per-label", type=int, default=1200)
    ap.add_argument("--max-total", type=int, default=9000)
    ap.add_argument("--out", default=str(ROOT / "data/gridcraft-v1"))
    a = ap.parse_args()
    spec = game_spec(a.game)
    rng = random.Random(a.seed0)
    raw = collect(a.game, a.scenarios or spec.scenarios, a.episodes, a.seed0, a.explore,
                  ROOT / "logs" / f"dataset-{a.game}")
    kept = dedupe_and_balance(raw, a.cap_per_label, rng)[: a.max_total]
    # split by seed so near-duplicate states from one episode never straddle splits
    seeds = sorted({s["meta"]["seed"] for s in kept})
    test_seeds = set(seeds[-max(1, len(seeds) // 6):])
    calib_seeds = set(seeds[-max(2, len(seeds) // 3):]) - test_seeds
    split = {"train": [], "calib": [], "test": []}
    for s in kept:
        sd = s["meta"]["seed"]
        split["test" if sd in test_seeds else "calib" if sd in calib_seeds else "train"].append(s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, ss in split.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in to_rows(ss):
                f.write(json.dumps(r) + "\n")
    stats = {"raw_decisions": len(raw), "unique_balanced": len(kept),
             **{f"n_{k}": len(v) for k, v in split.items()},
             "labels": dict(Counter(s["label"] for s in kept)),
             "scenarios": dict(Counter(s["meta"]["scenario"] for s in kept)),
             "test_seeds": sorted(test_seeds), "calib_seeds": sorted(calib_seeds)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
