"""Phase 0 benchmark: Laya single-decision and batched latency on local hardware.

Roadmap Phase 0 exit: "Laya answers a hand-written state text in < 50 ms on local hardware."

    python scripts/bench_laya.py                       # all checkpoints, fp32 + bf16 + int8
    python scripts/bench_laya.py --checkpoints laya-typed-decisions --variants fp32

Writes a JSON summary to docs/results/ and logs every decision as a protocol `Decision`
to logs/bench/<run>.jsonl (Architecture §12.1).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402

from nervous_system.laya import EscalationPolicy  # noqa: E402
from nervous_system.laya.model import CHECKPOINTS, LayaDecisionModel  # noqa: E402
from nervous_system.protocol import Decision  # noqa: E402

# Hand-written state texts in the Architecture §5 serializer format.
STATES = {
    "combat": (
        "GOAL engage player:Steve | CONSTRAINT stay_near base | PRIORITY survive>win\n"
        "SELF hp 12/20 hunger 18 armor diamond totem offhand yes cooldown ready\n"
        "OPP dist 3.1 hp ~8/20 weapon sword facing_me yes crit_threat high\n"
        "ENV ground flat lava_near no allies 0 enemies 1\n"
        "PRED opp_aim_lock 0.94 sprint_crit_incoming 0.81\n"
        "DRIVES fear 0.4 hunger 0.1\n"
        "WM last_actions engage,engage,engage last_outcome traded_hits",
        ["engage", "disengage", "heal", "reposition", "use_ability:totem_swap", "escalate"],
    ),
    "combat_critical": (
        "GOAL engage player:Steve | CONSTRAINT stay_near base | PRIORITY survive>win\n"
        "SELF hp 2/20 hunger 6 armor none totem offhand no cooldown charging\n"
        "OPP dist 1.2 hp 20/20 weapon netherite_sword facing_me yes crit_threat high\n"
        "ENV ground flat lava_near yes allies 0 enemies 3\n"
        "PRED opp_aim_lock 0.99 sprint_crit_incoming 0.95\n"
        "DRIVES fear 0.97 hunger 0.6\n"
        "WM last_actions engage,engage,engage last_outcome took_heavy_damage",
        ["engage", "disengage", "heal", "reposition", "use_ability:totem_swap", "escalate"],
    ),
    "gathering": (
        "GOAL gather:wood | CONSTRAINT stay_near base | PRIORITY survive>finish_task\n"
        "SELF hp 20/20 hunger 17 armor iron totem offhand no cooldown none\n"
        "OPP none\n"
        "ENV ground forest lava_near no allies 0 enemies 0 logs_held 41/64\n"
        "PRED none\n"
        "DRIVES fear 0.0 hunger 0.2\n"
        "WM last_actions continue,continue,continue last_outcome collected_log",
        ["continue", "return:base", "disengage", "heal", "explore", "escalate"],
    ),
}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def stats(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": pct(xs, 0.5),
            "p95": pct(xs, 0.95), "min": min(xs), "max": max(xs)}


def load(checkpoint: str, variant: str) -> LayaDecisionModel:
    os.environ.pop("LAYA_CPU_AMP", None)
    if variant == "bf16":
        os.environ["LAYA_CPU_AMP"] = "bf16"
    m = LayaDecisionModel(checkpoint, device="cpu")
    if variant == "int8":
        # Backbone only: quantising the head breaks nn.TransformerEncoderLayer's fast path.
        enc = m.agent.model.encoder
        m.agent.model.encoder = torch.ao.quantization.quantize_dynamic(
            enc, {torch.nn.Linear}, dtype=torch.qint8)
    return m


def run(checkpoint: str, variant: str, n: int, batch_sizes: list[int], log, policy) -> dict:
    t0 = time.perf_counter()
    m = load(checkpoint, variant)
    load_s = time.perf_counter() - t0
    out: dict = {"checkpoint": checkpoint, "variant": variant, "load_s": load_s, "states": {}}

    for name, (text, options) in STATES.items():
        for _ in range(3):  # warm-up
            m.decide(text, options)
        lat, dist = [], {}
        for _ in range(n):
            dist = m.decide(text, options)
            lat.append(m.last.latency_ms)
            act, choice, margin = policy.should_act(dist)
            log.write(Decision(
                bot_id="bench", game="bench", episode_id=f"phase0-{checkpoint}-{variant}", t=0.0,
                goal_id=name, state_text=text, options=options, choice=choice,
                distribution=dist, acted=act, margin=margin, latency_ms=m.last.latency_ms,
                model=f"{checkpoint}:{variant}").to_json() + "\n")
        act, choice, margin = policy.should_act(dist)
        out["states"][name] = {"latency_ms": stats(lat), "state_tokens": m.last.state_tokens,
                               "choice": choice, "acted": act, "margin": round(margin, 4),
                               "distribution": dist}

    text, options = STATES["combat"]
    out["batched"] = {}
    for bs in batch_sizes:
        items = [(text, options)] * bs
        m.decide_batch(items)  # warm-up
        times = []
        for _ in range(max(3, n // 4)):
            m.decide_batch(items)
            times.append(m.last.latency_ms)
        s = stats(times)
        out["batched"][bs] = {"batch_ms": s, "decisions_per_s": bs / (s["p50"] / 1000)}
    del m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINTS))
    ap.add_argument("--variants", nargs="+", default=["fp32", "bf16", "int8"],
                    choices=["fp32", "bf16", "int8"])
    ap.add_argument("-n", type=int, default=30, help="timed decisions per state")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--out", default=str(ROOT / "docs/results/phase0-laya-latency.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    run_id = time.strftime("%Y%m%dT%H%M%S")
    log_path = ROOT / "logs/bench" / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    policy = EscalationPolicy()
    report = {
        "run_id": run_id,
        "hardware": {"platform": platform.platform(), "processor": _cpu_name(),
                     "torch_threads": torch.get_num_threads(), "cuda": torch.cuda.is_available(),
                     "torch": torch.__version__},
        "laya_version": _laya_version(),
        "target_ms": 50,
        "results": [],
    }
    with open(log_path, "w") as log:
        for ck in args.checkpoints:
            for v in args.variants:
                print(f"== {ck} / {v}", flush=True)
                r = run(ck, v, args.n, args.batch_sizes, log, policy)
                for s, d in r["states"].items():
                    print(f"   {s:16s} p50 {d['latency_ms']['p50']:7.1f} ms  "
                          f"p95 {d['latency_ms']['p95']:7.1f} ms  -> {d['choice']} "
                          f"({'act' if d['acted'] else 'escalate'})", flush=True)
                for bs, d in r["batched"].items():
                    print(f"   batch {bs:3d}: {d['batch_ms']['p50']:7.1f} ms "
                          f"= {d['decisions_per_s']:6.1f} decisions/s", flush=True)
                report["results"].append(r)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out}; decisions logged to {log_path}")


def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() + f" x{os.cpu_count()}"
    except OSError:
        pass
    return platform.processor()


def _laya_version() -> str:
    from importlib.metadata import version
    return version("laya")


if __name__ == "__main__":
    main()
