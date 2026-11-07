"""Benchmark harness (Architecture §12.5): fixed scenario suite, metrics, regression gate.

    python scripts/bench.py --game gridcraft --mode brain --episodes 10 --out docs/results/x.json
    python scripts/bench.py --game gridcraft --mode laya --checkpoint checkpoints/laya-gc-v1 \
        --baseline docs/results/phase2-brain-only.json

Modes: `brain` (brain labels every decision; Phase 1/2 baseline) or `laya` (Laya decides,
escalates to the brain when unsure). Every decision is logged to logs/<run>/ (JSONL + SQLite).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nervous_system.runtime.log import ExperienceLog  # noqa: E402
from nervous_system.runtime.loop import RuntimeConfig  # noqa: E402
from nervous_system.wiring import build_runtime, game_spec, laya_model, scripted_brain  # noqa: E402

CORE = ("success_rate", "deaths_per_episode")
TOLERANCE = {"success_rate": 0.10, "deaths_per_episode": 0.10}


def run_suite(game: str, mode: str, scenarios: list[str], episodes: int, seed0: int = 1000,
              checkpoint: str | None = None, variant: str = "int8", temperature: float | None = None,
              event_driven: bool = True, label_fraction: float = 0.0, log_root: Path | None = None,
              run_id: str | None = None, quiet: bool = False, heartbeat_s: float | None = None,
              model=None) -> dict:
    spec = game_spec(game)
    run_id = run_id or time.strftime(f"{game}-{mode}-%Y%m%dT%H%M%S")
    log = ExperienceLog(log_root or ROOT / "logs" / run_id)
    if model is None and mode == "laya":
        model = laya_model(checkpoint, variant, temperature)
    out = {"run_id": run_id, "game": game, "mode": mode, "checkpoint": checkpoint, "variant": variant,
           "temperature": temperature, "event_driven": event_driven, "heartbeat_s": heartbeat_s, "episodes": episodes,
           "seed0": seed0, "scenarios": {}}
    t_wall = time.time()
    for sc in scenarios:
        rows = []
        for i in range(episodes):
            seed = seed0 + i
            brain = scripted_brain(spec)
            lcfg = spec.config.get("laya", {})
            cfg = RuntimeConfig(cadence_s=lcfg.get("cadence_ms", 100) / 1000,
                                brain_latency_s=spec.config.get("brain", {}).get("latency_s", 2.0),
                                heartbeat_s=heartbeat_s or spec.config.get("brain", {}).get("heartbeat_s", 10.0),
                                event_driven=event_driven, label_fraction=label_fraction, seed=seed)
            rt, adapter = build_runtime(spec, sc, seed, log, brain, model, run_id, cfg)
            w0 = time.time()
            st = rt.run()
            res = adapter.episode_result()
            hours = max(res["t"], 1e-6) / 3600
            res.update(decisions=st.decisions, acted=st.acted, escalations=st.escalations,
                       escalation_reasons=st.escalation_reasons, stale_replies=st.stale_replies,
                       brain_calls=brain.usage.calls, brain_tokens=brain.usage.tokens,
                       brain_calls_per_hour=brain.usage.calls / hours,
                       brain_tokens_per_hour=brain.usage.tokens / hours,
                       escalations_per_hour=st.escalations / hours,
                       laya_ms_mean=st.laya_ms / max(1, st.decisions),
                       cache_hits=st.cache_hits, wall_s=round(time.time() - w0, 2), seed=seed)
            rows.append(res)
            if not quiet:
                print(f"  {sc:12s} seed {seed} success {res['success']!s:5s} t {res['t']:6.1f} "
                      f"acted {st.acted}/{st.decisions} esc {st.escalations} brain {brain.usage.calls} "
                      f"wall {res['wall_s']}s", flush=True)
        out["scenarios"][sc] = summarise(rows)
        out["scenarios"][sc]["episodes"] = rows
    out["overall"] = overall(out["scenarios"])
    out["wall_s"] = round(time.time() - t_wall, 1)
    log.close()
    return out


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    dec = sum(r["decisions"] for r in rows)
    hours = sum(r["t"] for r in rows) / 3600
    return {
        "n": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "time_to_complete_s": statistics.fmean([r["t"] for r in rows if r["success"]] or [float("nan")]),
        "deaths_per_episode": sum(r["deaths"] for r in rows) / n,
        "decisions": dec,
        "acted_frac": sum(r["acted"] for r in rows) / max(1, dec),
        "escalations_per_hour": sum(r["escalations"] for r in rows) / max(hours, 1e-9),
        "brain_calls_per_hour": sum(r["brain_calls"] for r in rows) / max(hours, 1e-9),
        "brain_tokens_per_hour": sum(r["brain_tokens"] for r in rows) / max(hours, 1e-9),
        "laya_ms_mean": statistics.fmean(r["laya_ms_mean"] for r in rows),
        "cache_hit_frac": sum(r["cache_hits"] for r in rows) / max(1, dec),
    }


def overall(scen: dict) -> dict:
    keys = ["success_rate", "deaths_per_episode", "acted_frac", "escalations_per_hour",
            "brain_calls_per_hour", "brain_tokens_per_hour"]
    return {k: statistics.fmean(s[k] for s in scen.values()) for k in keys}


def compare(new: dict, base: dict) -> dict:
    """Promotion gate: no core metric may regress beyond tolerance in any scenario."""
    regressions = []
    for sc, s in new["scenarios"].items():
        b = base["scenarios"].get(sc)
        if not b:
            continue
        if s["success_rate"] < b["success_rate"] - TOLERANCE["success_rate"]:
            regressions.append(f"{sc}: success {s['success_rate']:.2f} < {b['success_rate']:.2f}")
        if s["deaths_per_episode"] > b["deaths_per_episode"] + TOLERANCE["deaths_per_episode"]:
            regressions.append(f"{sc}: deaths {s['deaths_per_episode']:.2f} > {b['deaths_per_episode']:.2f}")
    return {"regressions": regressions, "passed": not regressions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="gridcraft")
    ap.add_argument("--mode", choices=["brain", "laya"], default="brain")
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--checkpoint")
    ap.add_argument("--variant", default="int8", choices=["fp32", "bf16", "int8"])
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--poll-brain", action="store_true", help="brain reviews every heartbeat (not event-driven)")
    ap.add_argument("--label-fraction", type=float, default=0.0)
    ap.add_argument("--heartbeat", type=float)
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    a = ap.parse_args()
    spec = game_spec(a.game)
    res = run_suite(a.game, a.mode, a.scenarios or spec.scenarios, a.episodes, a.seed0, a.checkpoint,
                    a.variant, a.temperature, not a.poll_brain, a.label_fraction, heartbeat_s=a.heartbeat)
    if a.baseline:
        res["gate"] = compare(res, json.loads(Path(a.baseline).read_text()))
    for sc, s in res["scenarios"].items():
        print(f"{sc:12s} success {s['success_rate']:.2f} deaths {s['deaths_per_episode']:.2f} "
              f"acted {s['acted_frac']:.2f} esc/h {s['escalations_per_hour']:.0f} "
              f"brain tok/h {s['brain_tokens_per_hour']:.0f} laya {s['laya_ms_mean']:.1f}ms")
    print("overall", json.dumps(res["overall"]))
    if "gate" in res:
        print("gate", res["gate"])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
