"""Phase 6: N bots coordinate a base-building mission with one brain and batched Laya.

    python scripts/run_hive.py --episodes 10 --mode brain
    python scripts/run_hive.py --episodes 10 --mode laya --checkpoint checkpoints/laya-gc-v2
    python scripts/run_hive.py --episodes 10 --no-coordination      # ablation: brain ignores claims
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

from nervous_system.adapters.gridcraft import GridCraftAdapter, GridCraftPlaybook  # noqa: E402
from nervous_system.adapters.gridcraft.scenarios import build_base  # noqa: E402
from nervous_system.adapters.gridcraft.world import World  # noqa: E402
from nervous_system.brain.scripted import ScriptedBrain  # noqa: E402
from nervous_system.laya import EscalationPolicy  # noqa: E402
from nervous_system.laya.models import BrainLabelModel  # noqa: E402
from nervous_system.modules.forward import PhysicsForwardModel  # noqa: E402
from nervous_system.modules.salience import RuleSalience  # noqa: E402
from nervous_system.runtime.bus import Bus  # noqa: E402
from nervous_system.runtime.hive import HiveCoordinator, HiveRuntime  # noqa: E402
from nervous_system.runtime.log import ExperienceLog  # noqa: E402
from nervous_system.runtime.loop import BotRuntime, RuntimeConfig  # noqa: E402
from nervous_system.wiring import laya_model  # noqa: E402


def episode(seed: int, n_bots: int, mode: str, checkpoint: str | None, coordinate: bool, log, run_id: str) -> dict:
    scen = build_base(n_bots)
    w = World(seed=seed)
    scen.setup(w)
    ids = [a.id for a in w.agents.values() if a.kind == "player"]
    bus = Bus()
    coord = HiveCoordinator(bus)
    brain = ScriptedBrain(GridCraftPlaybook(coordinate=coordinate))      # ONE brain for the hive
    model = laya_model(checkpoint) if mode == "laya" else BrainLabelModel(brain.label)
    rts = []
    for bid in ids:
        a = GridCraftAdapter(scen, seed=seed, world=w, bot_id=bid, owns_world=False)
        a.connect("singleplayer")
        hdr = {"bot_id": bid, "game": "gridcraft", "episode_id": f"{run_id}-{seed}-{bid}"}
        log.start_episode(hdr["episode_id"], "gridcraft", {"scenario": "build_base", "seed": seed})
        rt = BotRuntime(a, brain, model, log, hdr, scen.mission,
                        RuntimeConfig(seed=seed, brain_latency_s=2.0), policy=EscalationPolicy(),
                        forward=PhysicsForwardModel(), salience=RuleSalience(dict(hdr)), bus=bus)
        rts.append(rt)
    adapters = {rt.hdr["bot_id"]: rt.a for rt in rts}

    def step_world():
        dangers = coord.active_markers(w.t, "danger")
        for a in adapters.values():
            a.markers[:] = [(m["x"], m["y"]) for m in dangers if m["by"] != a.bot_id]
        w.step()

    hive = HiveRuntime(rts, step_world, bus, model, coord,
                       positions=lambda rt: (rt.a.bot.x, rt.a.bot.y))
    t0 = time.time()
    hive.run(int(scen.time_limit_s * 20))
    st = w.structures["wall"]
    dec = sum(r.stats.decisions for r in rts)
    return {"seed": seed, "success": st["built"], "t": round(w.t, 1), "delivered": dict(st["have"]),
            "bots_dead": sum(not w.agents[i].alive for i in ids), "decisions": dec,
            "acted_frac": sum(r.stats.acted for r in rts) / max(1, dec),
            "escalations": sum(r.stats.escalations for r in rts),
            "brain_calls": brain.usage.calls, "brain_instances": 1, "bus_messages": bus.published,
            "hive_messages": coord.messages, "danger_markers": sum(1 for m in coord.markers if m["kind"] == "danger"),
            "laya_batches": hive.batcher.batches, "mean_batch": hive.batcher.batched_decisions / max(1, hive.batcher.batches),
            "max_batch": hive.batcher.max_batch, "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--bots", type=int, default=5)
    ap.add_argument("--mode", choices=["brain", "laya"], default="brain")
    ap.add_argument("--checkpoint")
    ap.add_argument("--no-coordination", action="store_true")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--out")
    a = ap.parse_args()
    run_id = time.strftime("hive-%Y%m%dT%H%M%S")
    log = ExperienceLog(ROOT / "logs" / run_id)
    rows = []
    for i in range(a.episodes):
        r = episode(a.seed0 + i, a.bots, a.mode, a.checkpoint, not a.no_coordination, log, run_id)
        rows.append(r)
        print(r, flush=True)
    log.close()
    ok = [r for r in rows if r["success"]]
    summary = {"mode": a.mode, "checkpoint": a.checkpoint, "bots": a.bots, "coordination": not a.no_coordination,
               "episodes": a.episodes, "success_rate": len(ok) / len(rows),
               "mean_time_s": statistics.fmean(r["t"] for r in ok) if ok else None,
               "mean_bots_dead": statistics.fmean(r["bots_dead"] for r in rows),
               "brain_instances": 1, "mean_brain_calls": statistics.fmean(r["brain_calls"] for r in rows),
               "mean_batch": statistics.fmean(r["mean_batch"] for r in rows),
               "acted_frac": statistics.fmean(r["acted_frac"] for r in rows), "episodes_detail": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
