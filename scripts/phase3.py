"""Phase 3 exit: brain tokens/hour drop >= 50% vs Phase 2 at equal benchmark scores.

Phase 2 wiring: the brain is polled with a world summary every `--p2-heartbeat` seconds
(salient events do not wake it; it still answers escalations).
Phase 3 wiring: event-driven brain (salience filter + surprise events + goal events), heartbeat
only as a backup. Same Laya checkpoint, same seeds.

    python scripts/phase3.py --checkpoint checkpoints/laya-gc-v1 --episodes 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402
from nervous_system.wiring import laya_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--p2-heartbeat", type=float, default=2.0)
    ap.add_argument("--p3-heartbeat", type=float, default=30.0)
    ap.add_argument("--p2", help="reuse an existing Phase 2-wiring bench JSON")
    ap.add_argument("--out", default=str(ROOT / "docs/results/phase3-event-driven.json"))
    a = ap.parse_args()
    model = laya_model(a.checkpoint)                       # one cache shared by both runs
    if a.p2:
        p2 = json.loads(Path(a.p2).read_text())
    else:
        p2 = bench.run_suite("gridcraft", "laya", bench.game_spec("gridcraft").scenarios, a.episodes,
                             checkpoint=a.checkpoint, event_driven=False, heartbeat_s=a.p2_heartbeat,
                             label_fraction=0.05, model=model, run_id="phase3-p2-polled")
        Path(a.out).with_name("phase2-laya.json").write_text(json.dumps(p2, indent=1, default=str) + "\n")
    p3 = bench.run_suite("gridcraft", "laya", bench.game_spec("gridcraft").scenarios, a.episodes,
                         checkpoint=a.checkpoint, event_driven=True, heartbeat_s=a.p3_heartbeat,
                         label_fraction=0.05, model=model, run_id="phase3-p3-event")
    Path(a.out).with_name("phase3-laya-event-driven.json").write_text(json.dumps(p3, indent=1, default=str) + "\n")
    t2, t3 = p2["overall"]["brain_tokens_per_hour"], p3["overall"]["brain_tokens_per_hour"]
    gate = bench.compare(p3, p2)
    res = {"p2_polled": {k: p2[k] for k in ("overall", "heartbeat_s", "event_driven")},
           "p3_event_driven": {k: p3[k] for k in ("overall", "heartbeat_s", "event_driven")},
           "per_scenario": {sc: {"p2_success": p2["scenarios"][sc]["success_rate"],
                                 "p3_success": p3["scenarios"][sc]["success_rate"],
                                 "p2_tokens_per_hour": p2["scenarios"][sc]["brain_tokens_per_hour"],
                                 "p3_tokens_per_hour": p3["scenarios"][sc]["brain_tokens_per_hour"]}
                            for sc in p3["scenarios"]},
           "token_drop": 1 - t3 / t2 if t2 else None, "scores_equal_gate": gate,
           "exit_criterion_passed": bool(t2) and (1 - t3 / t2) >= 0.5 and gate["passed"]}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("token_drop", "scores_equal_gate", "exit_criterion_passed")}, indent=1))


if __name__ == "__main__":
    main()
