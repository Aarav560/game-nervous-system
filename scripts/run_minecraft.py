"""Phase 1 end-to-end run on a real Minecraft protocol server (Roadmap Phase 1 exit).

Starts a local flying-squid server (own_server, 127.0.0.1) with a planted forest and the
Mineflayer bridge, then:
  1. play-rules check: connecting to an unlisted public server must be refused;
  2. kill-switch drill: start a reflex, trip the kill switch, verify inputs released + disconnect;
  3. the mission "gather 64 logs and return to base" with goals flowing down and events up.

    python scripts/run_minecraft.py --out docs/results/phase1-minecraft-gather-logs.json
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nervous_system.adapters.minecraft import MinecraftAdapter, MinecraftPlaybook  # noqa: E402
from nervous_system.brain.scripted import ScriptedBrain  # noqa: E402
from nervous_system.laya import EscalationPolicy  # noqa: E402
from nervous_system.laya.models import BrainLabelModel  # noqa: E402
from nervous_system.modules import KillSwitch  # noqa: E402
from nervous_system.modules.forward import PhysicsForwardModel  # noqa: E402
from nervous_system.modules.salience import RuleSalience  # noqa: E402
from nervous_system.runtime.log import ExperienceLog  # noqa: E402
from nervous_system.runtime.loop import BotRuntime, RuntimeConfig  # noqa: E402

BRIDGE = ROOT / "bridge/minecraft"


def start(cmd: list[str], env: dict | None = None, timeout: float = 60.0) -> tuple[subprocess.Popen, dict]:
    p = subprocess.Popen(cmd, cwd=BRIDGE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})})
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                raise RuntimeError(f"{cmd} exited with {p.returncode}")
            continue
        line = line.strip()
        if line.startswith("{") and '"ready"' in line:
            return p, json.loads(line)
    p.kill()
    raise TimeoutError(f"{cmd} not ready")


def stop(p: subprocess.Popen | None) -> None:
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def bridge(mc_port: int, port: int, base: dict, token: str) -> subprocess.Popen:
    p, _ = start(["node", "bridge.js", "--mc-port", str(mc_port), "--port", str(port),
                  "--base", f"{base['x']},{base['y']},{base['z']}"], env={"GNS_BRIDGE_TOKEN": token})
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-port", type=int, default=25599)
    ap.add_argument("--bridge-port", type=int, default=47100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trees", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "docs/results/phase1-minecraft-gather-logs.json"))
    a = ap.parse_args()

    run_id = time.strftime("minecraft-%Y%m%dT%H%M%S")
    address = f"127.0.0.1:{a.mc_port}"
    token = secrets.token_hex(16)
    report: dict = {"run_id": run_id, "server": "flying-squid (local, offline, 127.0.0.1)",
                    "brain": "scripted stand-in (no LLM key in this environment)", "laya": "not used in Phase 1"}
    server = br = None
    try:
        # The kill switch exists (and its triggers are armed) before anything connects.
        kill = KillSwitch(sentinel=str(ROOT / "KILL"))
        kill.install_signal_handlers()

        server, info = start(["node", "server.js", "--port", str(a.mc_port), "--seed", str(a.seed),
                              "--trees", str(a.trees)])
        report["world"] = info
        base = info["base"]

        # 1. play rules: an unlisted public server is refused before any socket is opened
        try:
            MinecraftAdapter(bridge_port=a.bridge_port, token=token).connect("permitted_server", "mc.example.net:25565")
            report["play_rules_refusal"] = "FAILED: connection was allowed"
        except PermissionError as e:
            report["play_rules_refusal"] = f"refused: {e}"

        # 2. kill-switch drill on the live bridge
        br = bridge(a.mc_port, a.bridge_port, base, token)
        drill = MinecraftAdapter(bridge_port=a.bridge_port, token=token)
        drill.connect("own_server", address)
        drill.client.send({"cmd": "reflex", "name": "explore", "params": {}})
        time.sleep(1.5)
        moving = drill.read_state()["controls"]
        dk = KillSwitch()
        acks = {}

        def release():
            drill.release_all()
            acks["release"] = drill.client.wait_ack("release_all", timeout=3.0)

        dk.on_trip(release)
        dk.on_trip(drill.disconnect)
        t0 = time.perf_counter()
        dk.trip("drill")
        rel = acks.get("release") or {}
        br.wait(10)
        report["kill_drill"] = {
            "controls_before": moving, "controls_after_release": rel.get("controls"),
            "all_released": bool(rel) and not any(rel.get("controls", {"x": True}).values()),
            "bridge_exited": br.poll() is not None, "stop_ms": round((time.perf_counter() - t0) * 1000, 1),
            "handler_errors": dk.errors}
        print("kill drill:", report["kill_drill"], flush=True)
        if not report["kill_drill"]["all_released"] or not report["kill_drill"]["bridge_exited"]:
            raise SystemExit("kill switch drill failed; refusing to run the mission")

        # 3. the mission
        br = bridge(a.mc_port, a.bridge_port, base, token)
        adapter = MinecraftAdapter(bridge_port=a.bridge_port, token=token)
        adapter.connect("own_server", address)
        log = ExperienceLog(ROOT / "logs" / run_id)
        hdr = {"bot_id": "gns_bot", "game": "minecraft", "episode_id": f"{run_id}-gather_logs"}
        log.start_episode(hdr["episode_id"], "minecraft", {"scenario": "gather_logs", "address": address})
        brain = ScriptedBrain(MinecraftPlaybook())
        cfg = adapter.cfg
        rt = BotRuntime(adapter, brain, BrainLabelModel(brain.label), log, hdr, adapter.scenario.mission,
                        RuntimeConfig(cadence_s=cfg["laya"]["cadence_ms"] / 1000, brain_latency_s=0.0,
                                      heartbeat_s=cfg["brain"]["heartbeat_s"], watchdog_s=3.0),
                        policy=EscalationPolicy(cfg["laya"]["act_threshold"], cfg["laya"]["min_margin"]),
                        kill=kill, forward=PhysicsForwardModel(), salience=RuleSalience(dict(hdr)))
        kill.start_watchdog()
        st = rt.run()
        res = adapter.episode_result()
        db = log.db
        n = lambda q: db.execute(q, (hdr["episode_id"],)).fetchone()[0]  # noqa: E731
        report["mission"] = {
            **res, "decisions": st.decisions, "goals_sent_down": n("SELECT COUNT(*) FROM goals WHERE episode_id=?"),
            "events_up": n("SELECT COUNT(*) FROM events WHERE episode_id=? AND kind IN ('event','surprise','override')"),
            "events_to_brain": st.events_to_brain, "brain_calls": brain.usage.calls,
            "goal_intents": [r[0] for r in db.execute("SELECT intent FROM goals WHERE episode_id=? ORDER BY id",
                                                      (hdr["episode_id"],))],
            "event_log": [r[0] for r in db.execute(
                "SELECT description FROM events WHERE episode_id=? AND kind IN ('event','override') ORDER BY id",
                (hdr["episode_id"],))][:30],
            "kill_switch": {"tripped": kill.tripped, "reason": kill.reason},
            "log": str((ROOT / "logs" / run_id).relative_to(ROOT))}
        adapter.disconnect()
        log.close()
        report["exit_criterion_passed"] = bool(res["success"])
        print(json.dumps(report["mission"], indent=1), flush=True)
    finally:
        stop(br)
        stop(server)
        Path(a.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
