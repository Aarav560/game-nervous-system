# Roadmap

Each phase has an exit criterion. Don't start the next phase until the current one passes its exit check.

> **Status (7 Nov 2026).** All phases have been built. Results are in [`docs/results/`](results/README.md).
> Read the caveats before quoting any number:
> - No GPU was available, so Phase 0 was measured on a 4-vCPU CPU.
> - No LLM API key was available, so the brain in every recorded run is a **scripted stand-in**
>   (`ScriptedBrain` + a game playbook). `ClaudeBrain` is implemented and unit-tested, but was not run live.
> - Phase 1 ran on a real Minecraft-protocol server: local flying-squid via Mineflayer.
>   Phases 2–7 ran in **GridCraft** (this repo's own simulator, a stand-in for a sped-up Paper
>   server) and **Skirmish** (the second game).
> - Phases 0 and 5 did **not** pass. They are marked below.

## Phase 0 — Scaffold and measurements — ❌ exit not met on CPU

- [x] Run Laya locally (Typed Decisions + English + Multilingual checkpoints); measure single-decision and batched latency. Measured on CPU only: no GPU in this environment.
- [ ] Claim free Jev / Minecraft Combat AI API credits only if needed for baselines; measure latency from Nepal. *Not needed so far; not done (no credentials here).*
- [x] Freeze the protocol schemas in `protocol/` (v0).
- [x] Repo licence: MIT. Never bundle GPL models in a distributed build (`ARCHITECTURE.md` §14).

**Exit:** Laya answers a hand-written state text in < 50 ms on local hardware.
**Result:** best p50 is about 80 ms (Laya-multilingual, int8, 4 vCPU). Not met on this hardware; it needs a GPU (published T4 figure ~33 ms). → [results](results/phase0-laya-latency.md)

## Phase 1 — Minecraft adapter, brain + scripted reflexes — ✅

- [x] Minecraft adapter: state reader (Mineflayer bridge), serializer, actuator.
- [ ] Wire the existing bot army's LLM commander in as the brain, emitting `Goal` objects. *The bot army is not in this repo. `ClaudeBrain` implements the interface; the runs used the scripted stand-in.*
- [x] Reflexes: Mineflayer pathfinding, gathering and combat reflexes in the bridge, plus autonomic eating. (Baritone/AltoClef are Java client mods and are not used.)
- [x] Kill switch and play-rules enforcement (live drill before every run).
- [x] Experience logging (JSONL + SQLite).

**Exit:** a bot completes "gather 64 logs and return to base" end-to-end with goals flowing down and events flowing up.
**Result:** passed on a local flying-squid server (64 logs, back at base in 265 s). → [results](results/phase1-minecraft.md)

## Phase 2 — Laya in the loop — ✅ (GridCraft)

- [x] Brain labels every decision in slow-speed runs to bootstrap a dataset (6,271 unique decisions).
- [x] Fine-tune Laya; calibrate; plug in with escalation thresholds.
- [x] Benchmark harness v1 (4 scenarios: gather 64 logs, survive 5 min, 1v1, mine to diamonds).

**Exit:** Laya handles ≥ 70% of decisions without escalating, with no benchmark regression vs brain-only.
**Result:** 99.8% acted; success 95% vs 70% for brain-only. → [results](results/phase2-3-laya.md)

## Phase 3 — Prediction and salience — ✅ (GridCraft)

- [x] Forward model v1 (physics extrapolation + analytic aim-lock / crit-incoming predictions; the learned combat reflex adds `will_hit` / `will_be_hit` heads).
- [x] Surprise events; salience filter (rules).
- [x] Brain becomes fully event-driven (heartbeat only as backup).

**Exit:** brain tokens per hour drop ≥ 50% vs Phase 2 at equal benchmark scores.
**Result:** −85% pooled over all play time, identical scores. The per-scenario mean is −24%, because 5-second duels end before any heartbeat. → [results](results/phase2-3-laya.md)

## Phase 4 — Combat reflex — ✅ (GridCraft; bar turned out to be low)

- [x] Study the open Minecraft Combat AI checkpoints (load safely, map features) as reference. → [study](results/phase4-reference-checkpoints.md)
- [x] Behaviour-clone a sword reflex (from a scripted expert: no human recordings available); self-play / PPO; league training with snapshots and an exploiter.
- [x] Plug combat reflex under Laya's `engage` (`--combat-policy`).

**Exit:** combat reflex beats a scripted practice bot ≥ 70% of rounds on your own server.
**Result:** 95% of 200 rounds. A scripted reflex without the retreat-to-heal rule already gets 92%. → [results](results/phase4-combat.md)
**Extension:** a 12,000-round PvP replay dataset, game-sense models (win model AUC 0.91; opponent style identified 91.5% of the time after 1 s), and counter-pick style adaptation (85.8% vs 78.1% mean win rate). → [results](results/pvp-gamesense-adaptation.md)

## Phase 5 — Memory, drives, sleep — ❌ exit not met

- [x] Working / episodic / skill memory.
- [x] Drives feeding Laya.
- [x] Automated sleep cycle: review → label → fine-tune → calibrate → benchmark → promote.

**Exit:** escalation rate falls across three consecutive sleep cycles with no benchmark regression.
**Result:** not met. Retrained candidates escalated more (184–564/h vs 31/h) and were rejected by the promotion gate. Root causes were found and fixed one by one (small-sample recalibration at two layers, combat forgetting, a gate bug). → [results](results/phase5-sleep.md)

## Phase 6 — Hive — ✅ (GridCraft)

- [x] Message bus between bots; shared markers; task claims.
- [x] Batched Laya inference for N bots (one `decide_batch` per cycle).

**Exit:** 5 bots coordinate a base-building mission with one brain.
**Result:** 100% success; claims halve completion time (45 s vs 94 s); with Laya, one brain makes 21 calls per episode for 5 bots. → [results](results/phase6-hive.md)

## Phase 7 — Second game — see results

- [x] Pick a game with easy hooks; write its adapter only (Skirmish: this repo's arena shooter).
- [x] Reuse Laya (with `GAME` line) and the brain unchanged.

**Exit:** the second game reaches a basic benchmark without changes to tiers 1–2 code.
→ [results](results/phase7-second-game.md)
