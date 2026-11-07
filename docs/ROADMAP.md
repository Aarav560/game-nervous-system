# Roadmap

Each phase has an exit criterion. Don't start the next phase until the current one passes its exit check.

## Phase 0 — Scaffold and measurements

- [ ] Run Laya locally (Typed Decisions + English checkpoints); measure single-decision and batched latency on your GPU.
- [ ] Claim free Jev / Minecraft Combat AI API credits only if needed for baselines; measure latency from Nepal.
- [ ] Freeze the protocol schemas in `protocol/` (v0).
- [x] Repo licence: MIT. Never bundle GPL models in a distributed build (`ARCHITECTURE.md` §14).

**Exit:** Laya answers a hand-written state text in < 50 ms on local hardware.

## Phase 1 — Minecraft adapter, brain + scripted reflexes

- [ ] Minecraft adapter: state reader (Mineflayer or Fabric mod), serializer, actuator.
- [ ] Wire the existing bot army's LLM commander in as the brain, emitting `Goal` objects.
- [ ] Reflexes: Baritone pathing, AltoClef tasks, autonomic rules (eat, re-equip, totem).
- [ ] Kill switch and play-rules enforcement.
- [ ] Experience logging (JSONL + SQLite).

**Exit:** a bot completes "gather 64 logs and return to base" end-to-end with goals flowing down and events flowing up.

## Phase 2 — Laya in the loop

- [ ] Brain labels every decision in slow-speed runs to bootstrap a dataset (target: 5–10k decisions).
- [ ] Fine-tune Laya; calibrate; plug in with escalation thresholds.
- [ ] Benchmark harness v1 (4 Minecraft scenarios).

**Exit:** Laya handles ≥ 70% of decisions without escalating, with no benchmark regression vs brain-only.

## Phase 3 — Prediction and salience

- [ ] Forward model v1 (physics extrapolation + combat-reflex auxiliary heads).
- [ ] Surprise events; salience filter (rules).
- [ ] Brain becomes fully event-driven (heartbeat only as backup).

**Exit:** brain tokens per hour drop ≥ 50% vs Phase 2 at equal benchmark scores.

## Phase 4 — Combat reflex

- [ ] Study the open Minecraft Combat AI checkpoints (load safely, map features) as reference.
- [ ] Record own PvP play; behaviour-clone a sword reflex; self-play on a private Paper server; league training.
- [ ] Plug combat reflex under Laya's `engage` / `disengage`.

**Exit:** combat reflex beats a scripted practice bot ≥ 70% of rounds on your own server.

## Phase 5 — Memory, drives, sleep

- [ ] Working / episodic / skill memory.
- [ ] Drives feeding Laya.
- [ ] Automated sleep cycle: review → label → fine-tune → calibrate → benchmark → promote.

**Exit:** escalation rate falls across three consecutive sleep cycles with no benchmark regression.

## Phase 6 — Hive

- [ ] Message bus between bots; shared markers; task claims.
- [ ] Batched Laya inference for N bots on one GPU.

**Exit:** 5 bots coordinate a base-building mission with one brain.

## Phase 7 — Second game

- [ ] Pick a game with easy hooks; write its adapter only.
- [ ] Reuse Laya (with `GAME` line) and the brain unchanged.

**Exit:** the second game reaches a basic benchmark without changes to tiers 1–2 code.
