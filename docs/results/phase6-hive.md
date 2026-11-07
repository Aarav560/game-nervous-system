# Phase 6 — Hive

**Exit:** 5 bots coordinate a base-building mission with one brain. **Result: passed in GridCraft.**

Raw data: [`phase6-hive-brain.json`](phase6-hive-brain.json),
[`phase6-hive-brain-nocoord.json`](phase6-hive-brain-nocoord.json),
[`phase6-hive-laya.json`](phase6-hive-laya.json). Reproduce: `python scripts/run_hive.py`.

## Setup

- **Mission** (`scenarios.build_base`): deliver 60 logs + 30 cobblestone to the base to complete the
  wall blueprint, within 400 s. Zombies spawn near random bots (max 3, one every 10 s).
- **One brain** (`ScriptedBrain`, a single instance) plans for every bot. Each bot's world summary
  carries a hive status block: current task claims, plus danger markers in world coordinates.
- **Bus** (`runtime/bus.py`, `runtime/hive.py`): `hive.claim` / `hive.release` carry each bot's
  current goal; `hive.marker` carries danger markers ("pheromones", published when a bot's
  salience filter flags a newly-near hostile). Gather reflexes avoid resources within 8 blocks of
  another bot's danger marker.
- **Batched Laya**: each 100 ms cycle, every bot's due decision goes into one `decide_batch` call.

## Results

| Setup | episodes | success | mean time | bots lost | brain calls / episode | Laya batch | acted |
|---|---|---|---|---|---|---|---|
| brain labels every decision, **with claims** | 10 | 100% | **44.8 s** | 0 | 2,268 | 5.0 | 100% |
| brain labels every decision, claims ignored (ablation) | 10 | 100% | 93.5 s | 0 | ~4,700 | 5.0 | 100% |
| **Laya (gc-v1, int8, 1 thread), with claims** | 3 | **100%** | **37.9 s** | 0 | **21** | 5.0 | **99.9%** |

- Coordination matters: with claims, the brain splits the bots across wood and stone (3 + 2), and
  the wall is finished in **half the time** of the ablation, where everyone gathers wood first.
- With Laya in the loop, the single brain is only consulted for plans and the rare escalation:
  21 calls per episode for 5 bots, against 2,268 when it labels every decision.
- Laya was never trained on the hive scenario. Its gathering/travel states look like the
  single-bot ones, which is part of why the shared vocabulary exists (Architecture §9).

## Limits

- Batching on this CPU brings little speed-up: batch 5 int8 is about 1.4× the throughput of
  single calls (Phase 0). The value of batching is on a GPU.
- 3 Laya episodes (CPU time). Brain-only runs are 10 episodes each.
