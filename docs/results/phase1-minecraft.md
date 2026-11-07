# Phase 1 — Minecraft adapter, brain + scripted reflexes

**Exit criterion:** a bot completes "gather 64 logs and return to base" end to end, with goals
flowing down and events flowing up. **Result: passed** (run `minecraft-20261107T162027`).

Raw data: [`phase1-minecraft-gather-logs.json`](phase1-minecraft-gather-logs.json). Reproduce:
`python scripts/run_minecraft.py` (needs `npm install` in `bridge/minecraft/`).

## Setup

| Part | What ran |
|---|---|
| Game server | [flying-squid](https://github.com/PrismarineJS/flying-squid) 1.12 (MIT), Minecraft protocol 1.21.4, survival, offline mode, bound to 127.0.0.1 — an `own_server` context. Superflat world plus 24 planted oak-log columns (84 logs) around the base. |
| Tier 3 | `bridge/minecraft/bridge.js`: Mineflayer 4.39 + mineflayer-pathfinder. Reflexes: gather (path, dig the column, collect the drops), goto, engage, disengage, heal, explore, wait; autonomic eating. The socket is localhost-only and token-authenticated. |
| Tier 2 | No Laya yet (Phase 1). `BrainLabelModel`: the brain labels every 100 ms decision. |
| Tier 1 | `ScriptedBrain` + `MinecraftPlaybook`. **This is a rule-based stand-in for Claude**: this environment has no LLM API key. `ClaudeBrain` (`src/nervous_system/brain/claude.py`) is implemented and unit-tested with a fake client, but was not run live. |

The roadmap item "wire the existing bot army's LLM commander in as the brain" was **not done**:
the bot army is not in this repository. The `Brain` interface is what it would plug into.

## Results

| Metric | Value |
|---|---|
| Mission | gather 64 logs and return to base — **succeeded** |
| Game time | 265.5 s |
| Logs in inventory / distance to base at end | 64 / 2.86 blocks |
| Decisions logged (JSONL + SQLite) | 2,080 (every one) |
| Goals sent down | 2 (`gather:wood` amount 64, then `return:base`) |
| Events sent up | `first oak_log`, `goal gather:wood succeeded` (the brain was woken by the goal completion) |
| Bridge events | 69 blocks dug, 64 item pickups |
| Deaths | 0 |

## Safety checks (run before the mission, against the live server)

- **Play rules:** connecting to an unlisted public server was refused before any socket was
  opened: `no recorded owner permission for 'mc.example.net:25565'`.
- **Kill-switch drill:** started a reflex (bot sprinting forward), then tripped the kill switch. All
  controls were released (the bridge acknowledged `forward/sprint/... = false`) and the bridge
  disconnected and exited, in 155 ms. The mission only starts if the drill passes.
- **Sentinel trigger:** during an earlier attempt (a pathing bug left the bot stuck), creating the
  `KILL` file stopped the run cleanly: inputs released, processes stopped, and the reason
  recorded in the log.

## Issues found and fixed during the phase

1. An entity with no name crashed the snapshot. Fixed by defaulting to the entity type.
2. Pathfinder "no path" stalls left the bot 14 blocks from base. Fixed: the bridge now falls back
   to walking straight with jumps and reports a `path_fallback` event.
3. The brain was only woken when the goal stack emptied. Per Architecture §4 it now also wakes
   on every goal completion or failure.

## Limits

- flying-squid has no hostile-mob AI, so no combat context came up in this run. Combat is
  exercised in GridCraft (Phases 2–6).
- One episode on one world seed. It is a real protocol run at real time (about 4.5 minutes per
  attempt), not a statistical benchmark.
