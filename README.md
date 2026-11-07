# Game Nervous System

An AI that plays real-time games by splitting **thinking** from **reacting**:

| Tier | Role | Component | Speed |
|---|---|---|---|
| 1 | Brain — strategy and goals | Claude (any LLM) | every 1–5 s, or on escalation |
| 2 | Nervous system — fast tactical choices + confidence | [Laya](docs/REFERENCES.md#laya-tier-2-model), local Jev-style decision model | 10–20 Hz |
| 3 | Reflexes — exact inputs | learned / scripted controllers | every tick or frame |

Information flows **down as intent** and **up as surprise**. When Laya is unsure it escalates to the brain; every correction becomes training data, so the system gets faster the more it plays.

First testbed: **Minecraft** (Mineflayer bridge, tested against a local flying-squid server). Fast training and benchmark loops run in **GridCraft**, this repo's own headless survival/PvP simulator. **Skirmish**, an arena shooter, is the second game and was added as an adapter only.

## Docs

- [Architecture](docs/ARCHITECTURE.md) — full design: tiers, modules, protocol, learning loop, multi-game plan, rules.
- [Roadmap](docs/ROADMAP.md) — phases with exit criteria and current status.
- [Results](docs/results/README.md) — what each phase measured, including what did **not** pass.
- [References](docs/REFERENCES.md) — Laya, Jev, Minecraft Combat AI, related work.

## Layout

```
docs/                    design docs + docs/results/ (per-phase results)
protocol/                JSON schemas for messages between tiers (v0, frozen)
configs/games/           per-game config: latency profile, play rules, rewards, option sets
bridge/minecraft/        Node: Mineflayer bridge (tier-3 reflexes) + flying-squid local test server
scripts/                 bench, run_minecraft, make_dataset, train_laya, sleep, train_combat, run_hive, phase3
src/nervous_system/
  protocol.py            pydantic message models (Goal, Decision, Escalation, Event)
  vocabulary.py          shared action vocabulary (Architecture §9)
  state.py               game-agnostic Facts / Entity (proprioception)
  brain/                 tier 1 — Brain ABC, ClaudeBrain, ScriptedBrain (playbook stand-in)
  laya/                  tier 2 — Laya wrapper, escalation policy, cache, calibration, metrics
  modules/               kill switch, forward model, salience, drives, memory
  runtime/               per-bot loop, experience log (JSONL + SQLite), bus, hive
  adapters/              GameAdapter base; minecraft/, gridcraft/, skirmish/
  wiring.py              builds runtimes from names (the only place that knows every game)
tests/                   pytest suite, including the CLAUDE.md ground rules as tests
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build on a GPU box
pip install -e '.[laya,dev]'
pytest

python scripts/bench_laya.py                                   # Phase 0 Laya latency
(cd bridge/minecraft && npm install) && python scripts/run_minecraft.py   # Phase 1, real protocol
python scripts/bench.py --game gridcraft --mode brain --episodes 10       # benchmark harness
```

To use Claude as the brain, install `anthropic`, set `ANTHROPIC_API_KEY` and construct
`nervous_system.brain.claude.ClaudeBrain`. The recorded results in this repo use the scripted
stand-in brain, because no API key was available where they were produced.

## Rules

Bots run only in singleplayer, on your own servers, or on servers whose owner has agreed. The adapter refuses anything else. See [Architecture §14](docs/ARCHITECTURE.md#14-rules-safety-and-licensing).

## Licence

MIT (see `LICENSE`). Keep GPL-licensed models (e.g. the Minecraft Combat AI sword transformer) out of any distributed build, or that build must be GPL — see Architecture §14. No third-party model weights are committed here.
