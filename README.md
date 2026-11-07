# Game Nervous System

An AI that plays real-time games by splitting **thinking** from **reacting**:

| Tier | Role | Component | Speed |
|---|---|---|---|
| 1 | Brain — strategy and goals | Claude (any LLM) | every 1–5 s, or on escalation |
| 2 | Nervous system — fast tactical choices + confidence | [Laya](docs/REFERENCES.md#laya-tier-2-model), local Jev-style decision model | 10–20 Hz |
| 3 | Reflexes — exact inputs | learned / scripted controllers | every tick or frame |

Information flows **down as intent** and **up as surprise**. When Laya is unsure it escalates to the brain; every correction becomes training data, so the system gets faster the more it plays.

First testbed: **Minecraft**, building on the existing Baritone/AltoClef bot army. Designed to extend to other games through per-game adapters.

## Docs

- [Architecture](docs/ARCHITECTURE.md) — full design: tiers, modules, protocol, learning loop, multi-game plan, rules.
- [Roadmap](docs/ROADMAP.md) — phases with exit criteria.
- [References](docs/REFERENCES.md) — Laya, Jev, Minecraft Combat AI, related work.

## Layout

```
docs/                 design docs
protocol/             JSON schemas for messages between tiers (v0, frozen)
scripts/              benchmark harness (bench_laya.py: Phase 0 latency)
tests/                pytest suite (protocol/schema agreement, Laya wrapper, escalation)
configs/games/        per-game config: latency profile, play rules, rewards, options
src/nervous_system/   Python package skeleton (interfaces only, no implementation yet)
  brain/              tier 1 — LLM planner interface
  laya/               tier 2 — Laya decision model wrapper, escalation, calibration
  vocabulary.py       shared action vocabulary (Architecture §9)
  reflex/             tier 3 — reflex controller contract
  modules/            forward model, salience, memory, drives, kill switch
  adapters/           GameAdapter base + Minecraft stub
  runtime/            bus + main loop
CLAUDE.md             instructions for the Claude Code session that builds this
```

## Status

Phase 0 in progress — see [Roadmap](docs/ROADMAP.md) and [results](docs/results/).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build on a GPU box
pip install -e '.[laya,dev]'
pytest
python scripts/bench_laya.py          # Phase 0 Laya latency benchmark
```

## Rules

Bots run only in singleplayer, on your own servers, or on servers whose owner has agreed. The adapter refuses anything else. See [Architecture §14](docs/ARCHITECTURE.md#14-rules-safety-and-licensing).

## Licence

MIT (see `LICENSE`). Keep GPL-licensed models (e.g. the Minecraft Combat AI sword transformer) out of any distributed build, or that build must be GPL — see Architecture §14.
