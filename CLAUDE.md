# CLAUDE.md — build guide

You are building the Game Nervous System. Read `docs/ARCHITECTURE.md` fully before writing code, then work phase by phase from `docs/ROADMAP.md`.

## Ground rules

- Tiers only talk to neighbours: brain ↔ Laya ↔ reflexes. Claude never emits key presses.
- All inter-tier messages follow `protocol/*.schema.json`. Change a schema only with a version bump and a note in the commit.
- Everything game-specific goes in `src/nervous_system/adapters/<game>/`. Tiers 1–2 must not import adapter code.
- Laya input text follows the serializer rules in Architecture §5: fixed field order, quantised values, `none` for missing, under ~350 tokens.
- Every Laya decision is logged (Architecture §12.1). No silent decisions.
- The kill switch must work before any bot connects to a game.
- Play rules (Architecture §14) are enforced in the adapter: refuse to connect outside allowed contexts.

## Safety

- Load any `.pt` file with `torch.load(path, weights_only=True)` in an isolated environment.
- Never train on Minecraft Combat AI API output; never reverse-engineer their service. Free HF checkpoints only, with licence conditions.
- Self-hosted Laya: bind to localhost and set `LAYA_API_KEY`.

## Stack

- Python 3.11+, PyTorch, Hugging Face transformers (Laya), pydantic for message models.
- Minecraft side: Mineflayer (Node) or a Fabric mod (Java) talking to Python over a local socket.
- Logs: JSONL per episode + SQLite index.

## Definition of done per phase

The phase's exit criterion in `docs/ROADMAP.md` passes on the benchmark harness, and results are recorded in `docs/results/`.
