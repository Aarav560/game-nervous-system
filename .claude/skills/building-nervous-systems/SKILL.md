---
name: building-nervous-systems
description: Use when adding a new game, simulator, robot or app to the Game Nervous System (brain/Laya/reflex tiers), writing a GameAdapter, training or calibrating Laya on brain-labelled decisions, running benchmarks or sleep cycles, or when asked to make an AI play, bot or automate a game.
---

# Building nervous systems

## Overview
A slow planner (LLM brain) sets goals; a fast typed-decision model (Laya) picks options and escalates when unsure; reflexes execute. Everything domain-specific lives in one adapter. **A result only counts if a regression gate on live rollouts says so** — held-out label accuracy has repeatedly lied.

## First: is the target allowed?
Build only for **a game/system you own, singleplayer, or a server whose owner agreed** (record it in `play_rules`). Someone else's live game (Rivals, Pet Sim, ranked Minecraft): no adapter, no screen-reading aimbot, no executor/anticheat work — offer "build your own version + NPC bots" instead. Private server against yourself doesn't change this when the code works identically on public servers or the loot enters a shared economy.

## Fit check
| Good fit | Poor fit |
|---|---|
| decisions at 1–20 Hz under slow strategy | sub-50 ms twitch control on CPU (Laya ≈ 80 ms CPU int8) |
| 3–8 options per context, a vocabulary verb each | free-form outputs |
| exact state via hooks/SDK/your own server | pixels only |

## Steps (each one gated)
1. **Adapter** in `adapters/<name>/` implementing `GameAdapter`; options from `vocabulary.py`; serializer follows Architecture §5 (fixed order, `none`, < 350 tokens). Register in `wiring.py`. Run `pytest tests/test_architecture.py` — tiers 1–2 must not change.
2. **Kill switch + play rules** before the first connection; drill it live.
3. **Brain-only baseline** with `scripts/bench.py --mode brain` (paired seeds, ≥ 5 per scenario).
4. **Dataset**: `scripts/make_dataset.py` (brain labels every decision, ~20% exploration, cap per label).
5. **Fine-tune + calibrate** with `scripts/train_laya.py`; calibrate the variant you run (int8).
6. **Live check before trusting metrics**: one rollout; compare Laya's choice distribution with the brain's on the same states.
7. **Laya bench** with `--baseline` (gate) → write `docs/results/`.

See `pitfalls.md` in this directory before steps 4–7.

## Common mistakes (all observed)
| Mistake | Symptom | Fix |
|---|---|---|
| Trusting test accuracy | 91% labels, 0% gunfights (always `reposition`) | step 6; check per-context choice histograms |
| Calibrating on < 300 items | T ≈ 1.5, acted 5% in combat, escalations ×17 | pool with a fixed anchor slice; same for `laya.train` internal slice |
| Promoting vs previous cycle | worse model promoted | compare with the **champion** |
| Sim fairness bugs | 100% or 0% win rates, double-KOs | simultaneous resolution, seeded noise; sanity-check mirror matches ≈ 50% |
| Teacher has a bad rule | student "beats" teacher by luck | fix the brain/playbook, then re-distil |
| Two Laya trainings at once | OOM kill, silent exit | one ~5 GB training at a time |
