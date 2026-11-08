# Guide: build an AI "nervous system" for your own game or app (with AI doing the work)

This repo splits an AI agent into three speeds:

| Tier | Speed | Job | Here |
|---|---|---|---|
| Brain | seconds | strategy, goals, answering "I'm not sure" | Claude (or a scripted stand-in) |
| Nervous system | 10–100 ms | pick one of a few typed options, with a confidence | Laya, fine-tuned on the brain's own choices |
| Reflexes | every tick | exact inputs | scripted or learned controllers |

The brain teaches Laya by labelling its decisions. Laya escalates when unsure, and every
escalation becomes training data, so the expensive brain is used less over time. This guide shows
how to point the system at something new, using Claude Code to write most of the code.

## 1. Pick a target you're allowed to automate

✅ **Good targets**
- **Your own game**: Roblox experience, Unity/Godot/Unreal project, Minecraft server or mod.
  Smart NPCs, rival bots, companions.
- **Game QA and balance testing**: bots that play your economy or levels for thousands of hours
  to find exploits, soft-locks and bad pacing.
- **Singleplayer games with official mod/scripting APIs** (e.g. Minecraft, Factorio), offline.
- **Simulators you control**: robotics in simulation, drones, traffic, warehouse layouts.
- **Non-game decision loops** with a slow planner and fast typed choices: support-ticket triage,
  alert routing, moderation queues. Laya's original domain.

❌ **Not this system**
- Someone else's live multiplayer game: Rivals, Pet Simulator, ranked lobbies, public servers.
  Even in a private server against yourself, the code would work the same on public servers,
  farmed items enter shared economies, and it breaks the platform's rules. The adapter refuses
  these contexts (`play_rules`), and the AI should decline to build exploit or anti-cheat tooling.
- Anything where you'd need to inject into a client, read memory, or evade detection.

Rule of thumb: **if you can't get exact state from an official hook or your own server, it's
probably not yours to automate.**

## 2. Check the fit

| Question | Good sign | Bad sign |
|---|---|---|
| How fast must decisions be? | 1–20 per second | every frame (< 16 ms) |
| How many choices at a time? | 3–8, nameable (`engage`, `heal`, `gather:wood`) | free-form text or continuous control |
| Can you read exact state? | SDK, mod, server script, your own sim | pixels only |
| Do you have a CPU or GPU budget? | GPU, or a slow loop on CPU | twitch play on CPU (Laya ≈ 80 ms there) |

Fast continuous control (aiming, steering) belongs in tier 3: a small learned policy, not Laya.

## 3. Get exact state (by platform)

| Platform | How | Example in repo |
|---|---|---|
| Minecraft | Mineflayer bot + localhost JSON socket; local server (flying-squid or your own Paper server) | `bridge/minecraft/` |
| Your own Python sim | adapter calls the sim directly; fastest training | `adapters/gridcraft/`, `adapters/skirmish/` |
| Roblox (your experience) | server-side Luau NPC; stream state from a **Studio playtest** to the Python stack over HTTP (enable HttpService). In a live game, ship a Luau-only reflex or host the model publicly | — (next step) |
| Unity | Unity ML-Agents side channel or a TCP socket from a MonoBehaviour | — |
| Godot | GDScript `StreamPeerTCP` to the same JSON-lines protocol | — |
| Apps / queues | read the record, emit the choice, log the outcome | — |

All tiers talk in `protocol/*.schema.json` messages (Goal, Decision, Escalation, Event), so the
brain and Laya never change; only the adapter does.

## 4. The workflow, with prompts for Claude Code

Open this repo in Claude Code. The `building-nervous-systems` skill
(`.claude/skills/building-nervous-systems/`) loads the procedure and the known pitfalls. Then go
step by step, and don't skip the gates:

1. **Adapter**
   > "Add an adapter for `<my game>` under `src/nervous_system/adapters/<name>/`, following the
   > Skirmish adapter. Options from the shared vocabulary; serializer per Architecture §5. Add 3–4
   > benchmark scenarios with clear success conditions."
2. **Safety first**
   > "Wire the kill switch and play rules for `<name>`; only allow `<singleplayer | my server
   > address>`. Run a kill-switch drill and show me the controls are released."
3. **Baseline**
   > "Write a playbook for the scripted brain, then run `scripts/bench.py --game <name> --mode
   > brain --episodes 10` and save the results."
   With a Claude API key you can use `ClaudeBrain` instead. It's slower and costs tokens, so
   benchmark it on fewer episodes.
4. **Dataset**
   > "Run `scripts/make_dataset.py --game <name>`: brain-labelled rollouts with exploration,
   > deduped and capped per label."
5. **Fine-tune and calibrate**
   > "Fine-tune Laya from `laya-multilingual` (or our latest checkpoint) with
   > `scripts/train_laya.py`, int8 eval, ≥ 300 calibration items."
6. **Live sanity check** (the step most often skipped)
   > "Run one Laya rollout per scenario and compare Laya's choice histogram with the brain's on
   > the same states."
7. **Gate**
   > "Benchmark Laya with `--baseline` against the brain-only run on the same seeds; write the
   > results to `docs/results/`, including what failed."
8. **Improve**: sleep cycles (`scripts/sleep.py`), event-driven brain (`scripts/phase3.py`), a
   learned reflex (`scripts/train_combat.py`), game sense and style adaptation
   (`scripts/pvp_dataset.py`, `scripts/train_adaptive.py`).

## 5. What it costs (measured here, 4-vCPU CPU, no GPU)

| Step | Time |
|---|---|
| Brain-only benchmark (40 episodes, scripted brain) | ~1 min |
| Dataset (48 rollouts → 6k unique labelled decisions) | ~2 min |
| Laya fine-tune, 3.5k rows × 2 epochs | ~95 min (minutes on a GPU) |
| Laya benchmark (20 episodes) | ~7 min |
| One sleep cycle | ~50 min |
| PvP replay dataset, 12k rounds | ~8 min |
| Combat reflex: BC → PPO → league | ~3 min |

One Laya training uses about 5 GB of RAM. Run one at a time.

## 6. Lessons that cost hours (read before trusting a number)

1. **Test accuracy is not live performance.** A Laya at 91.5% label accuracy chose the same
   option 100% of the time in live fights. Always do step 6.
2. **Calibrate on enough data.** Temperatures fitted on < 100 examples flattened confidence and
   multiplied escalations 17×. That happened twice, at two different layers.
3. **Gate against the champion,** not the previous attempt.
4. **Audit your simulator.** Tick order made one side win 100%; identical bots double-KO'd
   forever. If a mirror match isn't near 50%, the sim is lying.
5. **The teacher can be wrong.** Our brain's "retreat to heal" rule lost fights; the student
   looked better only because it learned that rule badly. Fix the teacher, then re-distil.
6. **Adaptation needs options.** A network given the opponent's style ignored it; picking among
   existing policies by expected win rate worked (+7.7 points). Nothing beats a style no policy
   can counter.
7. **Report failures.** `docs/results/` keeps the attempts that didn't pass, with causes. That is
   what makes the passing numbers believable.

Full numbers for every phase: [`docs/results/README.md`](results/README.md).
