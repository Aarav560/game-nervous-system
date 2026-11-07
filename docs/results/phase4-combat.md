# Phase 4 — Combat reflex

**Exit criterion:** the combat reflex beats a scripted practice bot in ≥ 70% of rounds on your own
server. **Result: passed in GridCraft, 95% of 200 rounds.** The criterion is met, but the practice
bot turned out to be weak. Read the comparison table before drawing conclusions.

Raw data: [`phase4-combat-reflex.json`](phase4-combat-reflex.json). Reproduce:
`python scripts/train_combat.py` (about 2.5 minutes on 1 CPU thread).

## What was built

- `adapters/gridcraft/combat.py`: an actor-critic MLP (24 observation features, 2×128 tanh). It
  has factorised action heads (9 relative move directions, sprint, attack, eat) and **auxiliary
  heads `will_hit` / `will_be_hit`** (next 0.5 s), copied from the Minecraft Combat AI design
  (see [phase4-reference-checkpoints.md](phase4-reference-checkpoints.md)). One observation in,
  one action out, every 50 ms tick.
- `LearnedEngage` replaces the scripted `engage` reflex when `--combat-policy` is given. The
  checkpoint is saved and loaded with `torch.load(..., weights_only=True)`.
- Training recipe (Architecture §12.4):
  1. **Behaviour cloning** on the scripted engage+heal controller (8 epochs, 26k samples). This
     stands in for "record own PvP play": there is no human play data in this environment.
  2. **PPO** for 120 iterations × 32 parallel duels against the practice bot and the mirror bot.
  3. **League** for 80 iterations: main policy vs {practice bot, mirror bot, past snapshots, an
     exploiter trained only against the current main}.
- The policy is an MLP, not the GRU the reference uses. The observation already includes
  velocities and cooldowns, so it is close to Markov. A GRU is a follow-up.
- No GPL weights were used; everything was trained from scratch.

## Results (200 evaluation rounds each, held-out seeds)

| Policy | vs practice bot | vs mirror bot¹ |
|---|---|---|
| Scripted engage + heal rule (the BC teacher) | 36.5% | 38% |
| **Scripted engage, no heal/retreat rule** | 92% | 93.5% |
| After BC | 86% | 96% |
| After PPO | 94% | 94% |
| **Final (after league)** | **95%** | **95%** |
| Final vs the league exploiter | 90% (50 rounds) | — |

¹ The mirror bot runs our own scripted engage+heal logic.

What the numbers say:

- The practice bot loses mainly because it **retreats to eat** when low, which gives the chaser
  free sprint-crits. Simple aggression beats it 92% of the time. The learned reflex adds about
  3 points on top of that (roughly 1.5 standard errors), so it is about as good as a well-tuned
  scripted reflex here, not dramatically better.
- RL did learn something real: it dropped the teacher's losing retreat-to-heal habit (36.5% → 95%).

## Plugged under Laya's `engage` (full stack, brain-only, 20 episodes)

| Setup | duel | survive |
|---|---|---|
| Playbook v1 + scripted reflex | 35% | 50% |
| Playbook v1 + learned reflex | 30% | 45% |
| Brain always picks `engage` (reflex decides) + scripted reflex | 95% | — |
| Brain always picks `engage` (reflex decides) + learned reflex | **100%** | — |

**Finding:** the reflex is not the bottleneck. The brain playbook's own "retreat to eat at low hp"
rule interrupts whichever reflex is running and loses fights. That rule was also in the data
Laya was trained on (Phase 2). The fix belongs in the brain or its labels; the learning loop
(Phase 5) is the intended way to correct it.

## Simulator fixes found while building this phase

Two GridCraft fairness bugs made early numbers meaningless, and both are fixed:

1. Attacks were resolved in agent order, so `bot` always struck first in simultaneous exchanges.
   Hits are now planned from the pre-attack state and applied together.
2. Identical scripted fighters stayed in lockstep and always double-KO'd each other. There is now
   a 10% whiff chance per swing (like aim error or ping), seeded per world.
