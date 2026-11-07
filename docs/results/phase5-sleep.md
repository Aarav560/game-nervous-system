# Phase 5 — Memory, drives, sleep

**Exit:** the escalation rate falls across three consecutive sleep cycles with no benchmark
regression. **Result: NOT passed.** Two full attempts produced candidates that escalated *more*
than the starting champion and regressed the benchmark. The promotion gate correctly rejected
them (after a gate bug was fixed). Attempt 3 targets the remaining cause; its outcome is below.

## What was built (all working)

- **Working memory:** the `WM` line of Laya's state text (last 3 actions).
- **Episodic memory** (`modules/memory.EpisodicMemory`, SQLite): every reviewed episode is stored
  with its game, situation type and outcome; `recall(situation, game=None)` works across games.
- **Skill library** (`modules/memory.SkillLibrary`, JSON): goal outcomes per game. After the sleep
  runs it holds `gather:wood`, `gather:iron_ore`, `return:base` and `build:iron_pickaxe`, each at
  100% of 45 attempts.
- **Drives feeding Laya:** `fear`, `hunger`, `fatigue` and `curiosity` (`modules/drives.py`) appear
  in every state text as the `DRIVES` line, and Laya was trained with them.
- **Automated sleep cycle** (`scripts/sleep.py`):
  1. **Review:** read the champion's last benchmark logs.
  2. **Label:** the brain labels every escalation, every spot check, and a DAgger sample of 800 of
     Laya's own confident decisions.
  3. **Fine-tune** from the champion: 1 epoch, plus replay.
  4. **Calibrate** on a held-out set.
  5. **Benchmark:** 4 scenarios × 5 seeds.
  6. **Promote** only if the gate passes against the brain-only baseline *and* the champion, and
     escalations do not rise.

## Results

The starting champion is `laya-gc-v1`: **31 escalations/hour**, 95% success.

| attempt | cycle | escalations / h | success | acted | temperature | promoted |
|---|---|---|---|---|---|---|
| 1 | 1 | 531 | 60% | 67.4% | 1.54 | no |
| 1 | 2 | 267 | 95% | 89.2% | 0.86 | yes — **wrongly** (gate bug) |
| 1 | 3 | 564 | 55% | 84.1% | 0.85 | no |
| 2 | 1 | 184 | 85% | 98.4% | 0.83 | no |
| 2 | 2 | 451 | 40% | 82.1% | 1.20 | no |
| 2 | 3 | — | — | — | — | stopped (same trend; CPU needed for Phase 7) |

Raw data: [`phase5-attempt1/`](phase5-attempt1/), [`phase5-attempt2/`](phase5-attempt2/).

## Why it failed: three causes, found in order

1. **Calibration on tiny, skewed sets (attempt 1).** Each night's calib split was 91–194 examples,
   mostly easy states. Cycle 1 fitted T = 1.54, which flattened every distribution: in the duel,
   Laya acted on only 4.7% of decisions and fell back to `disengage`, losing fights. The *same
   checkpoint uncalibrated* acted on 98% of its test set.
   *Fix (attempt 2):* calib and test are pooled with a fixed 300-row slice of the original
   held-out sets.
2. **Forgetting combat (attempt 1).** The review data is dominated by gathering and survival
   states, and only 400 rows were replayed. *Fix (attempt 2):* 1,200 replay rows, half from
   combat, and new labels capped per (option set, label).
3. **Promotion gate bug (attempt 1).** Candidates were compared with the *previous cycle*, not the
   champion, so cycle 2 (267/h, 9× the champion) was promoted. *Fix:* compare with the champion,
   on both escalations and the benchmark gate.
4. **Still open after attempt 2:** `laya.train.finetune` *also* refits the checkpoint's internal
   temperatures (`temperature_by_options`) on its own calibration slice, which was 5% (75–95
   items). This is the same small-sample failure one layer down: attempt 2 cycle 2 drifted to
   T = 1.20 and acted 43% in the duel. *Fix (attempt 3):* hold out 20% (up to 400 items) for that
   internal refit.

## Lessons for the design

- Continued fine-tuning of a calibrated decision model needs **frozen or large-sample
  calibration**. Recalibrating every night on a small, self-selected slice is how the escalation
  rate goes up instead of down.
- The promotion gate is what kept every bad night out of production. It is the most important
  part of the sleep cycle.
- On this CPU each night costs about 50 minutes (25 min training + benchmark), so three nights
  plus the debugging did not fit in one session. On a GPU this is a few minutes per night.
