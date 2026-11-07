# Phase 7 — Second game (Skirmish)

**Exit:** the second game reaches a basic benchmark without changes to tiers 1–2 code.
**Result: partly met.**

- ✅ The adapter-only part holds. Skirmish (this repo's top-down arena shooter: ranged hit chance,
  line of sight and cover, magazines and reloads, medkits, crates) was added as
  `adapters/skirmish/` + `configs/games/skirmish.yaml` + a 4-line registry entry in `wiring.py`.
  The commit that added it (`0701067`) changed **no tier-1/2 file**, and `tests/test_architecture.py`
  checks that tiers 1–2 never import adapters. The full stack (brain, runtime, forward model,
  salience, drives, logging, benchmark harness, sleep tooling) runs Skirmish unchanged and reaches
  its benchmark brain-only: **47% success**.
- ❌ Reusing Laya does not yet reach that benchmark. Fine-tuning the GridCraft checkpoint on 1,200
  brain-labelled Skirmish decisions (+300 GridCraft replay rows, 1 epoch) gives 91.5% label
  accuracy on held-out data. In live play it **regresses to 13% success**.

Raw data: [`phase7-skirmish-brain-only.json`](phase7-skirmish-brain-only.json),
[`phase7-skirmish-laya.json`](phase7-skirmish-laya.json).

| scenario (5 seeds) | brain-only | Laya (gc-sk-v1) | Laya acted | escalations / h |
|---|---|---|---|---|
| hold_out (2 min of waves) | 40% | 20% | 99% | 95 |
| gunfight (1v1) | 40% | 0% | 100% | 37 |
| supply_run (4 crates + return) | 60% | 20% | 100% | 24 |

## Why Laya fails here

In the gunfight Laya chose `reposition` on **100%** of 4,865 decisions, with p ≈ 0.98, on fresh
states (full hp, magazine full, `weapon ready`). The brain chooses `engage` there. Its training
data has many `reposition` labels from states that differ by a single token (`weapon reloading` /
`healing`), and one epoch on 1,200 rows did not learn the distinction. Per-label test accuracy
(engage 0.82, reposition 0.88) hid this: the live state distribution weights these confusable
states far more than the test split does.

## Attempt 2: all Skirmish data, 2 epochs (`laya-gc-sk-v2`)

All 2,236 Skirmish rows + 300 GridCraft replay rows, 2 epochs, a 10% internal calibration slice
(about 40 min). Held-out: accuracy 94.8%, acted 96.2%, ECE 0.022. Benchmark
([`phase7-skirmish-laya-v2.json`](phase7-skirmish-laya-v2.json), same 5 seeds):

| scenario | brain-only | v1 | **v2** |
|---|---|---|---|
| hold_out | 40% | 20% | **40%** |
| gunfight | 40% | 0% | **0%** |
| supply_run | 60% | 20% | **60%** |
| overall | 47% | 13% | 33% |

v2 matches the brain on 2 of 3 scenarios. The gate still fails on **gunfight**, the only
regression. `use_ability:reload` stays at 0% accuracy: it has only 6 training examples.

**Next step:** more Skirmish data and epochs (the GridCraft checkpoint needed 3.5k rows × 2 epochs),
plus a check of live-policy agreement on the first rollout before trusting test accuracy. No tier-1/2
change is needed for that.
