# Phases 2 & 3 — Laya in the loop; prediction, salience and an event-driven brain

All runs are in **GridCraft** (this repo's simulator) with the **scripted stand-in brain**. No LLM
API key was available, so brain "tokens" are estimated as characters ÷ 4 of what an LLM brain
would have been sent. Laya is `laya-multilingual` fine-tuned here, running int8 on 4 CPU threads.

## Phase 2 — Laya in the loop

**Exit:** Laya handles ≥ 70% of decisions without escalating, with no benchmark regression vs
brain-only. **Result: passed — 99.8% acted, no regression (Laya is better: 95% vs 70% success).**

### Bootstrap dataset (brain labels every decision)

`scripts/make_dataset.py`: 48 brain-labelled rollouts (4 scenarios × 12 seeds; 2/3 of them with
20% random exploration so recovery states appear) → 49,334 labelled decisions → deduplicated and
capped per label → **6,271 unique decisions** (target 5–10k). The split is by seed: train 3,726,
calib 1,453, test 1,092.

### Fine-tune + calibration

`scripts/train_laya.py`: `laya.train.finetune` on `laya-multilingual`, soft-CE, 2 epochs,
option-order shuffling, max_len 256, CPU → **94 minutes**. Evaluated int8, as the runtime uses it,
on 600 held-out test decisions:

| | zero-shot (Phase 0) | fine-tuned, raw | fine-tuned, temperature-calibrated |
|---|---|---|---|
| accuracy vs brain label | ~0.35 | **0.953** | 0.953 |
| ECE | — | 0.016 | 0.034 |
| Brier | — | 0.069 | 0.077 |
| would act (p ≥ 0.75, margin ≥ 0.15) | — | 0.948 | 0.982 |
| accuracy when acting | — | 0.979 | 0.963 |

Per label: continue 0.997, engage 0.99, explore 0.985, **disengage 0.76, heal 0.32**. The
minority labels are weak. Calibration is mandatory in the design and was applied, but on this test
split the temperature fitted on the calib split (T = 0.56) made ECE slightly worse. The sleep
cycles refit it every cycle.

### Benchmark (5 paired seeds per scenario, `docs/results/phase2-laya.json`)

| scenario | brain-only success | Laya success | Laya acted | escalations / h | Laya ms (mean, incl. cache) | cache hits |
|---|---|---|---|---|---|---|
| gather_logs | 5/5 | 5/5 | 100% | 0 | 36 | 63% |
| survive (5 min) | 2/5 | **5/5** | 99.8% | 14 | 10 | 90% |
| duel | 2/5 | **4/5** | 100% | 0 | 56 | 44% |
| diamonds | 5/5 | 5/5 | 99.4% | 108 | 48 | 51% |
| **overall** | 70% | **95%** | **99.8%** | 31 | | |

Brain tokens / hour (5% DAgger spot checks on): 4.8M (brain labels everything) → 0.62M.

**Why Laya beats its teacher:** Laya learned the brain's "retreat to eat at low hp" rule only
partly (32% accuracy on `heal`). In this simulator that rule loses fights (see the Phase 4 finding),
so Laya's mistakes happen to help. This is luck, not insight. The proper fix is in the brain
(`--game-sense`, playbook v2) and then distillation.

## Phase 3 — forward model, surprise, salience, event-driven brain

**Exit:** brain tokens / hour drop ≥ 50% vs Phase 2 at equal benchmark scores. **Result: passed —
−85.2% pooled, identical scores.** The unweighted per-scenario mean drops only −24%; the
explanation is below the table.

- Forward model v1 (`modules/forward.py`): constant-velocity extrapolation plus analytic
  `opp_aim_lock`, `crit_incoming` and `hp_loss_1s`, in the `PRED` line. Surprise = prediction error
  (position, unexpected damage, new hostiles nearby) → `surprise` events.
- Salience filter (`modules/salience.py`): hostile newly within 10, damage spike, critical health,
  death, first item of a kind.
- The forward model was built in Phase 1 and was already feeding `PRED` to Laya in Phase 2 (the
  training data includes it). The Phase 2 vs 3 comparison therefore isolates the brain wiring,
  which is what the exit criterion measures:
  - **Phase 2 wiring:** the brain is polled with a world summary every 2 s.
  - **Phase 3 wiring:** the brain wakes on salient/surprise/goal events and escalations;
    30 s heartbeat as a backup.
- Same checkpoint, same seeds, spot checks off. Spot checks are a training cost; with them on,
  ~150 labels per survive episode swamp the difference (−15%; kept in `*-spotchecks.json`).

| scenario | success P2 → P3 | brain tokens / h P2 → P3 | change |
|---|---|---|---|
| gather_logs | 5/5 → 5/5 | 123k → 8.3k | −93% |
| survive | 5/5 → 5/5 | 133k → 3.3k | −98% |
| duel | 4/5 → 4/5 | 1.13M → 1.13M | 0% |
| diamonds | 5/5 → 5/5 | 166k → 45k | −73% |
| **pooled (total tokens ÷ total game hours)** | 95% → 95% | **147k → 22k** | **−85%** |

Duel fights last about 5 s, so they end before any heartbeat. Their one planning call, divided by
5 seconds, dominates the unweighted average of per-scenario rates. The pooled number weights every
hour of play equally.
