# Phase 0 — Laya latency on local hardware

**Exit:** Laya answers a hand-written state text in < 50 ms on local hardware.
**Result: NOT met on this machine.** The best p50 is 76–93 ms (Laya-multilingual, int8). There is
no GPU here, so this needs re-measuring on one (the published T4 figure is about 33 ms).

Hardware: Intel Xeon @ 2.10 GHz, 4 vCPU (AVX-512, AMX-BF16), no GPU; torch 2.15 CPU;
`laya` 0.4.28. Reproduce: `python scripts/bench_laya.py`.

- Clean run (nothing else running):
  [`phase0-laya-latency-clean.json`](phase0-laya-latency-clean.json), multilingual + typed-decisions.
- First run (all 3 checkpoints, measured while other jobs shared the CPU):
  [`phase0-laya-latency.json`](phase0-laya-latency.json).
- Every timed decision was also logged as a protocol `Decision` (`logs/bench/`, not committed).

The state texts are the three hand-written texts in the §5 serializer format (combat, critical
combat, gathering), each with 6 options.

## Single decision, p50 (clean run)

| checkpoint | fp32 | bf16 (AMX) | int8 (dynamic, encoder only) |
|---|---|---|---|
| Laya-multilingual (mmBERT-base, 322M) | 125–157 ms | 116–137 ms | **76–93 ms** |
| Laya-typed-decisions (ModernBERT-large, 421M) | 315–380 ms | 297–323 ms | 171–196 ms |
| Laya-English (ModernBERT-large, 421M; first run, contended) | 334–394 ms | 277–332 ms | 169–190 ms |

## Batched throughput (combat text, clean run)

| checkpoint / variant | batch 1 | batch 4 | batch 16 |
|---|---|---|---|
| multilingual int8 | 11.5 /s | 15.4 /s | 15.0 /s |
| multilingual bf16 | 7.7 /s | 14.5 /s | **16.6 /s** |
| typed-decisions bf16 | 3.3 /s | 6.8 /s | 6.7 /s |

## What this means for the design

- On this CPU, 10–20 Hz decisions for one bot are possible only with the decision cache: the
  quantised serializer makes many states repeat. In the GridCraft benchmarks, 44–90% of decisions
  were cache hits, and the mean decision cost was 10–56 ms. A GPU is needed for twitch games and for
  the hive at scale.
- int8 changes decisions a little at the margin (some `act` → `escalate` on these zero-shot texts).
  The fine-tuned checkpoint is therefore calibrated **in int8**, as it is run.
- Zero-shot, every checkpoint picks `engage` even in the critical-combat text (2 hp, three
  enemies): confirmation that Laya is useless here until fine-tuned (Architecture §5).
