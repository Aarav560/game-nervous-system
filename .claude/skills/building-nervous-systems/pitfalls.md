# Pitfalls reference (measured in this repo; numbers in docs/results/)

## Simulators
- Resolve simultaneous actions together; never in agent-dict order (the bot "won" 100% because it was processed first).
- Identical scripted fighters lock-step → add seeded noise (whiff 10%) and check mirror ≈ 50% / no 100% double-KOs.
- Movement into obstacles needs sliding; shooters must not fire without line of sight (wasted mags looked like "bad AI").
- Label results "<sim name>", never the real game's name.

## Data
- Dedupe on (state text, options); cap per (option set, label) or `continue` floods everything.
- Split by episode/seed, never by row (near-duplicate states leak).
- Minority labels (heal 32%, reload 0% with 6 examples) need targeted data, not more epochs.
- 20% exploration in rollouts gives recovery states (DAgger); keep 1/3 clean rollouts.

## Calibration
- Fit on ≥ 300 representative items; pool new data with a fixed anchor slice of the original held-out set.
- `laya.train.finetune` refits the checkpoint's internal temperatures on its own slice (`calib_frac`); keep it ≥ 10–20% / 250+ items.
- Calibrate and evaluate the variant you deploy (int8 vs fp32 differs at the margin).

## Evaluation
- Gate on live rollouts vs brain-only on the same seeds; tolerance ≥ 1.5 episodes at n=10.
- Pool rates over time (total tokens ÷ total hours); unweighted per-scenario means let 5-second episodes dominate.
- Spot checks (DAgger labels) are a training cost — turn them off when measuring runtime brain usage.
- When two different policies score identically, diff their actions: identical trajectories = they converged to the same plan, not a harness bug.

## Sleep cycles
- Promote only vs the champion: gate vs brain baseline **and** vs champion, escalations ≤ champion × 1.05.
- Replay ≥ 1k old rows, half from the hardest context (combat), or the model forgets it.
- One night ≈ 50 min on a 4-vCPU CPU; budget accordingly or use a GPU.

## Adaptation
- One network fed "opponent style" features may ignore them (adaptive 78.1% = non-adaptive 78.1%).
- Counter-picking among existing policies by expected win under the style posterior worked (85.8%); pick by expected value, not argmax style, or misreads hurt.
- Adaptation can only choose among strategies that exist — if no policy beats a style, train one.

## Infra
- One Laya training at a time (~4.8 GB RSS). Background waits: set long timeouts; poll process exit, not log lines.
- Real-time adapters (Mineflayer bridge): localhost + token, pathfinder fallback, kill-switch drill before every run.
