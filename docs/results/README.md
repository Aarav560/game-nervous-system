# Results

Everything below was produced in one session on a 4-vCPU CPU machine with **no GPU** and **no LLM
API key**.

- **The brain** in every run is the scripted stand-in (`ScriptedBrain` + per-game playbook). "Brain
  tokens" are a characters ÷ 4 estimate of what an LLM brain would have been sent.
- **Phase 1** ran on a real Minecraft-protocol server: local flying-squid via Mineflayer.
- **Phases 2–7** ran in GridCraft and Skirmish, this repo's own simulators.

| phase | exit criterion | verdict | page |
|---|---|---|---|
| 0 | Laya < 50 ms locally | ❌ best 76–93 ms on CPU (needs a GPU) | [phase0-laya-latency.md](phase0-laya-latency.md) |
| 1 | gather 64 logs + return, goals down / events up | ✅ real protocol server, 265 s | [phase1-minecraft.md](phase1-minecraft.md) |
| 2 | Laya ≥ 70% without escalating, no regression | ✅ 99.8%, 95% vs 70% success | [phase2-3-laya.md](phase2-3-laya.md) |
| 3 | brain tokens/h −50% at equal scores | ✅ −85% pooled (−24% unweighted per-scenario mean) | [phase2-3-laya.md](phase2-3-laya.md) |
| 4 | combat reflex ≥ 70% vs practice bot | ✅ 95% (a simple scripted reflex gets 92%) | [phase4-combat.md](phase4-combat.md), [reference study](phase4-reference-checkpoints.md) |
| 4+ | PvP dataset, game sense, style adaptation | 12k rounds; win model AUC 0.91; style identified 91.5% after 1 s; counter-pick 85.8% vs 78.1% | [pvp-gamesense-adaptation.md](pvp-gamesense-adaptation.md) |
| 5 | escalations fall over 3 sleep cycles, no regression | ❌ candidates got worse and were rejected; causes found | [phase5-sleep.md](phase5-sleep.md) |
| 6 | 5 bots, one brain, base building | ✅ 100%, claims halve the time | [phase6-hive.md](phase6-hive.md) |
| 7 | second game, basic benchmark, no tier 1–2 changes | ◐ adapter-only works (brain-only 47%); Laya regresses (13%) | [phase7-second-game.md](phase7-second-game.md) |
