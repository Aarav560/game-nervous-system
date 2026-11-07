# References

Figures below come from the linked write-ups and project pages as of November 2026. Re-verify before relying on them; several are vendor-reported.

## Laya (tier 2 model)

- Open-source, Apache 2.0, local stand-in for TypeSafe's Jev.
- Variants: Laya English (421M, ModernBERT-large, 512 ctx), Laya Multilingual (322M, mmBERT-base, 1,024 ctx up to 8,192), Laya Typed Decisions (421M, ModernBERT-large).
- Latency on a Tesla T4: ~39.5 ms (English) / ~32.8 ms (Multilingual) for one decision; batched throughput ~103–332 decisions/s.
- Base model weak zero-shot on typed decisions; fine-tuned checkpoint is the strong one. Probabilities can be overconfident — calibrate. Accuracy drops with large option sets.
- Hugging Face: `convaiinnovations`; GitHub: `NandhaKishorM`. Self-hosted server needs `LAYA_API_KEY` for auth.

Sources:
- [Jev vs Laya: Benchmarks, Self-Hosting & Key Differences (Knolli)](https://knolli.ai/post/jev-vs-laya)
- [Laya: A Free, Local Stand-In for TypeSafe's Jev (Towards AI)](https://pub.towardsai.net/laya-a-free-local-stand-in-for-typesafes-jev-if-you-use-it-for-the-right-job-11aa0903f011)
- [Jev vs Laya: Hosted Decision API vs Open-Source Local Model (ZimaSpace)](https://shop.zimaspace.com/blogs/product-comparisons/jev-vs-laya-decision-model)
- [decision laya (Unsloth docs)](https://unsloth.ai/docs/models/decision-laya.md)
- [jevos vs jev vs laya (GitHub wiki)](https://github.com/feder-cr/jev/wiki/jevos-vs-jev-vs-laya)

## Jev (hosted decision model)

- By TypeSafe AI (San Francisco), launched 15 Sep 2026. "System One" model: reads state, returns a typed answer from defined options with probability and confidence; does not generate text.
- Company-reported latency ~70–500 ms. Available via TypeSafe's API, OpenRouter and Vercel AI Gateway as `jev-latest`.

## Minecraft Combat AI (tier 3 reference design)

From the research note *Minecraft Combat AI & Inventory Mechanics — Research* (7 Nov 2026):

- Family of small neural policies (transformers, then GRUs) for sword, crystal, UHC and mace PvP; one observation in, one action out per 50 ms tick; ~1.8 ms local inference.
- Training: behaviour cloning → self-play RL → league training with exploiters.
- Auxiliary prediction heads (opponent aim lock, sprint-crit incoming, space safe, commit/punish) with high AUC on held-out opponents.
- Access: paid streaming API (terms ban training on its output and reverse-engineering the service); R1 Local Fabric mod (singleplayer/LAN only, "coming soon"); free older checkpoints on Hugging Face (`sdffewe`): sword transformer (GPL-3.0), crystal-v2 (CC-BY-3.0, credit CRAZYON, CENTERX, Skorchekd).
- Load `.pt` checkpoints with `weights_only=True` in isolation.

## Related work

- NotCrazyRocker/Combat-Ai — CC0 Fabric mod + PPO training scripts for a PvP policy.
- Voyager — LLM agent building a code skill library in Minecraft.
- Gemini agents on Mineflayer — LLM planning beating the Ender Dragon; illustrates "good at planning, slow at reacting".
- AlphaStar (DeepMind) — origin of league training.
- DAgger (Ross et al.) — dataset aggregation for imitation learning; basis of §12.2 distillation.
- Figure Helix / dual-system robot controllers — slow VLM planner over a fast policy; same split applied to robotics.
