# Phase 4 — study of the open Minecraft Combat AI checkpoints

Date: 2026-11-07. Purpose: Roadmap Phase 4, first item — "study the open checkpoints (load safely,
map features) as reference". **No weights from these repos are in this repository or in any build**;
they were downloaded to a scratch directory outside the repo and inspected only.

## How they were loaded

CLAUDE.md requires `torch.load(path, weights_only=True)` in an isolated process. Both files were
inspected with `python -I` from a scratch directory.

| Repo (Hugging Face) | Licence | File | `weights_only=True` result |
|---|---|---|---|
| `sdffewe/minecraft-combat-transformer` | GPL-3.0 | `newgen6.pt` (9.2 MB) | Refused (needs numpy globals). Loaded with `torch.serialization.safe_globals([numpy _reconstruct, ndarray, dtype, codecs.encode])`, still in weights-only mode. No code was run. |
| `sdffewe/crystal-v2-reinforcement` | CC-BY-3.0 (credit CRAZYON, CENTERX, Skorchekd) | `policy_best.pt` (4.2 MB) | Refused (it is a TorchScript archive). **Not loaded.** I read its zip entries and TorchScript code as plain text instead. |
| `sdffewe/minecraft-combat-MLP`, `sdffewe/mcpvpcrystalaimodel` | AGPL-3.0 | — | Not downloaded. |

`weights_only=False` was never used.

## Sword transformer (`newgen6`)

- 2.29M parameters. Config: `seq_len 96`, `d_model 192`, 5 layers, 6 heads, FFN 640, dropout 0.15,
  `arch: attn_pool+aim_pathway`. Input is 78 features × 2 (raw + normalised, `input_dim 156`), with
  saved `mean` and `std` (78 each).
- Trunk: `seq_proj` (156→192) + learned `pos` (96×192) → 5-layer transformer encoder → attention
  pooling (`pool_q`) + `current_trunk` (current frame) + `raw_skip` + `aim_proj` → `shared` MLP.
- Action heads: `forward_h` (3: back/none/forward), `strafe_h` (3), `pivot_side` (3), `pressure_mode` (4),
  and sigmoids `sprint`, `jump`, `attack_now`.
- **Auxiliary prediction heads** (sigmoid): `will_hit`, `will_be_hit`, `hit_without_hitback`,
  `opp_aim_lock`, `crit_spam_threat`, `crit_counter_break`, `dodge_then_hit`, `commit_punish`,
  `space_safe`, `combo_risk`, `air_juggle`, `flank_opening`, `escape_clean`, `counter_hit_risk`,
  `outspaced`, `walkdown_trap`, `mirror_trade_risk`, `force_field`, `aim_lost_punish`, `reset_jump`,
  `combo_confirm`, `entry_window`, `sprint_crit_incoming`, `jump_crit_window`, `aim_sweep`,
  `crit_turtle_failure`.
- The 78 features map onto our design like this:

| Their features | Our equivalent |
|---|---|
| `dist`, `rel_dx/dy/dz`, `d_dist`, `self_closing_speed`, `o_closing_speed` | `Facts` entity positions/velocities → `PhysicsForwardModel` |
| `s_hp`, `o_hp`, `s_attack_strength_pre`, `o_attack_strength`, `ms_since_attack_pre` | `SELF hp`, `OPP hp`, `cooldown`, `crit_threat` lines |
| `aim_err`, `on_target`, `opp_aim_err`, `opp_on_us` | `facing_me`; `PRED opp_aim_lock` |
| `s_sprint`, `o_sprint`, `s_fall`, `o_fall`, `s_ground` | sprint-crit logic (GridCraft has no vertical axis or jump-crits) |
| `s_ping`, `o_ping`, `ping_gap` | not modelled (local sim; real servers would need it) |

What we take from it: the auxiliary heads (`opp_aim_lock`, `sprint_crit_incoming`, `will_be_hit`,
`space_safe`) are the `PRED` features Laya should see. Our forward model v1 computes the first two
analytically, and the learned GridCraft reflex (Phase 4) adds `will_hit` / `will_be_hit` heads.

## Crystal policy v2 (`policy_best`)

TorchScript module `{trunk: Sequential(Linear, Linear, act, Linear, Linear, act), pi: Linear, v: Linear}`
— an actor-critic MLP (`pi` policy head, `v` value head), saved from `cuda:0`. It confirms the
reinforcement-learning recipe (PPO-style actor-critic) the README describes. Tensor shapes were not
read because that would mean loading the TorchScript archive.

## Licence consequences (Architecture §14)

- The GPL-3.0 transformer must never be bundled in a distributed build of this MIT repo. It isn't:
  nothing here loads it, and the learned GridCraft combat reflex is trained from scratch.
- Crystal-v2 (CC-BY-3.0) could be used with credit, but its features are crystal-PvP specific and it
  can only be loaded as TorchScript (`weights_only` refuses it), so we don't use it.
