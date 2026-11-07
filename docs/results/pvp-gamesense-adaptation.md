# PvP replay dataset, game sense and play-style adaptation (GridCraft)

You asked for three things: a large dataset of the best PvP "footage", training game sense, and
adapting to the enemy's play style. Everything here is in GridCraft, this repo's own
simulator. No real Minecraft or third-party video was used (see the note at the end).

Raw data: [`pvp-adaptive.json`](pvp-adaptive.json). Reproduce:

```
python scripts/pvp_dataset.py --rounds 500 --out data/pvp-v1   # about 8 min, 1 CPU thread
python scripts/train_adaptive.py --data data/pvp-v1            # about 7 min
python scripts/eval_selector.py 150
```

## 1. The dataset (`data/pvp-v1`, not committed: 50 MB)

| | |
|---|---|
| Rounds | 12,000 = 4 player policies × 6 opponent styles × 500 |
| Frames (20 Hz ticks) | 1,774,246 |
| Per frame | 24-d combat observation, 8-d opponent-habit tracker, action (move/sprint/attack/eat), auxiliary labels (hit / got hit within 0.5 s), opponent style, round outcome |
| **Best footage** | 4,598 rounds / 715,798 frames: won rounds in the top half by damage efficiency (dealt / taken), chosen per style |
| Variety | Random kits (1–3 golden apples, 30% without a totem), random start offsets |

**Opponent styles** (`adapters/gridcraft/styles.py`):

| Style | Behaviour |
|---|---|
| `practice` | The Phase 4 practice bot |
| `aggro` | Rushes and never retreats |
| `kiter` | Stays at the edge of reach and darts in on cooldown |
| `healer` | Disengages early and eats a lot |
| `counter` | Waits for you to swing, then punishes |
| `erratic` | Unpredictable movement |

**Player policies:**

| Player | Behaviour |
|---|---|
| `reflex` | Learned combat reflex, greedy |
| `reflex_sampled` | Learned combat reflex, sampled |
| `pure_scripted` | Scripted engage, no retreat |
| `expert_heal` | Scripted engage, retreats to heal |

## 2. Game sense

| Model | Inputs → output | Held-out result |
|---|---|---|
| **Win model** (`checkpoints/win-model.pt`) | fight state + opponent habits → P(I win this fight) | **AUC 0.91** overall, 0.94 once 2 s into the fight; Brier 0.086 vs 0.153 for the base rate |
| **Opponent model** (`checkpoints/opponent-model.pt`) | last ~2 s of the opponent's habits → style | **91.5%** after 1 s of observation (chance 16.7%); 35% at <1 s, 83% at 1–2 s, 91% at 2–4 s, 99% after 8 s |

Style recognition by opponent: kiter 100%, healer 95%, aggro 92%, counter 86%, practice 77%,
erratic 76%.

In the stack, `--game-sense checkpoints` adds `PRED win_prob` and `opp_style` to the state text.
Brain playbook v2 then uses `win_prob` to decide between fighting on and retreating to heal,
the weak link found in Phase 4.

## 3. Adapting to the enemy's play style

Two approaches were tried. All numbers are win rates over 150 rounds per style on held-out seeds.

| Opponent style | Learned reflex | Pure scripted | Expert (heal) | Adaptive network | **Counter-pick selector** |
|---|---|---|---|---|---|
| practice | 93% | 90% | 50% | 93% | 90% |
| aggro | 37% | 34% | 31% | 37% | 37% |
| kiter | 87% | 81% | 100% | 87% | **100%** |
| healer | 99% | 96% | 91% | 100% | 99% |
| counter | 89% | 92% | 100% | 91% | **100%** |
| erratic | 63% | 87% | 87% | 61% | **89%** |
| **mean** | 78.1% | 80.0% | 76.4% | 78.1% | **85.8%** |

**(a) Adaptive network: no gain.** This is one policy whose input also includes the opponent
tracker and style posterior. It was behaviour-cloned on the best footage, then trained with PPO
and league play across all styles. It identifies the style in about 1 s, but its mean win rate
equals the non-adaptive reflex. Feeding it blank style inputs costs only 0.2 points, so it barely
uses them. Against `aggro`, the reflex and the adaptive network play **move-for-move identical
rounds**: both converged on "sprint in, swing on every cooldown", and against a rusher each round
is decided by random whiffs and crit timing.

**(b) Counter-pick selector: +7.7 points over the reflex, +5.8 over the best single policy.**
`gamesense.StyleSelector` watches the opponent for 1 s. It then uses the policy with the highest
expected win rate under the style posterior, looked up in the dataset's win table. It picks
`expert_heal` against kiters and counters (100%), and `pure_scripted` or `expert_heal` against
erratic opponents (89% vs the reflex's 63%). Picking by expected value rather than by the single
most likely style fixed an early version's mistake: it misread the practice bot and switched
to the policy worst against it (83% → 90%).

**Still unsolved: `aggro`** (37% for every policy). None of our policies has a counter to a
pure rusher, and adaptation can only choose among strategies that exist. Training a dedicated
anti-rush policy (e.g. a PPO run against `aggro` only, with the selector routing to it) is the
obvious next step.

## Why not real players' videos?

Short version: video has pixels but not the inputs or exact game state the reflex learns from,
other people's videos need their permission, and this machine has no GPU. The workable route is
recording your own (or consenting players') games through the Mineflayer bridge, which logs exact
state and inputs; the full answer is in the conversation.
