# GRF coaching action space and RL design audit

Date: 12 Oct 2026. Read-only. Nothing was modified in the engine, the patched copy, `grf_rl_coach/` or any earlier report; no RL was trained and no benchmark was run. Evidence labels used throughout:

- **[TESTED]** run and measured in this project
- **[SOURCE]** confirmed by reading the code only
- **[HYPOTHESIS]** not implemented or not yet measured

Source paths are relative to `grf_coach_probe/football/third_party/gfootball_engine/src/` (call it `src/`). Line numbers are from the original (unpatched) tree unless stated.

## 1. Bottom line

1. **Keep the action space depth-only (3 actions: compact / native default / stretched) for the submission.** Every extra control costs a new patch, a validation round and a larger state-action table, and the real constraint is statistical power (see 5.3), not action richness.
2. **The native layer genuinely has more levers than depth.** All 14 `position_*` tactics are consumed by live positioning code [SOURCE]. Team **width** is the only plausible second control: same mechanism, same patch pattern, and it acts on a different axis. It is **not** yet tested and I would not add it unless depth results are strong and time remains.
3. **The RL formulation is sound and genuinely sequential** [TESTED, 10/10 environment tests]. The main weaknesses are statistical, not structural: per-match goal difference is very noisy and seed-pairing barely reduces that noise.
4. **Primary algorithm: tabular Q-learning (243 states). Simple baselines around it: native default, fixed compact, fixed stretched, hand-written coach, random-switching coach. DQN as the comparison. PPO: not recommended. SARSA: optional, nearly free.**
5. **Main risk to 23 October is compute heat and sample size, not code.** Plan for roughly 1,200 training matches per algorithm and 200 held-out matches per policy.

## 2. Inventory of native tactical controls (Task 1)

Common facts [SOURCE]: tactics live in `TeamAIController` (`src/onthepitch/teamAIcontroller.hpp:84-88`, constructor values `.cpp:56-69`, caps `.cpp:72-81`). `UpdateTactics()` (`.cpp:1221-1280`) runs every 1000 ms of match time (`:104`), which is every **10 engine steps**. It rebuilds `liveTeamTactics` from base values plus the team's `userProperties` (`teamdata.cpp:90-111`, `:264-271`), then recomputes `offensivenessBias` (`:1250`). All consumers sit in `GetAdaptedFormationPosition()` (`.cpp:321-475`), called from the defence, midfield and attack off-ball strategies (`strategies/offtheball/default_def.cpp:30-56`, `default_mid.cpp:34-71`, `default_off.cpp:33-34`). On-ball logic (pass, shot, dribble, tackle) is separate and untouched by these parameters, so **native built-in AI stays active for all of them**. None of the 14 parameters is dead: each is read at `.cpp:338-351` and then used as listed below. All of them act through `possessionBias` (`.cpp:367-370`), which blends the "offense" and "defense" variant, so a `defense_*` parameter only matters in proportion to `(1 - possessionBias)`.

Cap by `userProperties` (`teamAIcontroller.cpp:72-81`, `:1258-1278`): offset = (user − 0.5) × 2 × multiplier. Multipliers: depth ±0.1, width ±0.1, midfield focus ±0.3, side focus ±0.1, micro focus ±0.15. Two parameter groups have **no multiplier**, so they are inert through `userProperties`: `*_ownhalf_factor` and `*_midfieldfocus_strength`. The patch writes the live value directly, bypassing this cap.

| Parameter (offense/defense pair) | What it changes (code) | Team-wide | Python today | Mid-match | Heuristic counteracts? | Verified | Measurable? |
|---|---|---|---|---|---|---|---|
| `position_defense_depth_factor` | team length: `adaptedDepth = 0.45 × (off×pb + def×(1−pb))`, back/front bounds (`:376,:407-408`); per-role offsets `mixup` (AM, CF +0.125, `:270-317`); back bound clamped by offside trap (`:418-422`) | yes | **yes, via our patch** (`ai.cpp:152-153` in copy) | yes, set immediately, persists | no; override re-applied inside `UpdateTactics` (`.cpp:1280`) | **[TESTED]** length 0.43 / 0.58 / 0.70 for 0.0 / default / 1.0 | length yes; outcomes **not significant** |
| `position_*_width_factor` | `adaptedWidth = 0.95 × (...)` (`:378`) sets the y-bounds `lowYBound/highYBound` (`:409-410`) and scales `centerY` (`:403`); CB/LB/RB offense width +0.2 (`mixup`) | yes | no | yes if patched (same pattern) | no (same mechanism) | **[HYPOTHESIS]** (source says it narrows or widens the lateral spread; not measured) | lateral spread likely; outcomes unknown |
| `position_*_ownhalf_factor` | `offsetX` shifts the team's centre toward own or opponent half (`:381-384`) | yes | no | yes if patched | no | **[SOURCE]** | probably (it moves the whole block), but overlaps with depth |
| `position_*_midfieldfocus` | stretches midfielders toward defending or attacking position (`AIfunctions.cpp:57-71`; `:456-463`, blended 70/30 with an automatic ball-following term) | midfield only | no | yes if patched | partly (30% automatic follow-ball) | **[SOURCE]** | midfield position only |
| `position_*_midfieldfocus_strength` | strength of the above | midfield only | no | patch only (no multiplier route) | no | **[SOURCE]** | as above |
| `position_*_sidefocus_strength` | how far the block centre follows the ball vs the long-run possession side (`:386-394`); additionally shifted by `offensivenessBias` (`:355-356`) | yes | no | yes if patched | **yes, partly**: the bias term is recomputed every second | **[SOURCE]** | shift of block toward ball side |
| `position_*_microfocus_strength` | pull of individual players toward the ball/focal point (`:441-450`, `AIfunctions.cpp:97-130`); scaled by an automatic factor (`:446-448`) | yes | no | yes if patched | partly (automatic scaling) | **[SOURCE]** | closing-down intensity; plausible but entangled |
| `offensivenessBias` | defensive hold line `startDistance = 30 + 20×bias`, fall-back `offsetX = 20 + 10×(1−bias)` (`:108,:118`), side focus (`:355-356`), `possessionBias` (`:370`); computed from score, match time, 60 s possession (`:1246-1250`) | yes | no | yes (needs an additive coach term, because it is recomputed each second) | **yes: built-in comeback/lead logic already uses it** | **[SOURCE]** | line height plausible, but competes with the native logic |
| `dribble_offensiveness`, `dribble_centermagnet` | on-ball dribble direction (`AIfunctions.cpp:249,274`) | all ball carriers | no | via `userProperties` | no | **[SOURCE]** | affects on-ball play, not positioning; not a "tactic" in the sense we want |
| Roles / mind-set (`AI_GetMindSet`, `AIfunctions.cpp:1196-1218`) | pass weights, hunting range, opponent repulsion, which strategy (def/mid/off) runs (`elizacontroller.cpp:313-333,835-847,424-429`) | per player | only at episode start via `FormationEntry` (`ai.cpp:206-212`, role is read-only in Python) | not without patch; not in `set_state` | no | **[SOURCE]** | potentially large, but it bundles position and strategy, hard to interpret |
| Formation positions | static home positions (`teamdata.cpp:205-250`, `team.cpp:117-138`) | yes | **episode start only** via scenario `left_team` | no | no | **[SOURCE]** | yes, but pre-match, not in-match |
| `*_team_difficulty` | scales every player's stats, reaction time, hunt range (`player.cpp:452-456`, `playercontroller.cpp:64`) | yes | yes | no (`const`) | n/a | **[SOURCE]** | skill nerf, not tactics; do not use |
| `team_pressure` action | 1 player, 0.5 s (`teamAIcontroller.cpp:1185-1201`) | no | yes (`action_set='full'`) | yes | n/a | **[SOURCE]** | weak; issuer loses native AI |

## 3. Ranked candidate actions (Task 2)

Scored 1 (poor) to 5 (good). "Native AI preserved" is 5 for all position-only controls.

| Option | Ease | Native AI preserved | Positional influence | State-dependence potential | Measurability | Compute | Coaching relevance | Disruption risk | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A. Depth only** (compact / native default / stretched) | 5 (done) | 5 | 3 (team length, back line deeper at the high end) | 3 | 4 (shape verified) | 5 | 4 (compactness vs stretched block) | low | **Recommended** |
| **B. Width only** | 3 (about 15 lines, one rebuild, a new validation round) | 5 | 3 (lateral spread, [HYPOTHESIS]) | 3 | 3 | 5 | 4 (narrow vs wide block, a standard coaching choice) | low-medium (extreme low width collapses the team onto a line) | best second control, **not recommended before depth results** |
| **C. Depth + width** (9 combinations, or 3 + 3 factored) | 2 | 5 | 4 | 4 | 2 (effects confounded, 9 arms to separate) | 3 | 4 | medium | no: tabular table triples, power worsens |
| **D. Offensiveness / midfield focus** | 2 (offensiveness needs an additive term; midfield strengths need patch) | 5 | 2-3 | 3 | 2 | 5 | 3 | medium (counteracts native comeback logic) | no |
| **E. Formation / roles** | 1 | 4 (changes strategy class too) | 4 | 4 | 1 | 5 | 5 (very realistic) | high (not in snapshots, bundled effects) | no, out of scope for this deadline |

Recommended smallest useful action space: **3 actions**: compact (override 0.0), **native default (clear the override, not a numeric neutral value)**, stretched (override 1.0). **[TESTED]**: clearing leaves trajectories bit-identical to the unmodified engine, and the numeric 0.71 also matched in the one seed tested, but only the clear is the "genuine native" definition. Caveat [SOURCE]: AM and CF roles keep a +0.125 role offset on depth even when overridden (`mixup`, `:310-317`), so 0.0 and 1.0 are the extremes of the base, not exact zero and one for those two roles.

## 4. RL formulation audit (Task 3)

Audited the files that exist in `grf_rl_coach/` (`coach_env.py`, `rollout.py`, `policies.py`, `train_q.py`, `train_dqn.py`, `evaluate.py`, `tests/test_env.py`, `results/env_tests.json`). **Not audited, because it has not run to completion:** DQN training, held-out evaluation, final tables. Q-learning training was paused at iteration 41 of 100 (492 matches), so no learning conclusion exists yet.

| Item | Finding | Status |
|---|---|---|
| Sequential, multi-decision | 30 decisions per 3001-engine-step match (the match really has 3001 steps; a 1-step stub is absorbed into the last interval) | **[TESTED]** |
| Interval vs native update cycle | decision every 100 engine steps; native tactics refresh every 10, so each decision spans 10 native refreshes. Clearing returns to native within 1 step, setting is immediate | **[TESTED]** appropriate |
| Observation | 14 bounded features from verified raw fields (ball, possession, score, steps left, our shape, opponent front line); previous action included, so persistence is observable | **[TESTED]** valid and finite |
| Omitted information | opponent shape beyond front line, fatigue (`*_tired_factor` exists in raw obs), per-player roles | acceptable; not needed for a first policy |
| Rewards | raw goal difference per interval, plus potential-based shaping `0.1·(γΦ(s')−Φ(s))` with Φ = ball x and terminal Φ = 0 | **[TESTED]** accounting (`Σ r_goal` equals final score diff) |
| Shaping risk | policy-invariant in theory, but with γ per decision and finite training it can steer toward territory rather than goals. Evaluation uses raw goals and also reports shaped return separately. **Recommended: report one ablation with shaping 0 if time allows; never claim improvement from training return** | partly mitigated |
| Discounting | γ applied per coach decision (not per engine frame), consistent in TD target and shaping; 0.99⁳⁰ ≈ 0.74 so far-future goals are weighted down mildly. Acceptable; document that γ is per decision | **[SOURCE]** consistent |
| Terminal handling | match end is terminal with time remaining in the observation (finite horizon), so no bootstrap past the end is correct | **[TESTED]** |
| Reset / `set_state` | `set_state` restores the live tactic value but **not** the coach override; the environment re-applies it explicitly after restore; control test confirms the revert otherwise | **[TESTED]** |
| Determinism | a match is a **deterministic function of its seed** once the episode number is pinned to the seed (the scenario alternates kick-off side by episode parity, which earlier looked like random modes). Identical seeds reproduce across processes | **[TESTED]**. Different action sequences diverge chaotically within a few steps |
| Reward exploitation | no reward for changing tactic or choosing a setting; cannot gain reward without changing field position or goals. Remaining exploit surface is the territory potential | acceptable |

**Recommended fixes (suggestions only, not applied):** (1) add the shaping-0 ablation and a uniform-random-switching baseline; (2) keep checkpoint selection on dev seeds only; (3) record the opponent's kick-off parity (seed parity) as an analysis covariate because the team that kicks off differs by parity; (4) log the number of default-action decisions to detect a degenerate "always default" policy.

## 5. Algorithms and experiment design (Tasks 4 and 5)

### 5.1 Algorithms

| Algorithm | Appropriate? | Observation | Complexity and key hyperparameters | Risk |
|---|---|---|---|---|
| **Tabular Q-learning (primary)** | yes: discrete 3 actions, coarse state, interpretable table, cheap | discrete id (zone × possession × score × phase × last action = 243) | α 0.1, γ 0.99, ε 0.5→0.05, ~1,200 matches = 36k transitions, about 50 per state-action | low; 243×3 table is well covered by 36k samples; risk is noise, not code |
| **DQN (comparison)** | yes, as a deep-RL comparison on the same 14 features | continuous 14-d vector | 64-64 MLP, lr 1e-3, buffer 50k, target sync 200, Huber loss; same episode budget | medium; with only 36k noisy transitions it may not beat the table. Report either outcome honestly |
| **PPO** | not recommended | continuous | needs fresh on-policy batches, high variance with 1 reward-bearing event per ~10 decisions | high; extra complexity with no expected benefit at this scale |
| **SARSA** | optional | discrete | same data collection as Q-learning, 3-line change | very low; only worth it as an extra baseline if time remains |

Primary: tabular Q-learning. Simpler baselines: native default and hand-written coach. The assessment does not require deep RL.

### 5.2 Baselines and metrics

Evaluate on identical held-out seeds: **native unmodified (no setter calls)**, fixed compact, fixed stretched, **hand-written situation coach** (declared before results), **uniform random switching** (control for "does switching itself hurt"), Q-learning, DQN, and the **observation-shuffled** version of each trained policy (already implemented in `evaluate.py`). Metrics: goal difference (primary), goals for/against, points, possession share and gains (recovery rate), time to recovery (needs the per-interval logs, available from the state-restore evaluator used earlier), danger-zone steps and furthest opponent progress, raw and shaped return, action distribution, W/D/L.

### 5.3 Power and budgets (measured, not assumed)

- **Speed [TESTED]:** about 22 to 25 s per full match per worker (about 135 engine steps/s). Twelve workers produced one 12-match batch per 33 to 52 s; 41 iterations took 37 min. RAM: about 1.8 GB for 12 workers.
- **Noise [TESTED, small sample]:** on 12 matched seeds the per-match goal-difference SD was about 1.4 (native), 1.7 (compact), 1.0 (stretched). Seed pairing did **not** reduce variance: correlation of per-seed outcomes between native and compact was −0.13, native and stretched +0.04, SD of the paired difference about 2.3. Treat comparisons as essentially unpaired.
- **Required sample:** n ≈ 7.84 × σd² / Δ² with σd ≈ 2.3: to detect a goal-difference effect Δ = 0.5 about **170 matches per policy**; Δ = 0.3 about **470**; Δ = 0.7 about **85**. These are rough because σd comes from only 12 seeds.
- **Recommended budget:** train each algorithm on about 1,200 matches (about 36,000 decisions); **evaluate the primary policies on 200 held-out matches each** (about 1.5 to 2 hours total at 12 workers; double that at 6 workers). If time allows, run the key pair (Q-learning vs native) on 400.
- **Seed hygiene (already used):** train 100000+, dev 500000+ (6 seeds for learning curves and checkpoint choice only), final test 900000+ untouched until the end. Choose the checkpoint on dev seeds only. Run at least 2 independent training seeds for Q-learning if time allows, to show variability of the learned table.

### 5.4 Deciding "genuine improvement" versus "learned the default"

Call it an improvement only if **all** hold: (a) paired-or-unpaired bootstrap CI for goal difference vs native excludes 0 on the held-out set; (b) the policy is not simply native (report the fraction of default decisions and how it varies by state); (c) the observation-shuffled copy is significantly worse than the trained policy; (d) it does not lose to the hand-written coach on raw goals. If the CI includes 0 or the greedy policy is about 100% default, report it as "learned to approximately match native" — a valid result. The previous depth study (nothing beat default with CI excluding 0 on 21 dev and 12 held-out states) makes a null result a realistic prior.

## 6. Minimum viable plan and deadline risk (Task 6)

Recommended observation, action and reward: as in `coach_env.py` (14 features, 3 actions, raw goal difference plus small potential-based territory shaping; evaluate on raw goals only).

| Date (2026) | Step |
|---|---|
| 12 to 14 Oct | finish Q-learning (59 iterations remain); run DQN with the same budget |
| 15 to 17 Oct | held-out evaluation, 200 matches per policy; policy-dependence and default-fraction analysis |
| 18 to 19 Oct | demo recording, timeline figures, tables |
| 20 to 21 Oct | report and slides |
| 22 to 23 Oct | buffer; nothing new may be started |

Risks, in order: **(1)** laptop heat limits workers (a pause was already needed); at 6 workers training and evaluation take about twice as long, so budget about 6 to 8 hours of compute in total. **(2)** a null result: acceptable academically if reported honestly with the controls above. **(3)** only 12 matches' worth of variance evidence; recompute σd from the first evaluation batch and size the rest accordingly. **(4)** scope creep: adding width or formation now would consume the buffer.

## 7. Recommendation

**Retain depth-only RL.** It is implemented, validated (10 environment tests, live-shape and native-AI regression tests, determinism per seed), and sufficient for a defensible sequential RL project. If, after the first held-out evaluation, the trained policy shows a clear state-dependent difference and more than 4 days remain, the only extension worth considering is a **width** control (same patch pattern, about 15 lines, then repeat the Stage 1 checks). Otherwise, spend the remaining time on more evaluation matches and the shaping-0 ablation, which add more to the argument than another action would.
