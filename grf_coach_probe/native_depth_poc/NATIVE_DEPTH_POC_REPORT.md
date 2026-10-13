# GRF native defensive-depth control: proof-of-concept report

Date: 12 Oct 2026. GRF 2.10.3 (commit `3d9e754`). No RL was trained, no reward changed, no other tactical control exposed, no player-movement override used (all 11 left players on `builtin_ai` in every run). The original build (`~/grf_coach_probe`, `git status` clean, 0 modified files) and all earlier project files, including the VMAS and RoboCup work, are untouched. The patched build lives only in `~/grf_native_poc` (WSL).

## Two separate conclusions

| Question | Verdict |
|---|---|
| **A. Engineering:** can Python change a native `TeamAIController` positioning parameter live while the built-in AI keeps playing? | **PASS** |
| **B. Tactical learning:** is native defensive depth a usable action for a high-level RL coach (is there a reliable, state-dependent advantage)? | **FAIL, not RL-training ready** |

## 1. Interface change (36 added lines, 5 files; exact diff in `engine_patch.diff`)

One parameter only: `position_defense_depth_factor` (the team's length when out of possession).

| File | Change |
|---|---|
| `src/onthepitch/teamAIcontroller.hpp` | new member `float coachDefenseDepth = -1` (override, <0 = none); `SetCoachDefenseDepth(float)`; inline read-back `GetLiveDefenseDepth()` |
| `src/onthepitch/teamAIcontroller.cpp` | `SetCoachDefenseDepth` clamps to [0,1] and writes the live value immediately; `UpdateTactics()` re-applies the override after rebuilding live tactics, so the engine's 1000 ms refresh does not overwrite it. Clearing does **not** call `UpdateTactics()` (see "bug found" below) |
| `src/game_env.hpp` / `.cpp` | `GameEnv::set_coach_defense_depth(team_id, value)` and `get_live_defense_depth(team_id)` via `GetGameTask()->GetMatch()->GetTeam(id)->GetController()` |
| `ai.cpp` | two `.def(...)` lines on `GameEnv` |

The override is deliberately **not** in `ProcessState`, so existing snapshots (including your saved states) still load. Python use: `env.unwrapped._env._env.set_coach_defense_depth(0, 0.95)` (team 0 = left, value in [0, 1], `-1` clears). Native default is 0.71 for both teams.

Bug found and fixed during the work: my first version called `UpdateTactics()` when clearing, which perturbed the engine's schedule, so the cleared patched engine did not match the original. Removed, rebuilt, regression then matched exactly.

## 2. Runtime evidence (Stage 1; `stage1_results.json`, `full_match_*.json`)

- **Applies live, to the right team only:** left live value 0.71 to 0.95 immediately after the call; right team stayed 0.71 for 300 steps.
- **Persists:** 30 samples over 300 steps (about 30 native update cycles), all exactly 0.95. Not overwritten by the engine's heuristics.
- **Repeated commands harmless:** setting every step vs once gave trajectories with max difference 0.0, identical run time (0.92 s vs 0.92 s per 120 steps). Same command twice in one process: identical.
- **Native AI intact:** with the override cleared, the patched engine reproduces the original engine's trajectories exactly (max difference 0.0 on all 6 test states). Full 3,001-step matches ran to the end at every setting (22 to 23 s each). Corrected pass counter (measurement-only change): **29 / 33 / 29 left-team passes** for depth default / 0.0 / 1.0, with 1 to 8 possession gains and 4 to 7 losses, i.e. normal passing, tackling and shooting continue.
- **`set_state` does not restore the override.** It restores the live value from the snapshot, but the override member persists independently: restoring a default snapshot while an override was active gave 0.71 immediately and 0.95 again after 15 steps; restoring an override snapshot after clearing gave 0.95 immediately, reverting to 0.71 after 15 steps. **Therefore the parameter is set (or cleared) explicitly after every `set_state`** in all paired rollouts.
- **Engine nondeterminism (unpatched too):** for 2 of 6 test states the *original* engine gives one of two different trajectories depending on the process (identical in the patched engine). So paired rollouts use 4 replicate processes per set and pair within each process.

## 3. Measured effect on team shape (live positioning really changes)

Mean over opponent-possession frames, pooled across rollouts (dev 21 states, held-out 12 states). Team length = max minus min x of the 10 outfield players; back line = mean x of the 4 deepest (our goal at x = -1).

| Setting | Team length dev / held | Back-line x dev / held |
|---|---:|---:|
| 0.0 | 0.434 / 0.429 | -0.377 / -0.258 |
| 0.35 | 0.456 / 0.435 | -0.383 / -0.339 |
| default (0.71) | 0.577 / 0.591 | -0.382 / -0.371 |
| 0.95 | 0.704 / 0.659 | -0.482 / -0.412 |
| 1.0 | 0.697 / 0.678 | -0.468 / -0.343 |

Dev paired differences vs default (95% bootstrap CI over states): team length 0.0: -0.143 [-0.186, -0.096]; 0.95: +0.127 [+0.096, +0.157]; back line 0.95: -0.101 [-0.167, -0.037]. Full-match check agrees (length 0.39 / 0.54 / 0.69 for 0.0 / default / 1.0). The parameter is **monotone in team length, but it is a length control, not a clean line-height control**: the back line moves deeper only at the high end because the offside-trap clamp (`teamAIcontroller.cpp` backXBound) caps it.

## 4. Outcome comparisons (Stage 2)

Design: restore each saved opponent-possession state, set depth explicitly, play 200 steps with `builtin_ai` for all 11 players, 4 replicate processes, settings {default, 0.0, 0.35, 0.95, 1.0}. **Pre-declared primary metric:** opponent progress toward our goal before first recovery (lower is better); secondary: recovery time (censored at 201), dangerous-ball entries, goals conceded. Preference rule fixed on dev only, then applied unchanged to held-out. Development = 21 states (`tactical_states_independent` + `tactical_states_grounded`, 7 per category); held-out = 12 states (`tactical_states_holdout_minimal`, 4 per category). Between-replicate noise is small (SD 0.008 progress dev, 0.003 held), so uncertainty is dominated by the few states. CIs are bootstrap over states and are wide.

**Development, difference vs default (mean [95% CI]):**

| Setting | Opp. progress | Recovery steps | Goals conceded |
|---|---|---|---|
| 0.0 | +0.035 [-0.081, +0.149] | +24.7 [-4.1, +55.5] | +0.10 [-0.07, +0.27] |
| 0.35 | +0.126 [-0.038, +0.301] | +34.5 [-3.1, +74.0] | +0.19 [-0.05, +0.43] |
| 0.95 | +0.058 [-0.023, +0.146] | +18.6 [-5.7, +46.3] | -0.10 [-0.24, 0.00] |
| 1.0 | +0.071 [-0.018, +0.169] | +5.1 [-18.7, +27.8] | +0.06 [-0.14, +0.26] |

Every interval includes zero; all non-default settings are directionally worse on the primary metric. The native default (0.71) is the best fixed setting on dev.

**Held-out, difference vs default:**

| Setting | Opp. progress | Recovery steps | Goals conceded |
|---|---|---|---|
| 0.0 | -0.127 [-0.306, +0.001] | -16.3 [-40.0, +6.0] | 0.00 |
| 0.35 | -0.076 [-0.254, +0.139] | +1.6 [-34.7, +44.1] | +0.17 [0.00, +0.42] |
| 0.95 | +0.026 [-0.068, +0.129] | +10.1 [-28.6, +49.3] | +0.08 [0.00, +0.25] |
| 1.0 | +0.016 [-0.080, +0.107] | +14.0 [-21.5, +44.1] | +0.08 [0.00, +0.25] |

**State-dependence test (the RL gate).** Dev-preferred setting by category on the primary metric: midfield = 0.0, penalty area = 0.95, wide = default. Applied to held-out: category policy minus default = **-0.108 [-0.287, +0.018]** (lower is better; CI includes 0). Dev's best fixed setting is the default itself, so the category policy minus best fixed is the same -0.108 [-0.288, +0.018]. Preferences are not stable: penalty-area preference was 0.95 on dev but 0.35 on held-out, and depth 0.0 was worse than default on dev (+0.035) yet better on held-out (-0.127). The only repeated hint is midfield preferring the compact setting (7 dev and 4 held-out states); it is not statistically supported.

## 5. Verdict against the stop criteria

| Criterion | Result |
|---|---|
| Interface requires major engine redesign | No (36 lines, one rebuild of about 2 minutes incremental) |
| Python changes affect live positioning | **Yes** (team length 0.43 to 0.70) |
| Native AI disrupted | **No** (matches identical to original at default; passes, tackles, shots normal) |
| Parameter persists | **Yes** |
| Meaningful state-dependent tactical advantage | **No**: no setting beats default with a CI excluding 0 on dev or held-out; preferences flip between dev and held-out |

**RL-training readiness: FAIL.** A single native team parameter that reshapes the whole team while the built-in AI keeps playing is a valid engineering result, but at this sample size and horizon the outcome effect is not distinguishable from zero, and the engine's hand-tuned default is already the best fixed setting. Training RL on it would be learning from noise.

## 6. Limitations

- 21 dev and 12 held-out states (7 and 4 per category), 200-step horizon: low power; CIs are wide and a real effect of moderate size would not be detected. The held-out compact-setting gain (-0.127, CI to +0.001) is suggestive but contradicted by dev.
- Only one parameter; the `userProperties` route would cap swings at +/-0.1, so I used a direct live override across the full [0, 1] range.
- States are opponent-possession snapshots, not whole-match outcomes; goals conceded are rare (0 to 0.29 per state).
- Engine process-level nondeterminism (also in the original) was handled by replicates, not removed.
- Depth controls team length, not line height directly.
- The full-match runs are one match per setting, only to show normal operation, not an effect estimate.

## 7. Reproduce (WSL, py39 env at `~/grf_coach_probe/py39`)

```bash
# isolated copy + patch + build (never touches the original)
git clone ~/grf_coach_probe/football ~/grf_native_poc/football
python3 apply_patch.py ~/grf_native_poc/football/third_party/gfootball_engine
cp build_poc.sh ~/grf_native_poc/ && ~/grf_native_poc/build_poc.sh     # builds libgame.so and ~/grf_native_poc/overlay

# run anything against the patched engine only via the overlay
export PATH=~/grf_coach_probe/py39/bin:$PATH LD_LIBRARY_PATH=~/grf_coach_probe/py39/lib
export PYTHONPATH=~/grf_native_poc/overlay PROBE_DIR=<project>/grf_coach_probe
python stage1_verify.py stage1_results.json
./run_stage2.sh tactical_states_independent,tactical_states_grounded dev      # 4 replicates, ~6 min
./run_stage2.sh tactical_states_holdout_minimal held
python analyze_stage2.py dev
python analyze_stage2.py held results/analysis_dev.json                        # dev-fixed preferences
./run_full.sh                                                                  # three full matches
# regression vs original: python regression_dump.py a.npz with and without PYTHONPATH, compare
```

Files (all in `grf_coach_probe/native_depth_poc/`): `apply_patch.py`, `engine_patch.diff`, `build_poc.sh`, `poc_lib.py`, `stage1_verify.py`, `stage2_paired.py`, `analyze_stage2.py`, `full_match_check.py`, `regression_dump.py`, `run_stage2.sh`, `run_full.sh`, `results/*.json`.
