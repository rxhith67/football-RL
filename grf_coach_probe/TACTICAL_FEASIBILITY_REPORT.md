# GRF high-level tactical coaching feasibility

Date: 2026-10-12. **Verdict: FAIL the RL-training gate for these three tactical actions.** GRF successfully executes a high-level instruction through sparse player overrides, but the final paired tests do not show a reliable advantage over leaving all players on built-in AI. No RL environment or agent was implemented or trained. The GRF engine and the existing VMAS, RoboCup and PITCHSIM projects were not modified.

## Translator that was tested

[`tactical_translator_py39.py`](tactical_translator_py39.py) turns each policy into the installed GRF V2 action set. [`tactical_action_ids.json`](tactical_action_ids.json) records the IDs directly from `/home/rohith_syam/grf_coach_probe/py39/lib/python3.9/site-packages/gfootball/env/football_action_set.py`: directions 1–8, sprint 13, release direction 14, release sprint 15, built-in AI 19. The observed Y axis is inverted relative to the engine's top/bottom labels; an initial pilot exposed this and was preserved as `tactical_pilot.json`. All final results use the corrected mapping.

| Instruction | Deterministic translation |
| --- | --- |
| Normal | Send `builtin_ai` to all 11 left players every frame. |
| Selective press | At opponent possession, select the two nearest outfield players to the right-side carrier; sprint once, then move them toward the carrier. |
| Defensive cover | Assign two outfield players to distinct near/deep points between the ball and the left goal using minimum total travel distance. |
| Hand-written adaptive comparator | Choose cover if ball x < −0.40 or absolute y ≥ 0.23; otherwise press. This rule was declared before the paired outcome comparison. |

At most two players are externally controlled. The goalkeeper and our ball carrier are never selected. The selected pair is retained during a continuous defensive engagement. Up to eight neutral-ball frames are treated as pass-flight grace; left possession sends **all 11 players directly back to `builtin_ai`** on that frame. This also resets the external direction/sprint inputs in GRF's native controller. A 12-frame cooldown prevents instant re-selection. Adaptive changes during a continuous possession require 30 frames of commitment. When a neutral-ball timeout or a tactical change ends an instruction, the translator sends `release_direction`, then `release_sprint`, then `builtin_ai`. GRF's native controller switches away from built-in AI for manually controlled players, so this translator does not give those selected players GRF's autonomous passing, shooting or defensive decisions during the override.

## Matched experiment

We used the existing `11_vs_11_hard_stochastic` scenario. [`collect_tactical_states_py39.py`](collect_tactical_states_py39.py) saved **12 natural opponent-possession states**, one of each type from four separate full matches: midfield (−0.30 ≤ x ≤ 0.30, |y| ≤ 0.18), near our penalty area (x ≤ −0.55, |y| ≤ 0.24), and wide (−0.50 ≤ x ≤ 0.15, |y| ≥ 0.23). Ball height was ≤ 0.20 and snapshots were taken after step 50. The [inspection output](tactical_states_inspection.json) confirms all 12 states had `game_mode=0`, opponent possession, and a carrier within 0.021 pitch units of the ball. The exact binary snapshots and manifest are in [`tactical_states_independent/`](tactical_states_independent/).

Each snapshot was restored for normal, press, cover and adaptive play for the **same 200-step horizon**: 48 rollouts. A normal rollout repeated after an intervening press rollout had an identical recorded trajectory and recovery time (`pairing_repeat_final_code.json`); a separate 50-step state-restoration check also had zero position difference. Results are grouped by snapshot, never by simulation frame as if frames were independent. Source data with sampled player/ball trajectories, targets, commands, override durations, score and possession is [`tactical_final_comparison.json`](tactical_final_comparison.json). The summary is [`tactical_final_summary.json`](tactical_final_summary.json), and the [recovery plot](tactical_final_recovery.png) visualises every paired state. Earlier pilot and pre-handoff results remain saved separately.

**First blue possession recovery, in engine steps.** `—` means no recovery within 200 steps; `*` marks an opponent goal within the horizon. A later goal can occur after an initial recovery, so recovery and goals are reported separately.

| State | Normal | Press | Cover | Adaptive |
| --- | ---: | ---: | ---: | ---: |
| Midfield 00 | 84 | 110 | 127 | 182 |
| Midfield 01 | 21 | **13** | 25 | **13** |
| Midfield 02 | **33** | — | 50 | 60 |
| Midfield 03 | 135 | **50** | 158 | **50** |
| Penalty 00 | **20** | 135 | 35 | 35 |
| Penalty 01 | 101 | 79 | **34** | **34** |
| Penalty 02 | **72** | 159 | — | — |
| Penalty 03 | **22** | —* | — | 139 |
| Wide 00 | —* | 182 | **100** | **100** |
| Wide 01 | 87 | 100 | **14** | **14** |
| Wide 02 | **10** | 49 | 171 | 171 |
| Wide 03 | **44** | — | 60 | 60 |

State-level preferences do differ: press recovered in 13 versus normal's 21 steps in Midfield 01; cover recovered in 34 versus normal's 101 in Penalty 01; normal recovered in 10 versus cover's 171 in Wide 02. These examples establish sensitivity, **not** a generalizable coaching advantage. The same broad zone can favor different actions (for example, Penalty 01 favored cover while Penalty 02 favored normal).

Across the four independent snapshots per broad situation, the count of states recovered within 200 steps was:

| Situation | Normal | Press | Cover | Adaptive |
| --- | ---: | ---: | ---: | ---: |
| Midfield | **4/4** | 3/4 | **4/4** | **4/4** |
| Near penalty area | **4/4** | 3/4 | 2/4 | 3/4 |
| Wide channel | 3/4 | 3/4 | **4/4** | **4/4** |

Cover's wide recovery count is a possible situational benefit, but each row contains only four states, and cover also had slower or worse outcomes in other wide states. The simple adaptive rule did not reliably improve on normal AI.

| Policy | Recovered / 12 | Mean recovery time, non-recovery counted as 201 | Mean opponent progress toward our goal | Goals conceded / 12 | Dangerous-ball proxy / rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | **11** | **69.2** | **0.304** | 1 | **0.83** |
| Press | 9 | 123.3 | 0.468 | 1 | **0.67** |
| Cover | 10 | 98.0 | 0.467 | **0** | 1.75 |
| Adaptive | **11** | 88.2 | 0.419 | **0** | 1.42 |

Opponent progress is the maximum movement of the ball toward the left goal before first recovery or concession; lower is better. The dangerous-ball measure is only a **proxy**: entry into x < −0.76, |y| < 0.22 while moving toward goal and not owned by blue. The raw observation does not label an attempted shot, so these counts must not be described as verified shot attempts. One versus zero goals across 12 states is too sparse for a reliable goals-conceded conclusion.

## Execution and ordinary play

- Across all 36 intervention rollouts, no goalkeeper was overridden, every rollout had positive mean movement toward its assigned targets, and at least nine left players received `builtin_ai` on every frame. For first engagements lasting at least 20 frames, the selected players were closer to their initial targets than the same players under normal AI in **18 of 22** paired checks. Full trajectories and selected-player frame counts are in the source JSON. Mean engagements per 200 steps were 3.17 press, 2.25 cover and 3.00 adaptive; mean engagement durations were 33.7, 47.4 and 35.4 frames respectively. The [final control trace](tactical_final_control_trace.json) shows press releasing immediately at blue possession in Midfield 01 (step 14), and a neutral-ball release sequence in Penalty 01 (direction at step 37, sprint at step 38, built-in AI at step 39).
- A full-length match was also run for each policy from the same kickoff snapshot. Results were: normal **1–2** at 135.7 steps/s; press **1–3** at 135.0; cover **0–1** at 140.6; adaptive **0–1** at 136.3 ([final match data](tactical_final_matches.json)). All matches reached 3,001 steps. Every policy had possession changes, all 11 players on each side moved, and the nine or more unselected blue players remained on built-in AI. Normal and press scored, confirming that native shooting continued among the AI-controlled players. There were 62, 37 and 55 short-gap teammate possession-transfer **candidates** under press, cover and adaptive; those candidates are not individually verified passes. The earlier isolated explicit pass check is in `pass_action_hold5.json`. One paired kickoff is only an execution check, not a reliable ranking of match outcomes.
- The unchanged normal policy was repeated from a saved state after an intervening press rollout and produced an exact trajectory match. Comparisons therefore share engine state, not merely similar-looking kickoffs.

## Interpretation and limit

The translator can make selected players run toward tactical targets, and choices have measurable, state-dependent consequences. The important training criterion is stronger: a policy with these actions must have a credible path to outperform normal built-in AI. In the 12-state sample, normal matched the adaptive recovery count, recovered faster, allowed less opponent progress and had fewer dangerous-ball entries. The single paired full match gives no reliable ranking of tactics. The adaptive rule did not beat the best fixed baseline overall. **No evidence here supports starting PPO or DQN training.**

The likely mechanism is that manual movement overrides suspend the selected players' built-in decision-making. This is verified at the controller-switch boundary in the engine source; the extent to which it reduces automatic interceptions is an inference from the resulting play, not a separately measured engine event. The sample is small (four source matches; 12 defensive situations), some rollouts share a source match, no explicit shot-attempt event is available, and reward based mainly on goals would be very sparse. A positive target-motion result alone cannot repair those outcome limitations.

## Reproduce the final checks

In PowerShell from the football workspace, with the [working isolated GRF installation](RUNTIME_REPORT.md):

```powershell
$probe = (Resolve-Path 'grf_coach_probe').Path
$linuxProbe = (wsl -d Ubuntu -- wslpath -a $probe).Trim()
$grfPython = '/home/rohith_syam/grf_coach_probe/py39/bin/python'

# Reuse the saved exact snapshots for the paired 12-state test.
wsl -d Ubuntu -- $grfPython "$linuxProbe/evaluate_tactics_py39.py" --states "$linuxProbe/tactical_states_independent" --policies normal,press,cover,adaptive --horizon 200 --out "$linuxProbe/tactical_repeat.json"
python grf_coach_probe/analyze_tactical_results.py grf_coach_probe/tactical_repeat.json --out grf_coach_probe/tactical_repeat_summary.json
python grf_coach_probe/verify_tactical_results.py

# One paired complete match per policy.
wsl -d Ubuntu -- $grfPython "$linuxProbe/full_match_tactics_py39.py" --out "$linuxProbe/tactical_full_repeat.json" --policies normal,press,cover,adaptive
```

The independent snapshots can be regenerated with `collect_tactical_states_py39.py --out <directory> --episodes 5 --per-category 4 --max-per-episode 1 --min-gap 100`; new stochastic matches will yield different snapshots and numerical results. The supplied `.state` files provide exact repeatability for this report. Existing successful GRF runtime results and other football projects remain intact.

The optional plot was generated by [`plot_tactical_results.py`](plot_tactical_results.py) with host Python and Matplotlib 3.10.5; it is not a dependency of the GRF simulation.
