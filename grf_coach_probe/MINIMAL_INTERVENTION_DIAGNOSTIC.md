# GRF minimal-intervention diagnostic

Date: 2026-10-12. **Decision: STOP; these movement overrides do not support a high-level RL coach.** This investigation added no RL training, coaching actions, player skills or engine modifications. The earlier tactical data and code remain intact. The new experiments use the same GRF 2.10.3, V2 actions and `11_vs_11_hard_stochastic` setup as the [original tactical report](TACTICAL_FEASIBILITY_REPORT.md).

## Why the continuous interventions lost

The installed engine's `game_env.cpp` calls `SetDisabled(false)` for an external action and `SetDisabled(true)` for `builtin_ai`. `playerbase.cpp::Process` runs the external `HumanController` when it is active, otherwise the native player controller. The two controllers do **not** simultaneously fight over a player. While overridden, a player loses its native controller's autonomous positioning and decision process. This is a verified controller switch, not an inferred effect. `HumanController` still calls automatic trap and interfere commands, so the evidence **does not** establish that all tackling or interceptions disappear. The Python translator sends only sprint/direction; it never requests pressure or sliding, and the rollouts do not expose a reliable tackle-event log. A loss of native marking/interception *decisions* is plausible, but assigning an individual missed tackle as the cause of a goal is not verified.

[Mechanism measurements](override_mechanism_cases.json) compare four representative original paired rollouts at equal engine steps. In Midfield 02, normal AI recovered at step 33 while continuous press did not recover within 200. By step 10, the two selected players' mean position differed from their normal trajectories by 0.052 pitch units while the ball position was still identical; by step 20, the difference was 0.117 while mean divergence among the eight other outfield players was only 0.007. In Penalty 02, cover delayed recovery from 72 to beyond 200; selected-player divergence was 0.075 at step 20, compared with 0.019 for the other outfield players. In a helpful Penalty 01 cover case, recovery improved from 101 to 34 and the selected players likewise diverged by 0.085 at step 20. Thus trajectory change alone is neither evidence of a useful nor harmful tactic.

The selected pair did **not** systematically lose simple goal-side status in those examples; this measure is too coarse to establish abandoned marking assignments. The two nearest *non-overridden* players in Midfield 02 had zero mean positional divergence at step 10 and 0.019 at step 20; in Penalty 02 the corresponding values were 0.003 and 0.010. Nearby AI teammates initially stayed close to their normal paths, then diverged as the ball/play diverged. This shows no immediate, measured team-wide compensation for the override, but does not prove that the AI never reacts. The original continuous press averaged 3.17 engagements per 200 steps, so repeated switches were possible. The one-burst test below still loses overall, ruling out repeated switching as the sole cause. Source files were read only; no GRF internals were changed.

## Locked minimal-intervention test

[`diagnose_minimal_interventions_py39.py`](diagnose_minimal_interventions_py39.py) applies one intervention at the first opponent-possession frame, then returns affected players to `builtin_ai` and never retriggers in that 200-step rollout. It tests one versus two selected players and predeclared 12- versus 36-frame limits. `cont` lasts until the first possession/neutral-ball end. The existing continuous press/cover implementations are retained as references and may reengage. All other players receive `builtin_ai`; the goalkeeper is never overridden. The tactical target and player-selection rules were reused unchanged.

Paired results on the 12 original saved defensive states ([raw rollouts](minimal_intervention_12states.json), [validated summary](minimal_intervention_summary.json)):

| Policy | Recoveries / 12 | Mean recovery steps* | Mean opponent progress | Goals conceded |
| --- | ---: | ---: | ---: | ---: |
| Normal AI | 10 | **73.0** | **0.346** | 1 |
| Two-player continuous press | 8 | 130.8 | 0.443 | 1 |
| One-player press, one continuous burst | **11** | 95.2 | 0.423 | 0 |
| One-player press, 12 frames | **11** | 101.8 | 0.468 | 0 |
| One-player press, 36 frames | 10 | 99.4 | 0.431 | 1 |
| Two-player press, 12 frames | 10 | 100.0 | 0.458 | 0 |
| Two-player continuous cover | 10 | 98.0 | 0.467 | 0 |
| One-player cover, one continuous burst | 10 | 109.8 | 0.516 | 1 |
| One-player cover, 12 frames | 9 | 109.9 | 0.441 | 1 |
| One-player cover, 36 frames | 10 | 103.3 | 0.522 | 1 |
| Two-player cover, 12 frames | 10 | 95.8 | 0.443 | 1 |

*No recovery within the 200-step horizon counts as 201. Lower recovery time and opponent progress are better. The apparently higher recovery count for one-player press does not overcome its slower recovery and greater opponent progress. One player is less harmful than two for pressing, but not for covering. Shorter duration helps some states and harms others; no monotonic duration benefit exists. The short two-player press won only four of twelve paired recovery-time comparisons against normal and lost seven, with one tie.

The original final result file recorded different normal-AI recovery times for two of these 12 states (Midfield 00 and 03), although the saved ball positions were the same. Current normal rollouts repeated exactly within a single process for both states after an intervening press rollout. GRF's scenario is stochastic and the older snapshot collection did not record an explicit numeric engine seed, so cross-process results from these snapshots should not be treated as bit-for-bit identical. All comparisons in each new experiment used paired restores within the same process; the held-out experiment also repeated normal AI after every intervention set as an audit. This variation further limits claims from a few favorable states.

## Preselected candidate and held-out check

Before seeing held-out outcomes, the only apparent situation-specific candidate was **two-player press for 12 frames in wide situations**: four of four original wide states recovered versus normal AI's three of four; mean censored recovery time was 59.2 versus 85.5 steps. Four new wide states, plus four new midfield and four new penalty-area states, were collected from four *different* live matches in [`tactical_states_holdout_minimal/`](tactical_states_holdout_minimal/). All 12 binary state hashes differ from the original 12. The rule, duration and player selection were frozen before evaluation.

| Held-out situation | Normal AI recoveries / 4 | Short press recoveries / 4 | Normal mean steps* | Short press mean steps* | Normal progress | Short press progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Midfield | 4 | 4 | 64.8 | 68.2 | 0.424 | 0.318 |
| Penalty area | 3 | 3 | 93.8 | 103.2 | 0.244 | 0.285 |
| **Wide** | **4** | 3 | **44.5** | 100.2 | **0.214** | 0.461 |
| All 12 | **11** | 10 | **67.7** | 90.6 | **0.294** | 0.355 |

The wide-channel signal **failed to replicate**. The short press was faster in only one held-out wide state, slower in two and tied in one; it conceded one goal while normal AI conceded none. This is a small sample, but it contradicts the claimed benefit and does not justify further tuning. The [held-out raw data](minimal_intervention_holdout.json) contains trajectories, action logs, ownership, scores, progress, danger proxies and defensive-shape measurements. Normal AI was replayed twice per held-out state, separated by the interventions: all 12 pairs had identical recorded trajectories. The [summary script](summarize_minimal_diagnostic.py) checks this, the 12/12 disjoint state hashes, one-/two-player limits, duration limits, nine or more AI-controlled players per frame and zero goalkeeper overrides.

Across all held-out states, the short press also had more dangerous-ball-region entries (0.83 versus 0.67 per rollout). Its mean count of goal-side teammates near the ball was *higher* (3.08 versus 2.64) despite worse recovery and progress. That confirms that simple shape or coordinate changes are insufficient to judge tactical value.

## Decision and limitations

**No reproducible low-interference coaching opportunity was found. Do not start PPO or DQN for these actions.** The interventions clearly change movement and sometimes change outcomes, but no fixed action has a repeatable advantage over normal AI in an identifiable situation class. A useful tactical coach would require additional player-control intelligence or an engine-level coaching interface, outside this limited diagnostic. The evidence supports stopping here rather than extending these movement overrides.

These are only 200-step defensive continuations, with 12 development and 12 held-out states from four matches each. Goals are sparse; ball entry into a dangerous region is a proxy, not a verified shot. Possession recovery and progress are more informative here than goals or match win rate. The current experiment cannot isolate a specific missed tackle from player-level animation/event logs, nor distinguish teammate adaptation from ordinary reactions to a changed ball trajectory.

## Reproduce

Run from the football workspace in PowerShell, using the existing isolated GRF Python installation. Reusing supplied state files gives the comparison; generating new states will create different stochastic matches.

```powershell
$probe = (Resolve-Path 'grf_coach_probe').Path
$linuxProbe = (wsl -d Ubuntu -- wslpath -a $probe).Trim()
$grfPython = '/home/rohith_syam/grf_coach_probe/py39/bin/python'

python grf_coach_probe/analyze_override_diagnostics.py
wsl -d Ubuntu -- $grfPython "$linuxProbe/diagnose_minimal_interventions_py39.py" --states "$linuxProbe/tactical_states_independent" --out "$linuxProbe/minimal_intervention_repeat.json"
wsl -d Ubuntu -- $grfPython "$linuxProbe/diagnose_minimal_interventions_py39.py" --states "$linuxProbe/tactical_states_holdout_minimal" --variants normal,press_2_12,press,cover,normal --out "$linuxProbe/minimal_holdout_repeat.json"
python grf_coach_probe/summarize_minimal_diagnostic.py
```

The saved results are not overwritten by these example commands. `summarize_minimal_diagnostic.py` validates the supplied final data files, so a rerun can be compared separately before replacing them.
