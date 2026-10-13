# GRF native tactical AI: source feasibility report

Date: 12 Oct 2026. Source read: `grf_coach_probe/football` (gfootball 2.10.3, commit `3d9e754`), engine under `third_party/gfootball_engine/`. No engine file, no earlier GRF/VMAS/RoboCup/PITCHSIM file was modified, no environment was built, no RL was run. The one runtime step was read-only: `dir()` on the installed `_gameplayfootball` module in WSL, which matches `ai.cpp` exactly. All paths below are relative to `third_party/gfootball_engine/` and `src/` is `third_party/gfootball_engine/src/`.

## 1. Verdict

**GRF does have a native, team-wide tactical layer that steers all ten outfield players while leaving their passing, dribbling, shooting and tackling code untouched. None of it is reachable from Python in the installed build.**

- **A (already Python-exposed):** nothing that changes tactics *during* an episode in a meaningful way. What exists is a formation chosen per episode (A2) and a one-player, half-second "team pressure" button (A1).
- **B (small interface change):** one patch of roughly 50 to 80 lines would expose three genuine team-wide controls. They are ranked in section 3.
- **C (redesign):** automatic team pressing, number of marked opponents, offside-trap geometry and pass-selection priorities are hard-coded constants. They are tunable only by editing constants, not by a coach interface.

**Recommendation:** do not build an RL environment on the existing three-action interface (already your decision). The native layer is real, but its **effect size is unmeasured and bounded by small multipliers**, so it is not yet a justified RL target. Given the submission deadline, my recommendation is **stop GRF coach development for this submission unless you are willing to spend one bounded go/no-go step**: apply the B1 patch (about half a day including the WSL rebuild), then run a no-RL paired screen of the extreme slider settings on the 12 saved states you already have. Proceed only if that screen shows a clear difference from default; otherwise stop. Section 5 gives the gate.

## 2. How the native tactical layer works

Each `Team` owns a `TeamAIController` (`src/onthepitch/team.cpp:39`). Its tactical state:

| Item | Where | What it is |
|---|---|---|
| `baseTeamTactics` | `teamAIcontroller.hpp:84`, set at `teamAIcontroller.cpp:56-69` | 14 positioning parameters, e.g. `position_defense_depth_factor` = 0.75 |
| `teamTacticsModMultipliers` | `teamAIcontroller.cpp:72-81` | how far a user setting may move each parameter |
| `TeamData.tactics.userProperties` | `src/data/teamdata.hpp:28`, loaded at `teamdata.cpp:264-271` from a hard-coded string (`teamdata.cpp:90`) | the "user sliders", 0..1, default values identical for both teams |
| `UpdateTactics()` | `teamAIcontroller.cpp:1221-1280`, called every 1000 ms (`:104`) | `live = clamp(base + (user-0.5)*2*multiplier)`; also recomputes `offensivenessBias` (`:1250`) from score, time and 60 s possession |
| `GetAdaptedFormationPosition()` | `teamAIcontroller.cpp:321-475` | turns the live tactics into each player's target position (depth, width, midfield stretch, side/ball focus) |
| Consumers | `strategies/offtheball/default_def.cpp:30-56`, `default_mid.cpp:34-71`, `default_off.cpp:33-34` | every outfield player's off-ball movement; also `CalculateDynamicRoles` (`teamAIcontroller.cpp:478-597`) |

Only off-the-ball *positioning* reads these. On-ball logic (`elizacontroller.cpp` pass selection around `:820-900`, `_BallControlCommand`, `_SlidingCommand`, `_InterfereCommand` at `:408-412`) does not, except the two `dribble_*` properties (`AIfunctions.cpp:249,274`). So a change here **preserves** built-in passing, shooting, dribbling and defending decisions.

## 3. Candidate controls (ranked)

Legend: **Py** callable from Python today, **Live** changeable mid-episode, **Native** preserves built-in on-ball skills, **Scope** team or one player, **Cost** to expose.

### Rank 1 (B): team tactic sliders (`userProperties`)

- **Source:** `UpdateTactics` `teamAIcontroller.cpp:1221-1280`; storage `teamdata.hpp:44` (`const` getter only); defaults `teamdata.cpp:90-111`.
- **Py:** No. `GameEnv` exposes only `config, game_config, get_frame, get_info, get_state, perform_action, render, reset, set_state, start_game, state, step, sticky_action_state, tracker_setup, waiting_for_game_count` (`ai.cpp:135-151`; confirmed at runtime in WSL).
- **Live:** Yes once exposed. Re-read every 1000 ms of match time (`:104`), so a coach decision takes effect within one simulated second.
- **Native:** Yes (positioning only, see section 2).
- **Scope:** Team-wide, all outfield players, for both defence and attack parameters.
- **Cost:** Small. Add a non-const accessor or setter in `teamdata.hpp`, one `GameEnv::set_team_tactic(team, name, value)` in `game_env.cpp` (reachable via `GetGameTask()->GetMatch()->GetTeam(i)->GetTeamData()`), one `.def` in `ai.cpp`, plus a Python wrapper. About 25 to 40 lines, then the existing `build_wsl_py39.sh` rebuild.
- **Effect size (important caveat):** The multipliers cap the swing. Maximum movement of a live value from its base: depth ±0.10, width ±0.10, midfield focus ±0.30, side focus ±0.10, micro focus ±0.15 (`:72-81`). Two inert entries: `position_*_ownhalf_factor` and `position_*_midfieldfocus_strength` have **no multiplier**, so setting them through `userProperties` does nothing (`GetReal(..., 0.0f)` at `:1260`). Worked example: team length scales with `adaptedDepth = 0.45 * (offense_depth*pb + defense_depth*(1-pb))` (`:376`) and the bounds use `pitchHalfW = 55` (`gamedefines.hpp:244`). Moving `defense_depth_factor` across its full ±0.10 range changes `adaptedDepth` by up to about 0.09 when fully defending, roughly **10 m of team length**. Noticeable, but small next to the player-level effects tested earlier. A larger swing needs a bigger multiplier or direct writes to `baseTeamTactics` (which, unlike `userProperties`, is saved by `ProcessState`, `teamAIcontroller.cpp:1311`).
- **Likely to change outcomes:** Plausible for shape and line height, **not demonstrated**. Both teams currently use the same sliders.

### Rank 2 (B): offensiveness bias override (defensive-line hold, fall-back and side focus)

- **Source:** computed `teamAIcontroller.cpp:1250` from `goalFactor`, `timeFactor`, 60 s possession. Used for the defensive hold line (`startDistance = 30 + 20*bias`, `:108`; `offsetX = 20 + 10*(1-bias)`, `:118`), which sets `offsideTrapX` (`:151`), applied in `ApplyOffsideTrap` (`:711-735`) from `default_def/mid`. Also shifts side focus (`:355-356`) and `possessionBias` (`:370`).
- **Py:** No (member `offensivenessBias`, `teamAIcontroller.hpp:88`, is private/protected, state-saved at `:1313`).
- **Live:** Yes once exposed, but it is recomputed every 1000 ms (`:1250`), so an override needs a coach term added to the formula rather than a write.
- **Native:** Yes.  **Scope:** Team-wide.
- **Cost:** Small. A `coachBias` float added at `:1250` and one setter, about 10 to 15 lines on top of Rank 1.
- **Caveat:** The engine already adapts this automatically (pushes up when trailing, drops when leading), so a coach competes with a built-in heuristic. Constants such as `forceDistance = 6` (`:109`), `cautionDistance = 4` (`:133`) and `allowSlackDistance = 4` (`:144`) stay fixed.
- **Likely to change outcomes:** Plausible for line height; unmeasured.

### Rank 3 (B): role and mind-set changes (decision priorities)

- **Source:** roles come from `FormationEntry` (`gamedefines.hpp:256`). `AI_GetMindSet(role)` (`AIfunctions.cpp:1196-1218`) is a fixed table (GK/CB 0.0 up to CF 1.0). It drives pass selection (`elizacontroller.cpp:835-847`: `forwardWeight = 2 + 6*mindSet`, `passMinimum`, `tacticalDiffWeight`), hunting distance (`:424-429`), dribble aggression (`AIfunctions.cpp:249`), and per-role position tweaks (`mixup`, `teamAIcontroller.cpp:270-300`). The static role selects which off-ball strategy runs (def/mid/off, `elizacontroller.cpp:313-333`).
- **Py:** Only **between episodes**: the scenario sets `left_team` (`ai.cpp:172-173`, read once in the `MatchData` constructor `matchdata.cpp:24` at `reset`, `game_env.cpp:384-395`; scenario files call `AddPlayer(x, y, role)`, `gfootball/env/scenario_builder.py:99`). `FormationEntry.role` is read-only in Python (`ai.cpp:212`).
- **Live:** Not today. Internally `TeamData::SetFormationEntry` (`teamdata.hpp:47`) and `Team::SetFormationEntry` (`team.cpp:132`) exist (already used for goalkeeper replacement, `player.cpp:436-442`), but formation data is **not** in the saved state (`MatchData::ProcessState` stores goals and possession only, `matchdata.cpp:38+`), so any `set_state` restore will not reset it.
- **Native:** On-ball skills yes, but changing a role also changes which positional strategy runs and the player's base position, so this is a bundle, not a single dial.
- **Scope:** Per player (a coach would set several).  **Cost:** Medium, about 20 to 30 lines plus a decision about roles versus base positions.
- **Likely to change outcomes:** Probably the strongest lever on decision priorities, but also the least controlled and hardest to interpret.

### Rank 4 (A): `team_pressure` / `keeper_rush` actions

- **Source:** actions exist in the Python `full` action set (`gfootball/env/football_action_set.py:140-155`) and engine (`game_env.cpp:223-231`, `gfootball_actions.h`). Executed in `HumanController` (`humancontroller.cpp:254-262`) only when the issuing player is human-controlled, **not** the ball holder, and the opponent has possession (`:427-491`, edge-triggered). `ApplyTeamPressure` (`teamAIcontroller.cpp:1185-1201`): sets the closest player to the carrier on man-marking for 500 ms (`:1187`) and `elizacontroller.cpp:381-391` turns on ball-magnet chasing for that one player.
- **Py:** Yes with `action_set='full'`.  **Live:** Yes.
- **Native:** Mostly, but issuing it calls `SetDisabled(false)` on the issuing controller (`game_env.cpp:176-178`), which removes that player's native AI until `builtin_ai` is sent (`:276`). The same controller-switch cost the earlier override experiments found.
- **Scope:** One player for 0.5 s. The native automatic team-pressure trigger is commented out as "DISABLED, interferes with other defense AI code" (`teamAIcontroller.cpp:~186-215`).
- **Likely to change outcomes:** Weak. Same family as the already-failed "selective press" override. Not run-verified (two-tick edge sequence inferred from source).

### A2: formation per episode (no engine change)

- **Source:** `ScenarioConfig.left_team/right_team` (`ai.cpp:172-176`) set by a scenario file; formation positions feed every player's base position (`AIfunctions.cpp:49-90`, `teamdata.cpp:205-250` also nudges overlapping entries).
- **Py:** Yes; **Live:** No, new match only. This supports a **pre-match formation choice** (a bandit over a few formations) but not in-match coaching, which is the goal in the brief.

### Not tactical / not available

- **Team difficulty** (`left_team_difficulty`, `right_team_difficulty`, `ai.cpp:184-187`): Python-exposed but constant per match (`const float aiDifficulty`, `team.hpp:139`; set at `match.cpp:117-124`). It scales every player's stats, reaction time and hunting range (`player.cpp:452-456`, `playercontroller.cpp:64`, `elizacontroller.cpp:429`). That is a skill nerf, not a tactic. Do not use.
- **Dynamic switching between predefined strategies:** none. The only strategy classes are per-role off-ball behaviours (`strategies/offtheball/`); there is no team-level strategy enum or selector.
- **Marking, pressing aggressiveness, offside-trap shape, pass priorities:** constants (`numMarkedOpponents = 3`, `teamAIcontroller.cpp:652`; trap area `:720`; `neededRating` for attacking runs `:224`; pass weights `elizacontroller.cpp:835-850`). Changing them is category C unless one is promoted to a parameter.
- **Width/compactness scale:** `depth = 0.45`, `width = 0.95` (`teamAIcontroller.cpp:42-43`, state-saved at `:1323-1324`) are plain members. They are the cleanest global compactness knobs and could ride on the Rank 1 patch (about 5 more lines), but they are not user-exposed today.

## 4. Summary table

| # | Control | Class | Py today | Mid-episode | Keeps native on-ball AI | Scope | Cost to expose | Expected effect |
|---|---|---|---|---|---|---|---|---|
| 1 | Tactic sliders (`userProperties`) | B | No | Yes (1 s lag) | Yes | Team | ~25-40 lines | Moderate, capped by multipliers |
| 2 | Offensiveness bias / hold line | B | No | Yes | Yes | Team | ~10-15 lines on top of 1 | Moderate, competes with native adaptation |
| 3 | Roles / mind-set | B | Episode start only | No (not in state) | Yes, but bundles position and strategy | Players | ~20-30 lines | Potentially large, hard to control |
| 4 | `team_pressure`, `keeper_rush` | A | Yes (`full` set) | Yes | Partly (issuer loses native AI) | 1 player, 0.5 s | none | Weak |
| 5 | Formation per episode | A | Yes | No | Yes | Team | none (custom scenario) | Moderate, pre-match only |
| 6 | Team difficulty | A | Yes | No | n/a | Team | none | Skill nerf, not tactics |
| 7 | Strategy switching, marking, pressing intensity | C | No | n/a | n/a | n/a | redesign | n/a |

## 5. Suggested go/no-go gate (if you choose to try once)

1. Apply only the Rank 1 setter (plus optionally Rank 2) in a **copy** of the engine, rebuild with the existing `build_wsl_py39.sh`, keep the original install intact.
2. Reuse the 12 saved opponent-possession states and the same 200-step horizon and paired seeds. Compare default sliders against two extremes (e.g. defence depth/width low versus high, offence width low versus high), `builtin_ai` for all players throughout.
3. Go only if a pre-declared metric (recovery time, shots conceded, or goal difference over the paired rollouts) separates from default beyond its bootstrap interval in at least one non-trivial direction, **and** the best setting differs by situation (otherwise there is nothing for an adaptive coach to learn). Otherwise stop.

## 6. Limits of this review

Source-only except for the binding `dir()` check. Nothing here has been run, so every "likely effect" is an inference from code. I did not verify the two-tick behaviour of `team_pressure` at runtime. The effect-size estimate for depth (about 10 m) is a back-of-envelope from `teamAIcontroller.cpp:376-400` and `pitchHalfW = 55`, not a measurement. Whether any slider changes results against `11_vs_11_hard_stochastic` (right team difficulty 0.95) is exactly what the Section 5 screen would test.
