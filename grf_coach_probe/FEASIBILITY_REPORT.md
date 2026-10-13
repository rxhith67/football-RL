# Google Research Football coaching feasibility — installation gate

**Historical snapshot (2026-10-11):** The installation blocker below was resolved on 2026-10-12 with isolated WSL Python 3.9. See [RUNTIME_REPORT.md](RUNTIME_REPORT.md) for executed match, mixed-control, and recording results. The following text preserves the earlier probe as it was observed at that time.

Date: 2026-10-11. Verdict: **FAIL for RL-coach readiness in this setup and investigation budget.** This is an installation-gate failure, **not proof that GRF cannot support mixed control in principle**. No match, mixed-control rollout, tactical comparison, or RL training was run. Stop here rather than replace GRF's football skills or claim that source-level API support is gameplay evidence.

## Installation and execution

| Check | Actual result |
| --- | --- |
| Host | Windows, Python 3.11.9; Docker CLI absent; WSL2 Ubuntu available with Python 3.12.3. |
| GRF source | Official [google-research/football](https://github.com/google-research/football) cloned at `3d9e754720a95621bba6475c4d3b0d56fe919014`; `setup.py` reports 2.10.3. |
| PyPI | Latest listed package was `gfootball==2.10.2`. No prebuilt binary wheel was available for this Windows Python (`pip download --only-binary=:all:` returned no matching distribution). |
| Isolated Windows install | `pip install --no-deps gfootball==2.10.2` reached the source wheel build, then failed: `VCPKG_ROOT environment variable is not defined`. The [official Windows build guide](https://github.com/google-research/football/blob/master/gfootball/doc/compile_engine.md) also requires Visual Studio C++ and vcpkg; neither was configured here. |
| Isolated WSL install | After Ubuntu's stale package indexes were refreshed, the minimal native build dependencies installed. `pip install .` from the official checkout stopped **before compiling GRF** while preparing its required `gym<=0.21.0`: `error in gym setup command: 'extras_require' must be a dictionary...`. WSL has Python 3.12.3. The full log is `install_wsl.log`. |
| Installed engine | **No.** Both isolated virtual environments report `Package(s) not found: gfootball` in `status.json`. Consequently there is no measured match throughput. |

The Python 3.12 / old Gym dependency is a concrete failure of the supported source installation attempted here. A possible recovery would require a different Python/toolchain or deliberately bypassing and testing an unsupported dependency combination. Given the deadline and the instruction to stop on a hard installation blocker, that is outside this bounded probe.

## Verified API facts from source, not gameplay

- The [official action definitions](https://github.com/google-research/football/blob/master/gfootball/env/football_action_set.py) have 19 V1 actions; V2 appends `action_builtin_ai` as **action ID 19** and has 20 actions total. `status.json` records this direct source inspection. This is **not an installed-version API test**.
- `create_environment(..., number_of_left_players_agent_controls=11, other_config_options={'action_set': 'v2'})` is represented in the [Python environment source](https://github.com/google-research/football/blob/master/gfootball/env/__init__.py). The environment exposes a multi-discrete action vector when more than one player is externally controlled. This call was **not executed**.
- In the [engine's `GameEnv::action`](https://github.com/google-research/football/blob/master/third_party/gfootball_engine/src/game_env.cpp), `game_builtin_ai` disables that player's external controller; other actions first enable it. The [provided wrapper](https://github.com/google-research/football/blob/master/gfootball/env/wrappers.py) fills nonselected players' actions with `action_builtin_ai`. This is strong source-level evidence for per-player switching, but it does not establish stable switching, football quality or tactical value during a live match.
- The engine exposes `get_state` and `set_state` in source, but snapshot reproducibility was not tested.

## Four separate feasibility checks

| Gate | Result | Evidence |
| --- | --- | --- |
| Simulator functionality | **Not verified** | Engine package did not install. Zero matches, goals, passes, shots or defensive sequences observed. |
| Player-control functionality | **Source-level plausible; unverified live** | V2 ID 19 and engine controller switching exist in source. Zero external movement/return-to-AI rollouts. |
| Tactical behaviour | **Not verified** | Zero paired trials; no trajectories, pressure, possession or score comparisons. No replay. |
| RL learning-readiness | **FAIL** | No executable simulator or measured state-dependent tactical advantage; training speed unknown. No RL training attempted. |

There are **no tactical sensitivity figures or gameplay numerical results to report**. Supplying zeros as match results would be misleading: the corresponding experiments never ran. The only measured numbers here are environment versions, source action counts/ID, install exit codes and the source commit in `status.json`.

## Reproduction

From the football workspace in PowerShell:

```powershell
# Official source (already cloned for this probe)
git clone --depth 1 https://github.com/google-research/football.git grf_coach_probe/football

# Windows package check and isolated install; the latter fails at VCPKG_ROOT
python -m pip download --only-binary=:all: --no-deps --dest grf_coach_probe/packages gfootball==2.10.2
python -m venv grf_coach_probe/.venv
grf_coach_probe/.venv/Scripts/python.exe -m pip install --no-deps gfootball==2.10.2

# WSL packages used for the source-install attempt. Ubuntu indexes first needed refreshing.
wsl -d Ubuntu -u root -- apt-get update -qq
wsl -d Ubuntu -u root -- apt-get install -y -qq --no-install-recommends libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-python-dev libboost-thread-dev libboost-system-dev libboost-filesystem-dev python3-dev python3-venv

# Run the isolated install and source/status probe
$probePath = (Resolve-Path 'grf_coach_probe').Path
$linuxPath = (wsl -d Ubuntu -- wslpath -a $probePath).Trim()
wsl -d Ubuntu -- python3 -m venv "$linuxPath/.venv-linux"
wsl -d Ubuntu -- bash "$linuxPath/install_wsl.sh" *> 'grf_coach_probe/install_wsl.log'
python grf_coach_probe/probe_status.py
```

The last command needs WSL access to populate the WSL fields in `status.json`; an ordinary restricted sandbox may record them as null. The checked-in `status.json` was generated with WSL access.

## Recommendation

**Stop the GRF pivot for this submission.** A rerun on a known-compatible Linux/Python image could be a later project, but a working `action_builtin_ai` interface and a positive tactical trade-off still require live experiments. For the present deadline, finalise the existing, untouched [`vmas_setpiece` result](../vmas_setpiece/REPORT.md): 65.72% held-out goals for its learned shooting-target coach versus 60.64% for the strongest rule, with the limitation that it is a one-decision attacking restart rather than a full-match tactical coach.
