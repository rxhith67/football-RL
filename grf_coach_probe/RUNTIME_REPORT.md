# Google Research Football runtime probe

Date: 2026-10-12. **GRF is installed and running in an isolated WSL Python 3.9 environment.** This report supersedes the installation-only verdict in `FEASIBILITY_REPORT.md`; that older report remains as a record of the Python 3.12 and native-Windows failures. No tactical coach or RL training was added.

## Environment and build

- Host: Windows with WSL2 Ubuntu 24.04.2 LTS. Docker CLI was unavailable, so the official Docker route was not attempted. WSL was the next preferred route.
- GRF source: official `google-research/football` commit `3d9e754720a95621bba6475c4d3b0d56fe919014`; installed package `gfootball 2.10.3` in `/home/rohith_syam/grf_coach_probe/py39`.
- Python 3.9.23; pip 23.0.1; setuptools 65.5.0; wheel 0.38.4; Gym 0.21.0; NumPy 1.23.5; Boost/Boost.Python 1.82.0 for Python 3.9. Exact installed Python packages are in `pip_freeze_py39.txt`; the conda inventory is in `micromamba_list_py39.txt`.
- CMake 3.28.3 and g++ 13.3.0. `cmake_py39.log` confirms Python 3.9 and Boost.Python `python39`; `make_py39.log` ends with `Built target game`; `install_py39.log` records a successful source wheel build and install. `pip check` returned `No broken requirements found`.
- The source package omits `six` from its declared requirements despite importing it. An initial post-build import failed on that module; installing pinned `six==1.16.0` resolved it. This is reflected in the setup script. The old Gym metadata error was resolved by Python 3.9 with pinned packaging tools and `--no-build-isolation`. No GRF engine source patch was required. The build script limits the initial C++ compile to two jobs and points CMake to the isolated Python and Boost libraries.

## Executed gameplay checks

| Check | Actual result | Evidence |
| --- | --- | --- |
| Native import/reset/step | `gfootball.env` imported, V2 exposed 20 actions and `action_builtin_ai` was ID 19. `academy_empty_goal_close` scored 1–0 in 9 native engine steps and returned reward 1.0. | `smoke_py39.py` |
| Complete 11v11 match | `11_vs_11_stochastic` reached `done=True` after 3,001 steps. The final run ended 2–0 in 22.40 s (133.96 steps/s). All 11 left actions were `builtin_ai`; all 11 opposing players moved more than 0.01 pitch units. | `full_match_final.json` |
| Ball play/defending | The final standard match recorded 1,055 left-possession frames, 489 right-possession frames, 35 changes of possession side, and 47 short-gap teammate-to-teammate transfer candidates. These measure live interaction and defending pressure; the transfer count alone does not prove every transfer was a deliberate pass. | `full_match_final.json` |
| Explicit pass | In an existing `academy_3_vs_1_with_keeper` scenario, one controlled player received a five-frame short-pass button press followed by release. In 4 of 5 episodes, the ball became unowned and then teammate 2 received it **before** any shot command. One attempt did not complete the transfer. | `pass_action_hold5.json`, `pass_action_check_py39.py` |
| Shots/scoring | Built-in AI scored in the 9-step empty-goal check, the 27-step passing-and-shooting academy recording, and the complete 11v11 matches. The 3v1 explicit-action check included one 1–0 episode after a completed pass; it was not designed as a shooting-success benchmark. | `smoke_py39.py`, `recordings/episode_done_20261012-000257090076.dump`, `full_match_final.json` |
| Mixed control | In a live 11v11 match, player 4 moved +0.201 pitch units in x during a 25-step external `right` command. Nine other left players moved in that interval. When player 4 was returned to `builtin_ai`, its x drift was −0.0015 over the next 25 steps and its y position changed, unlike continued rightward control. The ball continued moving in all phases. | `mixed_control.json` |
| Recording | GRF's own headless replay writer saved a 100-frame, 10 fps, 800×450 11v11 AVI and matching `.dump`. OpenCV decoded a middle frame successfully; `recordings/shutdown_20261012-000328475322.png` is the screenshot. A second 27-frame academy AVI/dump also decoded. | `recordings/`, `verify_recordings_py39.py` |

The 2–0 match is one stochastic rollout, **not** an evaluation of tactical quality or a win-rate estimate. A previous complete easy-opponent match ended 5–0 at 136.53 steps/s (`full_match.json`); a separate standard match ended 1–0 at 138.38 steps/s (`full_match_standard.json`). Scores vary between stochastic runs. The explicit pass test establishes that the action can produce a teammate reception, not a pass success rate under general match conditions.

## Reproduce

From the football workspace in PowerShell, install only WSL system build libraries (not Python packages), then run the isolated bootstrap:

```powershell
wsl -d Ubuntu -u root -- apt-get update -qq
wsl -d Ubuntu -u root -- apt-get install -y --no-install-recommends cmake build-essential git curl bzip2 libgl1-mesa-dev libegl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev
$probe = (Resolve-Path 'grf_coach_probe').Path
$linuxProbe = (wsl -d Ubuntu -- wslpath -a $probe).Trim()
wsl -d Ubuntu -- bash "$linuxProbe/bootstrap_wsl_py39.sh"
wsl -d Ubuntu -- /home/rohith_syam/grf_coach_probe/py39/bin/python "$linuxProbe/smoke_py39.py"
wsl -d Ubuntu -- /home/rohith_syam/grf_coach_probe/py39/bin/python "$linuxProbe/runtime_probe_py39.py" full --scenario 11_vs_11_stochastic --out "$linuxProbe/full_match_repeat.json"
wsl -d Ubuntu -- /home/rohith_syam/grf_coach_probe/py39/bin/python "$linuxProbe/runtime_probe_py39.py" mixed --out "$linuxProbe/mixed_control_repeat.json"
```

The bootstrap copies only the isolated official source checkout into WSL's ext4 filesystem for compilation, creates the pinned Python 3.9 environment, installs matching Boost.Python, and builds GRF. Scripts and results in this workspace remain under `grf_coach_probe/`; existing VMAS, RoboCup, and PITCHSIM implementations were not changed.

The bootstrap script passed shell syntax validation; its component commands were executed successfully in this probe. It was not rerun from a blank WSL installation after assembly, so a clean-machine bootstrap remains to be checked.

## Limits for a future coaching project

GRF's built-in AI and per-player control switching now work at runtime, removing the installation blocker. This does **not** establish a high-level coaching interface or tactically meaningful formation control. The mixed-control test applied a direct movement command to one player. The next research gate, if pursued, is to test whether sparse high-level instructions can consistently alter several built-in AI players' behaviour and improve outcomes without replacing their low-level skills. No PPO/DQN training should be inferred from this probe.
