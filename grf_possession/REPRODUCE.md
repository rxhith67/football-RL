# Reproducing the GRF possession / Double Q-learning coach experiment

This experiment asks whether a small reinforcement-learning coach, which only chooses a team's defensive depth (compact / native default /
stretched) every 100 engine steps in Google Research Football (GRF), improves match results or possession compared with the same team
without a coach. Models: **A** Q-learning with the original reward, **B** Q-learning with the possession-plus-goals reward,
**C** Double Q-learning with the possession-plus-goals reward, against **native** GRF. The final result is in
[`results/final100_tables.md`](results/final100_tables.md).

The coach needs a **patched GRF engine** (two new functions that set and read a team's defensive depth). The patch is
[`../grf_coach_probe/native_depth_poc/engine_patch.diff`](../grf_coach_probe/native_depth_poc/engine_patch.diff) (5 files, 36 lines added,
1 removed against GRF commit `3d9e754720a95621bba6475c4d3b0d56fe919014`). Everything below rebuilds that engine and runs the code on it.

## 0. What you need

| | |
|---|---|
| OS | Linux x86-64 (we used Ubuntu 24.04). **Windows: WSL2 Ubuntu** (below). macOS or ARM: use an x86-64 Linux VM; not tested. |
| RAM | 8 GB minimum for WSL (we ran with 7.8 GB; 2-3 workers use about 2 GB). |
| Disk | about 5 GB free (toolchain 0.9 GB, two engine builds 1.3 GB, venv 0.9 GB, data). |
| CPU | any x86-64; a GPU is not used. Training and evaluation are CPU-bound (the engine is C++). |
| Time | install about 1 hour (mostly compiling). Verification 5-10 minutes. |
| Internet | needed once (GitHub, conda-forge, PyPI). |

### Windows: set up WSL2 first (PowerShell, as administrator, once)
```powershell
wsl --install -d Ubuntu-24.04      # reboot if asked, then create the Linux user
```
Open the Ubuntu terminal for everything below. Give WSL enough memory by creating `C:\Users\<you>\.wslconfig` containing
`[wsl2]` and `memory=8GB`, then run `wsl --shutdown` once.

## 1. Get the code
```bash
cd ~
git clone -b grf-possession-double-q https://github.com/rxhith67/football-RL.git
cd football-RL
```
(If you cloned onto the Windows drive instead, it also works from `/mnt/c/...`, but builds are much slower there. Cloning inside the Linux
home directory is recommended.)

## 2. Install everything (one command)
```bash
bash grf_possession/setup/setup_all.sh
```
This runs the five steps below; it is safe to re-run and skips finished parts. Already have the system packages? `SKIP_APT=1 bash ...`.
Compiler jobs default to 2 (gentle on laptops); on a desktop use `JOBS=8 bash grf_possession/setup/setup_all.sh`.

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `setup/01_system_packages.sh` | apt packages: build-essential, cmake, git, curl, SDL2, GL/EGL, Boost headers (needs `sudo`) | 3-5 min |
| 2 | `setup/02_toolchain_and_stock_grf.sh` | clones google-research/football at the pinned commit, builds a Python 3.9.23 + Boost 1.82 toolchain with micromamba, installs the pinned packages (gym 0.21.0, numpy 1.23.5, ...), compiles and installs **stock** GRF | 20-40 min |
| 3 | `setup/03_patch_and_build_engine.sh` | copies the stock source to a separate folder, applies `engine_patch.diff`, compiles it, assembles `overlay/` (stock `gfootball_engine` package with the patched `_gameplayfootball.so`) | 10-25 min |
| 4 | `setup/04_python_env.sh` | creates the virtualenv that runs our code (matplotlib etc.; torch only if `INSTALL_TORCH=1`, and only the old DQN script needs it) | 2 min |
| 5 | `setup/05_verify.sh` | checks the loaded engine is the patched one, runs the 29 tests, and **re-plays 12 of our matches and requires them to match ours exactly** | 5-10 min |

Install locations (override with environment variables, and export the same values when you later run the experiments):
`GRF_PROBE=$HOME/grf_coach_probe` (toolchain + stock GRF), `GRF_POC=$HOME/grf_native_poc` (patched copy + `overlay/`),
`GRF_VENV=$HOME/grf_rl_env`. `grf_possession/run.sh` puts the overlay first on `PYTHONPATH` so the patched engine is the one that loads;
the stock GRF installed in step 2 is never modified.

**You are set up correctly when step 5 ends with `12/12 matches identical to the reference - REPRODUCED` and `SETUP COMPLETE AND VERIFIED`.**
Matches are deterministic for a given seed, side and policy, so any mismatch means your engine build or GRF revision differs from ours.

## 3. Reproduce the results (three levels, cheapest first)

All commands run from `grf_possession/`.

**Level 1: the tables, in seconds, no simulator.** The 800 evaluated matches are stored in `reference/final100_per_match.csv`.
```bash
bash run.sh final_report.py reference/final100_per_match.csv mytables      # writes results/mytables_tables.md
```
It is identical to `results/final100_tables.md` (bootstrap seed fixed).

**Level 2: re-run the 800-match evaluation (about 3.3 hours with 3 workers; native, A, B, C x 100 seed pairs 7,000,000-7,000,099 x both sides).**
```bash
bash eval100.sh                                   # results/eval_final100/ ; each finished policy is cached, re-running resumes
bash run.sh final_report.py                       # tables from the fresh raw results
```
Worker count is read from `logs/eval100_workers.txt` before each policy (default 3). Use 2 on a laptop: `echo 2 > logs/eval100_workers.txt`.
Evaluation uses the frozen final checkpoints (greedy, no exploration):

| Model | File | SHA-256 |
|---|---|---|
| A | `../grf_rl_coach/checkpoints/q_it100_final.npz` | `c54f3c63eae23dc8ed1d2ad79ecd5ddb5e1486917ff4ca858a43729918a34f17` |
| B | `checkpoints/q_possession_run1_it100.npz` | `8618bfc92835193e308d32ce41b6ad0f1a9b748276a37c4c3f9d7dda803669a6` |
| C | `checkpoints/double_q_possession_run1_it100.npz` | `6b451d8a886e3067b733e7c558dbcbbc5d95d2b80c4f22d2f6b1a28ed07fc98e` |

**Level 3: retrain B and C from scratch (1,200 matches each, 100 iterations x 12 matches, 2 workers, 30 s rest per iteration).**
```bash
bash full_train.sh q possession              # B, about 4-5 hours; resumes automatically if interrupted
bash full_train.sh double_q possession       # C, about 5-6 hours
```
Same training seeds (2,010,000-2,011,199), epsilon schedule 0.5 to 0.05 over 70 iterations, alpha 0.1, gamma 0.99. Checkpoints are written
atomically every iteration to `checkpoints/` and logged to `logs/*.jsonl`; the learner RNG state is saved so a resumed run continues exactly.
Model A was trained in the earlier project (`../grf_rl_coach/train_q.py`, 1,170 matches); we ship its frozen checkpoint and do not retrain it.
Training from scratch should give the same tables if the engine is identical, but we have not re-run training on a second machine.

### Heat and power (important on laptops)
Plug in the charger and disable sleep while a run is going; a sleep only pauses the run (matches are deterministic) but wastes hours.
Our launchers run at lowest priority (`nice 19`) with rests between batches. If the machine gets hot, lower the worker count.

## 4. What is in the repository

| Folder | Contents |
|---|---|
| `grf_possession/` | this experiment: possession detector (`possession.py`), environment with side handling (`poss_env.py`), Q and Double Q updates (`learners.py`), training (`train_poss.py`), evaluation (`evaluate_poss.py`), analysis (`analyze_poss.py`, `final_report.py`), tests, launch scripts, `setup/`, `reference/`, small results and logs |
| `grf_rl_coach/` | the earlier coach project (environment, Model A's training and checkpoint); imported read-only by `grf_possession` |
| `grf_coach_probe/native_depth_poc/` | the engine patch (`engine_patch.diff`, `apply_patch.py`), its build script and the proof-of-concept report |
| `grf_coach_probe/*.md` | feasibility reports for GRF's native tactical AI |

Seeds used (never reused across purposes): training 2,010,000-2,011,199; earlier pilots 3,100,000 / 3,200,000; preliminary 20-pair evaluation
6,000,000-6,000,019; **final evaluation 7,000,000-7,000,099**; 5,000,000+ reserved and unused.

## 5. How the coach works (short)
Every 100 engine steps (30 decisions per 3,001-step match) the coach picks one of three team defensive-depth settings: compact (0.0), native
default (override cleared, about 0.71) or stretched (1.0). All 11 players per side stay under GRF's built-in AI. The state is 243 discrete
states (ball zone x possession x score x phase x last action). Reward for B and C: `(possession share - 0.5) + 2 x (goals for - goals against)`
per decision. Q-learning and Double Q-learning are tabular (a 243 x 3 table), not neural networks.

## 6. Troubleshooting
* **`PATCH NOT PRESENT` or "stock engine was loaded"** in step 5: the overlay is missing or not first on the path. Re-run step 3 and always run Python through `run.sh`.
* **`patch does not apply`**: GRF is not at commit `3d9e754...`; step 2 checks this, so do not edit `$GRF_PROBE/football`.
* **`bad_alloc` / killed processes**: not enough RAM for WSL; set `memory=8GB` in `.wslconfig` and use fewer workers.
* **cmake cannot find Boost.Python**: re-run step 2 (it installs Boost 1.82 into the toolchain; the apt Boost packages are only a fallback).
* **A mismatch in step 5(c)**: your engine differs from ours. Confirm `git -C $GRF_POC/football diff --stat` shows `5 files changed, 36 insertions(+), 1 deletion(-)` and that the GRF commit is the pinned one.
* **Different machine, different numbers at Level 2/3**: tell us; matches should be bit-identical on the same engine build, but we have only verified this on one machine.

## 7. What has and has not been verified
Verified on the original machine: the saved patch is identical to the live patched source and applies cleanly to a fresh checkout of the pinned
GRF commit; the 29 tests pass; the verification script re-plays 12 matches identically; the CSV rebuilds the report exactly.
**Not yet verified:** the setup scripts on a completely clean machine (they are assembled from the scripts that built our environment), and
training reproducibility on a second machine.
