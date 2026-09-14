# GAFFER / Defensive Line Lab — reduced finished project

This is the submission-sized project. The larger PITCHSIM project and earlier
RoboCup probes are preserved separately; they are not required to view this demo.

**Question:** Can PPO choose defensive-line instructions more effectively than
fixed or random choices in a bounded RoboCup simulation task?

## Run the demonstration (Windows, no RoboCup needed)

From this project directory:

```powershell
python scripts/serve_line_coach.py
```

Open **http://127.0.0.1:8765**. Python's standard library is enough. No network,
GPU, Linux, machine-learning package, or native simulator is needed for playback.
Stop the server with Ctrl+C. Use `--port 8766` if the default port is occupied.
The server exposes only `line_coach_demo/`, not the rest of the project.

The interface shows recorded trajectories, not a live simulator. Use the policy
selector to compare replays, play/pause, change speed, scrub the episode, or
jump to the next decision. Scroll for held-out results and training curves.
Replays are the first predetermined evaluation episode, not selected victories.

## Deliverables

- `line_coach_demo/`: standalone replay app and exported result/replay data.
- `docs/line_coach_final_report.md`: methods, measured results, limitations.
- `data/line_coach/ppo_seed_*.zip`: trained PPO models (three actions, 149 observations).
- `data/line_coach/protocol.json`: frozen budget, seeds and source fingerprints.
- `data/line_coach/evaluation.json`: raw held-out episode outcomes.
- `data/line_coach/training_*.json`: learning traces and completion checks.
- `envs/line_coach_env.py`: finite-horizon three-action task.
- `scripts/finish_line_coach.py`: training, baseline evaluation and replay capture.
- `artifacts/GAFFER_Line_Coach_Submission.zip`: portable project package after publishing.

Do not describe this as a complete football manager, a Bayesian opponent model,
a regulation-match study, or evidence of transfer to real football. Consult the
report before claiming PPO improves on a baseline.

## Native training prerequisites

Training is **not** a clean-machine one-command installation. It needs the
working, separately built Linux RoboCup/R2DRL/HELIOS stack. On this machine:

- WSL distribution: Ubuntu.
- Bundle: `/home/rohith_syam/robocup-proto/R2DRL`.
- Python: `/home/rohith_syam/.cache/football-env-probe/venv/bin/python`.
- Historical native build notes/patches:
  `tmp/environment-comparison/robocup-results/native-suite/`.

The bundled upstream precompiled players did not work on this OS; use the
locally rebuilt stack. The native bundle path must not contain spaces. The
experiment uses unique Linux temporary directories for native log/config paths.
The portable submission package intentionally excludes native binaries and
their large build tree. Rebuilding on another machine is additional setup.

The actual training environment used Python 3.12.3, Stable-Baselines3 2.9.0,
Gymnasium 1.3.0, NumPy 2.5.3 and PyTorch 2.8.0+cpu. These versions are recorded in
`data/line_coach/software_versions.json`. Demo playback has no ML dependencies.

## Reproduce training/evaluation in the existing WSL setup

Open Ubuntu, change into this project directory, then run:

```bash
export ROBOCUP_BUNDLE=/home/rohith_syam/robocup-proto/R2DRL
/home/rohith_syam/.cache/football-env-probe/venv/bin/python \
  scripts/finish_line_coach.py --output data/line_coach_new \
  --timesteps 2304 --seeds 31 47 59 --workers 3 --horizon 1200 --eval-episodes 12
```

Use a new output directory for a fresh experiment. `--resume` continues saved
checkpoints/evaluations; use it only with exactly the same configuration and
source code. Native execution is stochastic even for an identical configured
seed. Do not expect byte-identical replays.

Then publish that run (overwrites the demo's exported data and generated report):

```bash
python scripts/publish_line_coach.py --source data/line_coach_new
```

The saved finite-horizon model expects the custom 149-value observation and
three-action mapping. It is incompatible with the older PITCHSIM and five-action
prototype checkpoints. Do not resume those old models.

## Tests

```powershell
python -m pytest tests/test_line_coach.py tests/test_robocup_coach.py -q
```

The native stack is not needed for these unit tests. Browser verification uses
`scripts/check_line_coach_demo.py` with Playwright installed; it is optional for
ordinary demonstration use.

After publication, `python scripts/verify_line_coach_artifacts.py` verifies that
the source fingerprints, raw results, models, exported replays and ZIP agree.

## Suggested presentation (about three minutes)

1. State the narrow question and distinguish the manager from the HELIOS players.
2. Play the first PPO replay; show its line instructions and decision history.
3. Switch to a fixed baseline and demonstrate scrubbing.
4. Show the results table and training curves. Explain what actually happened,
   including a negative result or a policy that mainly chose one instruction.
5. State the fixed-opponent, short-horizon, small-budget and simulator limitations.

The defensible achievement is a complete RL experiment with independent
evaluation and an inspectable demo, not guaranteed tactical superiority.

## Attribution

RoboCup server, HELIOS and R2DRL supply the native simulation and player skills.
Stable-Baselines3 supplies PPO; this project supplies the reduced manager task,
experiment orchestration and presentation. Upstream links are in the report.
No third-party native code is relicensed by this package.
