"""Publish real experiment artifacts into the portable demo and report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def report_text(summary):
    results = summary["results"]
    ppo = [r for r in results if r["policy"] == "ppo"]
    baseline = max((r for r in results if r["policy"] != "ppo"), key=lambda r: r["mean_goal_difference"])
    mean = sum(r["mean_goal_difference"] for r in ppo) / len(ppo)
    rows = "\n".join(f"| {r['name']} | {r['episodes']} | {r['wins']}/{r['draws']}/{r['losses']} | {r['mean_goal_difference']:.3f} | [{r['goal_difference_ci95'][0]:.3f}, {r['goal_difference_ci95'][1]:.3f}] | {r['points_per_episode']:.3f} |"
                      for r in results)
    training = "\n".join(f"| {t['seed']} | {t['timesteps']} | {len(t['episodes'])} | {t['seconds'] / 60:.1f} |" for t in summary["training"])
    distributions = "\n".join(f"| {r['name']} | {' | '.join(str(n) for n in r['actions'])} |" for r in results)
    fixed_like = [r for r in ppo if sum(n > 0 for n in r["actions"]) == 1]
    behaviour = ("Observed policy collapse: " + "; ".join(
        f"{r['name']} chose {('Normal play', 'Low line', 'High line')[next(i for i, n in enumerate(r['actions']) if n > 0)]} for every evaluated decision"
        for r in fixed_like) + ". These policies did not demonstrate adaptive switching."
        if fixed_like else "The evaluated PPO policies used multiple instructions, which alone does not prove effective tactical adaptation.")
    conclusion = ("PPO's mean was higher in this sample; the small evaluation does not establish reliable superiority."
                  if mean > baseline["mean_goal_difference"] else
                  "PPO did not outperform the strongest fixed/random baseline on the primary metric. This is a negative experimental result, not evidence of an effective adaptive coach.")
    return f"""# GAFFER: Defensive-Line Selection with Reinforcement Learning

## Abstract

This reduced project studies whether PPO can choose defensive-line instructions
more effectively than fixed or random instructions in RoboCup 2D episodes.
An existing HELIOS player team executes the individual football skills. The
manager selects Normal play, Low line or High line. Three independently
initialised PPO policies are evaluated alongside four baselines using held-out
episode seeds. Across trained seeds, mean goal difference was **{mean:.3f}**;
the strongest baseline, **{baseline['name']}**, obtained **{baseline['mean_goal_difference']:.3f}**.
{conclusion}

## Scope and research question

Does learned defensive-line selection outperform fixed instructions in this
specific simulation task? The deliverable is a tested environment wrapper,
bounded training/evaluation pipeline, trained checkpoints, recorded replay demo,
and results. The original broad football-manager proposal was narrowed: pressing,
formation selection, substitutions, opponent-belief inference, self-play and
human-versus-AI play are excluded. Claims are limited to the tested simulator.

## Environment and method

- Engine: locally built RoboCup server 19.0.0 and HELIOS via R2DRL commit
  `d4647b7e8cc31386a6e5cf94141cd2483113b56b`. This is an existing native engine,
  not newly written football physics. The native build requires Linux/WSL.
- Teams: 11v11; controlled left team against the fixed default HELIOS opponent;
  homogeneous player types and the bundle's noise settings. Native stamina,
  effort and recovery defaults are restored.
- Actions: 0 = native HELIOS play; 1 = low defensive-line target x = -30 m;
  2 = high target x = -5 m. Off-ball defenders receive masked native movement
  skills; a nearby defender can challenge the ball. When our team can kick the
  ball, native passing, shooting and support decisions are retained.
- Observation: 136 normalized global-state values, 11 own measured stamina
  values, horizon progress and clipped score difference, for 149 values. There
  is no hidden opponent-style label or Bayesian inference component.
- Decision interval: 20 native action steps. Horizon: {summary['horizon']} steps,
  or {summary['horizon'] // 20} decisions in the configured task. These are
  fixed-length experimental episodes, **not regulation football matches**.
  Completion of this finite task is termination; remaining time is observed.
- Reward: score-difference increment plus potential shaping
  `0.05 * (0.99 * Phi(next) - Phi(current))`, where Phi is clipped ball x/52.5
  and terminal Phi is zero. This does not directly reward choosing a particular
  instruction. Evaluation uses raw goals and results, not the shaping return.
- Algorithm: Stable-Baselines3 PPO, CPU, two 64-unit hidden layers,
  learning rate 0.0003, gamma 0.99, rollout 128 steps per worker, minibatch 128,
  four update epochs, entropy coefficient 0.01. Three subprocess environments
  collect experience. Normalized native inputs are used directly.
- Checkpoints: final checkpoint at the declared budget for every training seed.
  No test-set selection or hyperparameter sweep. The demo uses the first
  predeclared seed, {summary['selected_demo_seed']}, not the best evaluated model.

## Training budget and execution

| Training seed | Manager decisions | Completed training episodes | Minutes |
| --- | ---: | ---: | ---: |
{training}

All saved policies passed the finite-parameter and changed-parameter checks.
Training curves in the demo are five-episode rolling averages of shaped return.
They are diagnostic and are not held-out performance evidence.

## Evaluation protocol

The experiment evaluates each of the three policies and the four baselines on
{results[0]['episodes']} episodes, giving {summary['evaluation_episodes']} evaluation episodes.
Evaluation uses reserved native seed identifiers starting at 800000 and a
separate RNG for random instructions. Identical seeds do **not** imply identical
trajectories in this multiprocess build, so the study does not claim deterministic
paired rollouts. Episodes are independent restarts. All models and baselines use
the same horizon, observations, player team and execution rules.

Primary metric: mean raw goal difference. Secondary metrics: win/draw/loss counts
and points per episode (3/1/0). Intervals below use 3,000 nonparametric episode
bootstrap resamples with fixed analysis RNG seed 4001. They are exploratory,
not multiple-comparison-adjusted tests or training-seed-level confidence bounds.

## Results

| Controller | Episodes | W/D/L | Mean goal difference | Exploratory 95% interval | Points/episode |
| --- | ---: | --- | ---: | --- | ---: |
{rows}

{conclusion}

### Instruction usage in evaluation

Counts below are managerial decisions, not individual player actions. A policy
that repeatedly selects one instruction should not be described as learning
context-sensitive tactical adaptation solely because it is implemented with PPO.

| Controller | Normal | Low | High |
| --- | ---: | ---: | ---: |
{distributions}

{behaviour}

## Software verification and demonstration

The reduced environment tests verify the three-action boundary, observation
shape, goal reward, potential cancellation at termination, seed progression,
reset/close and 22-player replay schema. Earlier native integration checks
established live 11v11 operation and correctly signed line-position responses;
they did not establish football realism or tactical superiority.

The browser demo displays actual recorded native trajectories, current line
instructions, observed defender position, stamina, decision history, baseline
results and training curves. Play/pause, restart, speed, scrubbing and policy
selection work without a native simulator. Positions are sampled every two
native steps and interpolated for playback; this is not a live trained-agent run.
Each displayed replay is the first predetermined evaluation episode for that
controller, not a hand-picked win.

## Limitations and conclusion

{chr(10).join('- ' + item for item in summary['limitations'])}
- The global observation is privileged relative to real coaching perception.
  Player-skill delegation bypasses RoboCup coach-message restrictions; no
  competition-compliance claim is made.
- Fixed teammate/opponent heuristics, simplified line targets, kickoff-side
  asymmetry, noisy native execution and limited training constrain conclusions.
- A changed neural network and a working replay are not evidence that the agent
  learned useful football management. The held-out comparison is the relevant
  evidence, and greater training could help or fail to help.

The project delivers a complete, narrowly scoped RL experiment. {conclusion}

## Reproduction and attribution

See `LINE_COACH_README.md` for commands and native prerequisites. Full protocol,
raw episode results, training traces, checkpoints and source fingerprints are in
`data/line_coach/`. The portable demonstration is `line_coach_demo/`.

- RoboCup server: https://github.com/rcsoccersim/rcssserver
- R2DRL integration: https://github.com/open-starlab/R2DRL
- HELIOS base team: https://github.com/helios-base/helios-base
- PPO: Schulman et al., 2017, https://arxiv.org/abs/1707.06347
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

Native software is externally supplied and retains its upstream licensing.
The portable replay package does not redistribute native simulator binaries.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/line_coach")
    args = parser.parse_args()
    source = args.source.resolve()
    summary = json.loads((source / "summary.json").read_text())
    if not summary.get("complete"):
        raise ValueError("Refusing to publish incomplete results")
    dest = ROOT / "line_coach_demo/data"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "summary.json", dest / "summary.json")
    for row in summary["results"]:
        if row["replay"]:
            shutil.copy2(source / row["replay"], dest / row["replay"])
    (ROOT / "docs/line_coach_final_report.md").write_text(report_text(summary), encoding="utf-8")
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    files = list((ROOT / "line_coach_demo").rglob("*"))
    files += [ROOT / p for p in ("LINE_COACH_README.md", "docs/line_coach_final_report.md",
              "scripts/serve_line_coach.py", "scripts/finish_line_coach.py", "scripts/publish_line_coach.py",
              "scripts/probe_robocup.py", "envs/line_coach_env.py", "envs/robocup_coach_env.py",
              "tests/test_line_coach.py", "tests/test_line_coach_delivery.py", "tests/test_robocup_coach.py",
              "scripts/check_line_coach_demo.py", "scripts/verify_line_coach_artifacts.py",
              "requirements-line-coach.txt", "data/line_coach_tests.xml")]
    files += list((ROOT / "artifacts/browser-check").glob("*.png"))
    files += list((ROOT / "artifacts/browser-check").glob("result.json"))
    browser_result = ROOT / "artifacts/browser-check/result.json"
    if browser_result.exists():
        video = json.loads(browser_result.read_text()).get("video")
        if video:
            files.append(ROOT / video)
    files += list(source.glob("ppo_seed_*.zip")) + list(source.glob("training_*.json"))
    files += list(source.glob("replay_*.json"))
    files += [source / p for p in ("protocol.json", "summary.json", "evaluation.json", "software_versions.json")]
    with zipfile.ZipFile(artifacts / "GAFFER_Line_Coach_Submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, str(path.relative_to(ROOT)))
    print("Published replay demo, final report and artifacts/GAFFER_Line_Coach_Submission.zip")


if __name__ == "__main__":
    main()
