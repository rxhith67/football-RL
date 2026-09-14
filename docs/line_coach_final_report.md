# GAFFER: Defensive-Line Selection with Reinforcement Learning

## Abstract

This reduced project studies whether PPO can choose defensive-line instructions
more effectively than fixed or random instructions in RoboCup 2D episodes.
An existing HELIOS player team executes the individual football skills. The
manager selects Normal play, Low line or High line. Three independently
initialised PPO policies are evaluated alongside four baselines using held-out
episode seeds. Across trained seeds, mean goal difference was **-2.194**;
the strongest baseline, **Normal HELIOS**, obtained **0.250**.
PPO did not outperform the strongest fixed/random baseline on the primary metric. This is a negative experimental result, not evidence of an effective adaptive coach.

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
- Decision interval: 20 native action steps. Horizon: 1200 steps,
  or 60 decisions in the configured task. These are
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
  predeclared seed, 31, not the best evaluated model.

## Training budget and execution

| Training seed | Manager decisions | Completed training episodes | Minutes |
| --- | ---: | ---: | ---: |
| 31 | 2304 | 36 | 6.7 |
| 47 | 2304 | 36 | 7.6 |
| 59 | 2304 | 36 | 6.9 |

All saved policies passed the finite-parameter and changed-parameter checks.
Training curves in the demo are five-episode rolling averages of shaped return.
They are diagnostic and are not held-out performance evidence.

## Evaluation protocol

The experiment evaluates each of the three policies and the four baselines on
12 episodes, giving 84 evaluation episodes.
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
| PPO seed 31 | 12 | 1/0/11 | -2.250 | [-3.083, -1.417] | 0.250 |
| PPO seed 47 | 12 | 0/1/11 | -2.333 | [-3.000, -1.667] | 0.083 |
| PPO seed 59 | 12 | 1/1/10 | -2.000 | [-2.752, -1.167] | 0.333 |
| Normal HELIOS | 12 | 4/7/1 | 0.250 | [-0.083, 0.583] | 1.583 |
| Always low | 12 | 0/0/12 | -1.833 | [-2.500, -1.250] | 0.000 |
| Always high | 12 | 1/0/11 | -3.000 | [-4.083, -1.833] | 0.250 |
| Random | 12 | 1/2/9 | -1.417 | [-2.083, -0.667] | 0.417 |

PPO did not outperform the strongest fixed/random baseline on the primary metric. This is a negative experimental result, not evidence of an effective adaptive coach.

### Instruction usage in evaluation

Counts below are managerial decisions, not individual player actions. A policy
that repeatedly selects one instruction should not be described as learning
context-sensitive tactical adaptation solely because it is implemented with PPO.

| Controller | Normal | Low | High |
| --- | ---: | ---: | ---: |
| PPO seed 31 | 0 | 0 | 720 |
| PPO seed 47 | 0 | 713 | 7 |
| PPO seed 59 | 0 | 720 | 0 |
| Normal HELIOS | 720 | 0 | 0 |
| Always low | 0 | 720 | 0 |
| Always high | 0 | 0 | 720 |
| Random | 211 | 265 | 244 |

Observed policy collapse: PPO seed 31 chose High line for every evaluated decision; PPO seed 59 chose Low line for every evaluated decision. These policies did not demonstrate adaptive switching.

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

- Small fixed-budget experiment; no guarantee of learned advantage.
- Fixed-length episodes, not regulation matches; one opponent and left-side control.
- Same native seed does not give identical trajectories; comparisons use independent runs.
- Intervals are exploratory episode-bootstrap intervals, not proof of superiority.
- No pressing, substitutions, opponent-belief model or real-football transfer claim.
- The global observation is privileged relative to real coaching perception.
  Player-skill delegation bypasses RoboCup coach-message restrictions; no
  competition-compliance claim is made.
- Fixed teammate/opponent heuristics, simplified line targets, kickoff-side
  asymmetry, noisy native execution and limited training constrain conclusions.
- A changed neural network and a working replay are not evidence that the agent
  learned useful football management. The held-out comparison is the relevant
  evidence, and greater training could help or fail to help.

The project delivers a complete, narrowly scoped RL experiment. PPO did not outperform the strongest fixed/random baseline on the primary metric. This is a negative experimental result, not evidence of an effective adaptive coach.

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
