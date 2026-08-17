# Project Abstract (draft)

**GAFFER-RL: Belief-Conditioned Macro-Tactical Coaching as a Partially-Observable
Semi-Markov Decision Process**

Status: **draft.**
submitted until the corresponding gate has produced them.

---

## Abstract

Modern football-AI research operates at two extremes. One line of work controls
individual players inside expensive physics simulators — Google Research Football
and its successors such as TiZero require cluster-scale compute to master 11-a-side
play. The other analyses historical data offline, as in DeepMind and Liverpool FC's
TacticAI, which predicts and generates corner-kick routines but is neither a
reinforcement-learning system nor open source. The layer in between — the
*manager's* decision layer of formation, block height, pressing intensity, tempo and
substitutions — has been largely ignored as a control problem, despite being the
layer that human football discourse actually concerns itself with.

GAFFER-RL formulates in-match coaching as a **partially-observable semi-Markov
decision process** over PITCHSIM, a fast 2D football simulator calibrated against
StatsBomb open data. The agent issues macro-tactical directives at roughly eighteen
decision epochs per match — on a five-minute schedule, and on triggers such as goals
and red cards — over a factored action space of 810 combinations. Because epoch
durations vary, transitions are discounted by γ^τ rather than a flat γ, making the
semi-Markov structure explicit rather than incidental. The low-level player policy
is a fixed, hand-written controller: in the language of Sutton, Precup and Singh's
options framework, the coach learns a policy *over* options whose intra-option
policies are given.

The central difficulty is that **the opposing manager's game plan is hidden state**.
It is never observed; only its statistical footprint — PPDA, field tilt, defensive
line height, pass-length profile — is visible. The agent therefore maintains an
explicit Bayesian belief over K = 6 opponent tactical archetypes and conditions its
policy on that belief. Scripted opponents change plan mid-match according to
state-dependent rules, so the filtering problem does not degenerate into a
one-shot classification solved by minute ten.

We train with maskable PPO under a three-stage curriculum culminating in
prioritised fictitious self-play, using a potential-based shaping term
Φ(s) = w·(xG_for − xG_against) whose policy invariance follows from Ng, Harada and
Russell. The resulting tactic space is then analysed with **empirical game theory**:
we construct the payoff matrix over archetypes and trained checkpoints, compute Nash
equilibria and α-Rank, and test whether the environment's non-transitive
("rock–paper–scissors") structure is one the learned coach exploits by playing a
best-response *mixture* rather than a single dominant plan.

Evaluation is pre-registered before training: ≥ 500 matches per agent–opponent
cell with 95% bootstrap confidence intervals, ≥ 5 training seeds with the spread
plotted rather than hidden behind a mean, and a declared-budget exploitability
probe. The headline ablation contrasts blind, Bayesian-belief, recurrent and
oracle agents, isolating the value of opponent inference. Deliverables include a
sim-versus-real calibration table, and a browser-based coach's console in which a
human can manage the opposing side against the trained agent in real time and
replay counterfactual "what-if" scenarios.

**Contribution.** To our knowledge this is the first open, reproducible RL
formulation of football management at the tactical layer, with opponent inference
treated as a first-class POMDP problem and the tactic space characterised
game-theoretically rather than by win rate alone.

---

## Key references

1. Kurach et al. (2019), *Google Research Football*, arXiv:1907.11180
2. Tuyls et al. (2021), *Game Plan: What AI can do for Football*, JAIR 71
3. Wang et al. (2024), *TacticAI*, Nature Communications
4. Lin et al. (2023), *TiZero*, AAMAS
5. Sutton, Precup & Singh (1999), *Between MDPs and semi-MDPs*, AIJ 112
6. Ng, Harada & Russell (1999), *Policy invariance under reward transformations*, ICML
7. Vinyals et al. (2019), *Grandmaster level in StarCraft II*, Nature 575
8. Omidshafiei et al. (2019), *α-Rank*, Scientific Reports

---

## Notes before submission

- Replace ⟨TBC⟩ values once Gate 1 and Gate 2 have produced them.
- The claim "first open, reproducible RL formulation of football management at the
  tactical layer" is defensible against the surveyed literature but should be
  softened to "to our knowledge" phrasing throughout — which it is.
- Do **not** cite any figure produced by `StubPitchSim`.
- Confirm the coordinator has signed off on the custom-environment route; the
