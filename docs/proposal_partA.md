# Project Proposal — content for the registration template

Copy each block into the matching field of the .docx.
Proposal only: no run outputs, measurements or results anywhere in this document.

---

## Project Title

**GAFFER-RL: Belief-Conditioned Macro-Tactical Coaching for Football**

*(shorter alternative: "GAFFER-RL: Learning to Manage a Football Match Under
Hidden Opponent Tactics")*

---

## Team Name

`<fill in>`

---

## Team Members

| Student Name | Student ID | Email ID |
|---|---|---|
| Rohith Syam | `<fill in>` | rohith4admissions@gmail.com |
| `<fill in>` | `<fill in>` | `<fill in>` |
| `<fill in>` | `<fill in>` | `<fill in>` |

---

## Abstract (~200 words)

Football-AI research operates at two extremes: micro-control of individual
players inside expensive physics simulators, or offline analysis of historical
event data. The manager's decision layer — formation, defensive line height,
pressing intensity, tempo and substitutions — remains largely unexplored as a
control problem, despite being the layer human football discourse actually
concerns itself with.

This project formulates in-match coaching as a partially-observable semi-Markov
decision process over a purpose-built two-dimensional football simulator
calibrated against public match data. The agent issues macro-tactical directives
at roughly eighteen decision epochs per match, over a factored action space of
810 combinations. Because epoch durations vary, transitions are discounted by
gamma raised to the dwell time rather than by a flat discount factor. The
low-level player policy is a fixed, hand-written controller, so the coach learns
a policy over options rather than controlling individual players.

The central difficulty is that the opposing manager's plan is hidden state,
observable only through its statistical footprint, such as pressing intensity,
field tilt and pass-length profile. The agent maintains an explicit Bayesian
belief over six opponent archetypes and conditions its policy on that belief. We
train with maskable PPO under a curriculum ending in prioritised fictitious
self-play, then analyse the resulting tactic space using empirical game theory to
test whether the learned coach exploits its non-transitive structure.

---

## Dataset Details (if known, optional)

**StatsBomb Open Data** — https://github.com/statsbomb/open-data

A free, publicly released collection of football event data covering men's and
women's competitions, including domestic leagues and international tournaments.
Each match provides a time-stamped event stream, and selected competitions add
360 freeze-frames giving the positions of visible players at the moment of an
event.

Two intended uses:

1. **Expected-goals model.** Shot events, together with their freeze-frames, give
   shot location, the angle subtended by the goal, defenders in the shooting
   cone, nearest-defender distance and goalkeeper position. These are used to fit
   a shot-quality model, so that goal probability in the simulator is grounded in
   real football rather than hand-specified.

2. **Simulator calibration targets.** Aggregate match statistics — goals per
   match, shots per team, expected goals per shot, pass completion — are derived
   from the same source and used as acceptance bands the simulator must reproduce
   before any learning results are trusted.

No private, licensed or personally identifying data is used. StatsBomb's terms
require attribution, which will be carried in the project repository and report.

---

## Additional Support Required (if any)

None anticipated.

The workload is CPU-bound rather than GPU-bound: the simulator is vectorised
NumPy and the policy network is a small multilayer perceptron over a
low-dimensional observation, so a GPU allocation would sit idle. Development and
training are expected to run on a standard laptop.

Optional cloud CPU capacity (for example Modal) may be used to parallelise
evaluation sweeps and tournament play, but this is a convenience rather than a
requirement for the core result.
