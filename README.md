# GAFFER-RL

> **Reduced finished-project deliverable:** use [Defensive Line Lab](LINE_COACH_README.md).
> It has three coaching actions, PPO checkpoints, held-out baseline comparisons
> and an offline replay demo. Launch with `python scripts/serve_line_coach.py`.
> The broader PITCHSIM project below is preserved as earlier development work.

*An RL football coach that infers the opposing manager's hidden game plan from the
run of play, counter-punches with macro-tactical changes, and can be challenged
live by a human coach.*

Full design document: [`GAFFER_RL_Research_and_Plan.md`](GAFFER_RL_Research_and_Plan.md)

---

## What this is

Football-AI research sits at two extremes: micro-control of individual players in
expensive physics simulators, or offline analytics on historical data. The
*manager's* layer — shape, block height, press, tempo, substitutions — is largely
untouched as a control problem, despite being the layer humans actually argue
about.

GAFFER-RL formulates in-match coaching as a **partially-observable semi-Markov
decision process** over a fast, calibrated 2D simulator. The agent makes ~18
macro-decisions per match, cannot see the opposing manager's plan, and must infer
it from observable statistics (PPDA, field tilt, pass profile).

**Three novelty pillars**

1. **Belief-conditioned coaching.** An explicit, interpretable belief over
   opponent archetypes, ablatable against blind and oracle agents.
2. **Empirical game-theoretic analysis.** Payoff matrix over tactics, Nash and
   α-Rank, testing whether football's rock–paper–scissors structure exists here.
3. **Human-in-the-loop demo.** A console where a human manages the opposing side
   in real time, plus a counterfactual "what-if" panel.

---

## Ownership

independently owned, so boundaries are explicit. Everyone must be able to
defend the **whole** system in review — weekly 30-minute cross-explain sessions
are mandatory.

| Owner | Area | Directories |
|---|---|---|
| **A** — Environment | PITCHSIM physics and event layer, calibration, low-level controller, archetype scripts | `pitchsim/` |
| **B** — Learning | Gym wrapper, MDP, masking, belief filter, PPO/DQN/tabular, self-play league, EGTA | `envs/`, `agents/`, `analysis/egta.py` |
| **C** — Interface & evidence | FastAPI backend, console, replay format, evaluation harness, compute, figures, report | `console/`, `analysis/`, `tests/` |

**The serial dependency is broken by design.** `pitchsim/stub.py` implements the
full `PitchSim` interface with a cheap generative model, so B and C build against
a working API from day one rather than waiting on the physics. When the real
engine lands it drops in behind an unchanged interface.

> ⚠️ **Nothing produced by `StubPitchSim` may appear in the write-up as a result.**
> `StubPitchSim.IS_STUB is True` exists so that downstream code and figures can
> assert against it.

---

## Phase status

Phases are gated by **evidence**, not dates. Full detail is in section 6 of the
plan. Evidence files carry a fingerprint of the simulator code and configuration;
the gate rejects stale results.

| Phase | Focus | Gate | Status |
|---|---|---|---|
| **0** | Foundations & de-risking | xG AUC > 0.75 · stub live | ✅ **PASSED** |
| **1** | Environment | realism, speed, separability, cycles | ✅ **PASSED** (visual signed off 22 Aug) |
| **2** | Learning | PPO > random, non-overlapping CIs | 🔴 2 of 4 checks failing at current budgets |
| **3** | Science | EGTA, ablations, exploitability | 🟡 3 of 4 measured, all at reduced budget |
| **4** | Consolidation | offline demo and rehearsal | 🟡 console and animation built; offline rehearsal pending |

> **Every Phase 2–3 number below comes from smoke-scale runs**, roughly 25×
> under the budgets declared in `config/train.yaml` (2k/20k/15k macro-steps
> against a declared 150k/400k/600k, with 20–40 evaluation matches against a
> declared 500 × 5 seeds). At this size the gates flip between runs. The
> pipeline is proven end to end; the science is not yet run.

### Gate 1 detail (22 Aug) — PASSED

| Condition | Current fingerprinted evidence |
|---|---|
| Calibration | ✅ 3.062 goals/match; 13.865 shots/team; 0.121 xG/shot |
| Throughput | ✅ 420+ matches/min at 48 environments; sustained floor 60 |
| Archetype separability | ✅ 0.927 accuracy; 0.850 worst recall; 2,160 windows |
| Non-transitive structure | ✅ 3 significant cycles at 32 matches/pairing; transitivity 0.358 |
| Looks like football | ✅ signed off 22 Aug against the current config fingerprint |

Sign-off is tied to the simulator fingerprint: any change to the fingerprinted
code or configuration marks it stale and the animation must be reviewed again
with `python scripts/check_gate.py --approve-visual "Reviewer name"`.

### Gate 2 detail (22 Aug) — BLOCKED

| Condition | Current evidence |
|---|---|
| Belief re-identifies a switch | ✅ mean 1.21 epochs, 86% recovered, 649 true switches |
| Console renders a real replay | ✅ `/simulate_policy` returns a schema-valid replay from a trained checkpoint |
| PPO beats random-tactic | 🔴 1.70 ppm [1.325, 2.075] vs random 1.349 [1.266, 1.490] — CIs overlap |
| Elo rises across generations | 🔴 1441 → 1432 → 1627; generation 1 fell below its parent |

Both failures are sample-size effects, not broken mechanics: an earlier run of
the same code gave 2.225 ppm and a monotone 1452 → 1524 → 1524. Note also that
the two Stage-2 entries are seed **siblings** of one Stage-1 parent rather than
successive generations, so this check does not yet demonstrate generational
improvement even when it passes.

### Gate 3 detail (22 Aug)

| Condition | Current evidence |
|---|---|
| EGTA payoff + Nash/α-Rank | ✅ Nash peak High-press 4-3-3 (0.44); α-Rank peak the same (0.38) |
| Exploitability probe | 🟡 ran, but at 15k steps × 1 seed × 30 matches, not the declared 200k × 3 × 500 |
| All ablations run | 🔴 belief ordering is backwards: blind 2.00 > bayesian 1.80 > oracle 1.57 |
| Feature freeze 15 Oct | ⏳ pending |

The belief ordering is the headline concern. The oracle should lead; it trails
blind by 0.43 ppm. At 30 matches the intervals overlap heavily, so this is most
likely noise plus undertraining rather than a real finding — but Pillar 1 has no
support until a full-budget run says otherwise. `analysis/ablations.py` now
refuses to report a recovery percentage when the oracle gap is non-positive,
because dividing two negatives yields a healthy-looking number for a result
pointing the wrong way.

The γ^τ versus flat-γ comparison is valid again as of 22 Aug (1.80 vs 2.03 ppm).
It had been silently comparing γ^τ against itself: `load()` restored `smdp` from
the parent checkpoint, so the ablation trained a bit-identical policy under a tag
claiming otherwise. Fixed by `SMDPMaskablePPO.set_discounting`, with a regression
test.

Two declared ablations remain unimplemented and are reported as such rather than
skipped quietly: a recurrent belief mode and the action-masking ablation.

The fixes behind these results are structural: press closing speed no longer
changes every player's permanent running speed; PPDA counts pressure actions in
the correct pitch zone; the hidden matchup affects conversion rather than belief
observables; defensive-line height is now measured; and batched evidence jobs
checkpoint and resume under local CPU limits.

### Known evidence gaps

- **Baselines are fingerprint-stale.** `data/eval_baselines.json` was measured
  under learning fingerprint `998f31e0…`, which the current stack no longer
  matches. `ppo_vs_random_report` does not check that fingerprint, so Gate 2's
  PPO-versus-random comparison currently spans two different learning stacks.
  Only the `random` cell has been run; `static`, `greedy` and `tabular_q` have
  not.
- **Checkpoint lineage is strict, and rightly so.** Editing any fingerprinted
  file invalidates every checkpoint for further training, which forces a
  curriculum refresh. Freeze the code before starting long runs.
- **No mid-stage resume.** A stage is one uninterruptible `learn()` call, so a
  400k-step Stage 1 cannot be split across sessions or survive a sleeping
  laptop.
- **Two ablations unimplemented**: recurrent belief mode, action masking.

### Gate 0 evidence (15 Aug 2026)

| Check | Result |
|---|---|
| xG model, 5-fold GroupKFold by match | **AUC 0.7945** (per-fold 0.7935 ± 0.0210) vs required ≥ 0.75 |
| Calibration | mean predicted xG 0.1020 vs actual conversion 0.1020 |
| Reference ceiling | StatsBomb's own xG scores 0.8105 on the same shots; correlation 0.891 |
| Dataset | 8,536 shots · 871 goals · 350 matches · 8 competitions |
| Stub PITCHSIM | full 90′ match, replay schema valid — B and C unblocked |
| Contract tests | 16 passed |

Measured real-world calibration reference: **2.49** goals/match, **12.19**
shots/team, **0.0985** xG/shot. These replaced the plan's remembered values in
`config/pitchsim.yaml`.

---

## Setup

```bash
pip install -r requirements.txt

# Fetch StatsBomb open data and fit the xG model (Gate 0)
python -m pitchsim.statsbomb --fetch --max-matches 220
python -m pitchsim.xg --fit

# Verify the shared interfaces still hold — run before every push
python -m pytest tests/ -q

# Check gate status at any time
python scripts/check_gate.py 0
```

---

## Layout

```
config/         pitchsim.yaml · tactics.yaml · train.yaml   (FROZEN in Phase 0)
pitchsim/       types.py base.py replay.py  <- the shared contract
                stub.py                     <- Phase 0 placeholder engine
                statsbomb.py xg.py          <- real-data grounding
                engine.py controller.py     <- Phase 1 (the engine layer)
envs/           coach_env.py belief.py archetypes.py        (Phase 2, the learning layer)
agents/         baselines.py league.py                      (Phase 2, the learning layer)
analysis/       egta.py evaluate.py figures.py              (Phase 2–3, the interface layer)
console/        server.py static/                           (Phase 2, the interface layer)
scripts/        check_gate.py                               (all phases)
tests/          test_contract.py                            (guards the interfaces)
```

**`pitchsim/types.py` and `pitchsim/replay.py` are the load-bearing files.**
Changing them breaks all three workstreams simultaneously. Additions are cheap;
renames and removals are expensive and require bumping `SCHEMA_VERSION`.

---

## Compute

**This is a CPU workload, not a GPU one.** PITCHSIM is a NumPy physics loop and
the policy is an MLP over ~55 inputs; on an A100 the GPU would idle near zero
while one core does all the work. What is worth renting is *horizontal CPU
fan-out* for the embarrassingly-parallel work — the EGTA round-robin (~30k
matches), the ≥ 500-match evaluation cells, and the seed sweeps.

- **Modal:** CPU-only functions, no `gpu=` argument. Volume for checkpoints and replays.
- **Hugging Face Hub:** *tagged* checkpoints and final artefacts only, via
  `huggingface_hub`. It is a model hub, not a blob store — raw logs and per-tick
  replays stay on the Modal Volume.
- **GPU is justified in exactly two cases:** a JAX/Brax rewrite of the engine (a
  Phase-0 commitment or never; the default answer is no), and the
  natural-language coaching stretch goal, where an L4 is ample.

---

## Data & attribution

Uses [StatsBomb open data](https://github.com/statsbomb/open-data) for the xG fit
and calibration targets. If this repository is made public, StatsBomb's terms
require attribution — keep this section intact and cite them in the write-up.

Cached raw data lives in `data/statsbomb/raw/` and is **git-ignored**; regenerate
with the fetch command above.
