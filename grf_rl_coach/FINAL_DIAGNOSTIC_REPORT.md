# GRF depth coach: final diagnostic report

Date: 12 Oct 2026. Everything below comes from files in this folder: `results/final_eval.json` (raw per-match rows), `results/final_eval_analysis.json`, `results/final_eval_tables.md`, `results/q_training_analysis.json`, `results/dev_compare.json`, `demo/q100_seed800001/timeline.json`. Tables and figures are in `results/final_eval_tables.md` and `figs/`.

## 1. What was built and verified

A sequential RL coach on Google Research Football 2.10.3 with a patched engine exposing one native team-positioning control. Every 100 engine steps (30 decisions per 3,001-step match) the coach chooses **compact** (depth override 0.0), **native default** (override cleared: the unmodified engine) or **stretched** (override 1.0). All 11 players stay on GRF's built-in AI. Tabular Q-learning (243 states) trained for 100 iterations: 1,170 matches, 35,100 decisions, 3.5 million engine steps, 1.4 h. Environment tests (10) and learning-logic tests (4) pass; two logged training iterations were reproduced exactly in a fresh process; matches are deterministic functions of their seed.

## 2. Demonstration (not performance evidence)

Folder `demo/q100_seed800001/`. The iteration-100 checkpoint was loaded from disk and played one full match (demo seed 800001, outside every other seed range). Result 1-0 (one goal at decision 17).

| Check (from `timeline.json`, `verification`) | Result |
|---|---|
| Coach decisions | 30 (11 default, 10 compact, 9 stretched) |
| Native live depth equals the coach's action at the end of every decision | 30 of 30 (0.0 / 0.71 / 1.0) |
| Actions sent to players | 3,001 engine steps, **0** that were not `builtin_ai` |
| Rerun with same checkpoint and seed | identical decisions, scores, rewards, team shape |

Files: `annotated.avi` (banner shows current coach action, decision number, step, score), the native replay AVI and dump, `timeline.png/csv/json`, and `demo/README.md` with the commands. One 1-0 match says nothing about whether the coach helps.

## 3. Final held-out evaluation

Seeds 900000 to 900199, **200 matches per policy, identical seeds across policies**, none used for training, curves, the preliminary comparison or the demo. The policy was not changed or tuned after seeing these results. "Native" never calls the setter, so it is the true unmodified engine. Total compute 1.81 h wall-clock with 6 workers.

| Coach | Goal diff (mean ± SE) | W-D-L | Goals for | Goals against | Points/match | Return | Actions C/D/S |
|---|---|---|---|---|---|---|---|
| **Native GRF AI (unmodified)** | **+0.315 ± 0.100** | 97-47-56 | 1.10 | 0.79 | 1.69 | +0.314 | 0 / 1 / 0 |
| Always compact | +0.120 ± 0.101 | 76-63-61 | 1.14 | 1.02 | 1.46 | +0.118 | 1 / 0 / 0 |
| Always stretched | +0.035 ± 0.094 | 71-64-65 | 0.80 | 0.77 | 1.39 | +0.036 | 0 / 0 / 1 |
| Hand-written coach | +0.250 ± 0.101 | 78-66-56 | 1.07 | 0.82 | 1.50 | +0.249 | .27 / .66 / .07 |
| Random switching | +0.115 ± 0.099 | 75-58-67 | 1.01 | 0.90 | 1.42 | +0.114 | .34 / .33 / .34 |
| Q-learning (iter 100) | +0.210 ± 0.092 | 81-61-58 | 0.96 | 0.75 | 1.52 | +0.209 | .28 / .36 / .35 |
| Q-learning, shuffled obs. | +0.210 ± 0.090 | 81-65-54 | 1.02 | 0.81 | 1.54 | +0.209 | .28 / .36 / .36 |

Goal-difference differences (A minus B), 95% bootstrap CI (unpaired is primary; same-seed pairing barely helped, seed correlations −0.07 to +0.28):

| Comparison | Difference | Unpaired 95% CI |
|---|---|---|
| Q-learning minus native | **−0.105** | [−0.37, +0.17] |
| Q-learning minus always compact | +0.090 | [−0.18, +0.36] |
| Q-learning minus always stretched | +0.175 | [−0.09, +0.43] |
| Q-learning minus hand-written coach | −0.040 | [−0.30, +0.23] |
| Q-learning minus random switching | +0.095 | [−0.16, +0.36] |
| **Q-learning minus Q-learning with shuffled observations** | **0.000** | [−0.25, +0.26] |
| Always stretched minus native | −0.280 | [−0.55, −0.01] |
| Always compact minus native | −0.195 | [−0.48, +0.08] |
| Random switching minus native | −0.200 | [−0.48, +0.07] |

Process metrics (possession share, possession gains, time with the ball in our danger zone, opponent's furthest advance, team length) differ little between policies; see `results/final_eval_tables.md`. Always-compact let the ball spend more time in the danger zone (117 vs 97 steps per match) and conceded most (1.02 goals vs 0.79); always-stretched scored least (0.80 vs 1.10).

**Resolution:** with 200 matches each, a single comparison has SE of about 0.14, so only differences of roughly 0.4 goals per match or more are reliably detectable (80% power). Smaller effects can neither be confirmed nor excluded.

## 4. The questions

**Does the trained Q-learning coach beat any fixed tactic?** Its point estimate (+0.21) is above always-compact (+0.12) and always-stretched (+0.04) but below the native default (+0.315), which is itself a fixed tactic. None of these gaps is statistically significant. So: no demonstrated advantage over any fixed tactic, and it does not beat the best one.

**Does it outperform the unmodified GRF AI?** No. Difference −0.105 [−0.37, +0.17]: indistinguishable from native, with the point estimate below it. Note that the preliminary 30-match development comparison had suggested a larger deficit (−0.80, CI excluding zero); that did not replicate on 200 held-out matches, a reminder that 30 matches is far too few.

**Does the policy use observations beneficially?** There is no evidence it does. Feeding the same policy shuffled states gives an identical goal difference (+0.210 vs +0.210, difference 0.000 [−0.25, +0.26]) and an identical action mix. Training diagnostics point the same way: the greedy action is spread almost evenly over states (84 compact, 78 default, 81 stretched), with a median margin between best and second-best action of 0.09, and about 27 visits per state-action pair. This does **not** prove the table contains nothing; it shows any state dependence has no measurable football effect at this resolution.

**Consistent with the earlier native-depth experiments?** Yes. There, no fixed depth beat the native default on 21 development or 12 held-out defensive states (all intervals included zero, the default was the best fixed setting), and a dev-derived situation-dependent rule gave −0.108 [−0.287, +0.018] on held-out states. Here, fixed compact and stretched are both below native, always-stretched significantly so (−0.28 [−0.55, −0.01], unadjusted for multiple comparisons), and the learned policy lands between the fixed extremes and the default.

**Is DQN training justified?** On this evidence, no. The limiting factor is not the function approximator: depth changes team shape clearly but football outcomes barely, and match results are very noisy (SD about 1.3 goals). A neural network trained on the same weak signal and comparable budget is not expected to find structure the table could not. If an assignment requirement or presentation story needs a deep-RL comparison, a small DQN with the same interaction budget would be an honest negative-result comparison, but it should not be expected to change the conclusion and it costs about 1.5 h of compute plus 4 h evaluation.

## 5. Limitations

- One training run (one seed) for the Q-table; run-to-run variability of the learned policy is unmeasured.
- Resolution about 0.4 goals/match; smaller effects are undetected. 200 matches each, one opponent, one scenario (`11_vs_11_hard_stochastic`), our team on the left.
- Training iterations 42 to 46 used 6-match batches by mistake before the schedule was restored to 12; negligible.
- Reward includes a small potential-based territory term (0.1 × ball position); evaluation uses raw goal difference, and shaped and raw returns are nearly identical in every row.
- Preliminary comparison and diagnostic runs were done on separate seeds; the final test seeds were used once.
- A single control parameter (defensive depth) was tested; other native controls (for example width) were not.

## 6. Reproduce

```bash
bash grf_rl_coach/run.sh -W ignore -m unittest discover -s tests          # 14 tests
bash grf_rl_coach/run.sh train_q.py --iters 100 --workers 6 --episodes-per-iter 12 --tag q_main   # about 1.4 h
bash grf_rl_coach/run.sh final_eval.py --episodes 200 --workers 6          # 1.8 h, resumable
bash grf_rl_coach/run.sh analyze_final.py                                  # tables, figures
bash grf_rl_coach/run.sh demo.py --policy q --ckpt checkpoints/q_it100_final.npz --seed 800001 --out demo/q100_seed800001
```

Preserved: all checkpoints (iteration 10 to 100, the frozen iteration-41 and iteration-100 copies; sha256 of the evaluated checkpoint is stored in `results/final_eval.json`), the training log, the preliminary comparison and diagnostic outputs, and every earlier report.
