# Does the trained Q-learning coach improve possession or defence? Exploratory secondary analysis

Date: 12 Oct 2026. **Exploratory.** The primary result (goal difference, 200 held-out matches) was a null; these are secondary statistics examined after seeing that, so they cannot count as proof of success and nothing was pre-registered. All 17 measured statistics are reported, including those where the RL coach is worse. Files: `results/secondary_tables.md` (full tables), `results/secondary_analysis.json`, raw per-step streams `results/secondary/*.pkl`, figures `figs/secondary_*.png`.

## 1. What was measured and how

The trained policy, rewards, engine and tactical actions were not changed. A read-only spy on the observation returned after each engine step recorded the ball, possession and score stream (`measure.py`); every statistic is computed offline from those arrays (`metrics.py`, definitions fixed before the run).

| Statistic | Definition |
|---|---|
| Possession share | our owned steps / (ours + opponent owned steps); loose-ball steps excluded. A debounced version also carries possession through loose balls |
| Debounced possession | a team holds possession only after owning the ball for 5 consecutive engine steps; loose balls and shorter touches do not switch it. This removes rapid flicker (raw: 357 team-to-team ownership flips in 24 calibration matches versus 572 debounced switches that also include recoveries through a loose ball) |
| Live recovery | debounced switch opponent to us with **no dead ball and no goal** in between (throw-ins, kick-offs etc. are counted separately as dead-ball recoveries) |
| Time to recover | per opponent possession spell: mean of min(duration, 150 steps) ("RMST"), with spells ended by goal, dead ball or match end treated as censored; mean of completed live recoveries is also shown but is biased by censoring |
| Opponent advancement | mean and max of (−ball x) while the opponent owns the ball; share of their possession spent in our defensive third; steps with the ball in our danger zone (x < −0.76, \|y\| < 0.22) not owned by us |
| Shots / on target (**heuristic**) | GRF exposes no shot events. A shot-like event is an opponent release at speed ≥ 0.040 two steps later, heading at our goal from within 0.65 of the goal line; on target if its projected path crosses the goal mouth. Calibrated on 24 separate native matches (seeds 810000+): flags a shot before 28 of 33 goals (85%). Precision is unknown, so use for comparisons between coaches only |
| Goals | final score |

**Data:** 150 matches per policy on fresh seeds **1,000,000 to 1,000,149**, never used for training (100000+), curves (500000+), preliminary comparison (700000+), demo (800001), calibration (810000+) or the final test (900000+). Identical seeds across policies; paired comparisons use per-seed differences. Seven policies: native (never calls the setter), Q-learning iteration 100, always compact, always stretched, hand-written, random switching, and a shuffled-observation control for Q-learning (added to answer the situation-dependence question). Intervals are 95% percentile bootstraps over matches; p-values are paired sign-flip permutation tests; Holm adjustment covers the 17 Q-versus-native tests.

## 2. Comparison table (mean ± SE per match; 150 matches each)

| Statistic (better) | Native | **Q-learning** | Always compact | Always stretched | Hand-written | Random switching | Q, shuffled obs. |
|---|---|---|---|---|---|---|---|
| Possession share, debounced (↑) | 0.524 ± 0.007 | **0.520 ± 0.008** | 0.549 ± 0.009 | 0.482 ± 0.008 | 0.533 ± 0.008 | 0.529 ± 0.009 | 0.514 ± 0.007 |
| Live recoveries per match (↑) | 9.06 ± 0.20 | **8.79 ± 0.19** | 8.11 ± 0.21 | 8.68 ± 0.18 | 9.00 ± 0.20 | 8.54 ± 0.21 | 8.75 ± 0.17 |
| Recoveries per 1000 opp. steps (↑) | 6.64 ± 0.20 | **6.37 ± 0.18** | 6.28 ± 0.20 | 5.88 ± 0.17 | 6.79 ± 0.21 | 6.35 ± 0.21 | 6.21 ± 0.16 |
| Time to recover, RMST, steps (↓) | 90.1 ± 1.3 | **90.5 ± 1.2** | 89.6 ± 1.2 | 93.1 ± 1.3 | 90.1 ± 1.3 | 90.6 ± 1.3 | 92.2 ± 1.1 |
| Opp. advancement, mean (↓) | 0.234 ± 0.009 | **0.238 ± 0.010** | 0.201 ± 0.010 | 0.263 ± 0.008 | 0.221 ± 0.010 | 0.213 ± 0.009 | 0.231 ± 0.010 |
| Opp. possession in our third (↓) | 0.458 ± 0.010 | **0.475 ± 0.010** | 0.445 ± 0.009 | 0.515 ± 0.009 | 0.452 ± 0.010 | 0.440 ± 0.009 | 0.471 ± 0.009 |
| Danger-zone steps (↓) | 105.6 ± 4.3 | **105.9 ± 4.4** | 108.7 ± 4.3 | 95.0 ± 4.0 | 109.2 ± 4.9 | 100.8 ± 4.5 | 105.8 ± 4.5 |
| Shots conceded, heuristic (↓) | 1.55 ± 0.10 | **1.77 ± 0.11** | 1.79 ± 0.10 | 1.81 ± 0.11 | 1.67 ± 0.10 | 1.69 ± 0.11 | 1.74 ± 0.10 |
| Shots on target conceded, heuristic (↓) | 1.15 ± 0.09 | **1.34 ± 0.10** | 1.17 ± 0.08 | 1.40 ± 0.10 | 1.16 ± 0.08 | 1.13 ± 0.09 | 1.27 ± 0.08 |
| Goals conceded (↓) | 0.83 ± 0.07 | **0.79 ± 0.07** | 0.93 ± 0.07 | 0.70 ± 0.07 | 0.71 ± 0.07 | 0.78 ± 0.07 | 0.83 ± 0.08 |
| Goal difference (↑) | 0.11 ± 0.11 | **0.14 ± 0.11** | 0.13 ± 0.12 | 0.21 ± 0.11 | 0.35 ± 0.10 | 0.31 ± 0.11 | 0.29 ± 0.12 |

Other measured statistics (possession losses, completed-only recovery time, max advancement, goals scored, shots taken) are in `results/secondary_tables.md`. Note the native goal difference here (+0.11) is much lower than in the earlier 200-match evaluation (+0.315): the same policy differs by about 0.2 between two independent samples, which is the noise level of this measure.

## 3. Q-learning minus native, paired (95% CI; all 17 metrics)

No interval excludes zero, and every Holm-adjusted p-value is 1.000. Point estimates are worse than native on 14 of 17 statistics:

| Statistic | Difference [95% CI] | p |
|---|---|---|
| Possession share, debounced | −0.004 [−0.023, +0.015] | 0.69 |
| Live recoveries per match | −0.27 [−0.78, +0.23] | 0.30 |
| Time to recover, RMST | +0.4 [−2.9, +3.6] steps | 0.81 |
| Opp. advancement, mean | +0.004 [−0.019, +0.028] | 0.73 |
| Danger-zone steps | +0.3 [−10.6, +11.2] | 0.96 |
| Shots conceded (heuristic) | +0.22 [−0.05, +0.49] | 0.12 |
| Shots on target conceded (heuristic) | +0.19 [−0.05, +0.43] | 0.14 |
| Goals conceded | −0.04 [−0.22, +0.14] | 0.72 |
| Goal difference | +0.03 [−0.24, +0.29] | 0.88 |

Only possession losses (−0.28), goals conceded and goal difference have point estimates in Q-learning's favour, all tiny and unreliable.

## 4. Answers

**1. Does the RL coach improve any football statistic compared with native GRF AI?** No measurable improvement. Three of 17 point estimates favour it (fewer possession losses, fewer goals conceded, slightly higher goal difference), all well inside noise; the other 14 point estimates are worse, notably shots conceded (+0.22) and shots on target conceded (+0.19), though not significantly.

**2. Is any improvement reliable?** No. No unadjusted interval excludes zero and every Holm-adjusted p is 1.0. With 17 statistics and several comparisons, a few "significant" results would be expected by chance anyway.

**3. Does it outperform the strongest fixed-positioning strategy?** No. The best fixed setting depends on the statistic (compact for possession and opponent advancement, native for recoveries and shots conceded, stretched for goals conceded and danger-zone time), and the Q coach's point estimate is worse than the best fixed setting on all nine key statistics in the table above. Its behaviour sits between the fixed extremes, as expected for a policy that uses all three actions about equally. Direct paired comparisons:
- versus always compact: Q has fewer possession (−0.029 [−0.054, −0.005]) and larger opponent advancement (+0.037 [+0.010, +0.064]) but more live recoveries (+0.68 [+0.15, +1.22]);
- versus always stretched: Q has more possession (+0.038 [+0.013, +0.062]) and less opponent possession in our third (−0.040 [−0.066, −0.013]) but higher max opponent advancement (+0.019 [+0.005, +0.034]).
These unadjusted differences reflect the physical effect of the two shapes (compact keeps the ball more and concedes ground less; stretched does the opposite), not a learned skill.

**4. Is there evidence its decisions depend usefully on the match situation?** Dependent: yes. Useful: no.
- *Dependence:* the chosen action is statistically associated with the situation (mutual information with ball zone 0.055 bits, score state 0.041, match phase 0.045, all permutation p = 0.003) while random switching and the shuffled copy show none (p ≥ 0.2). The hand-written coach shows much stronger dependence (0.10 to 0.36 bits).
- *Usefulness:* feeding Q-learning shuffled observations changes no statistic reliably (all p > 0.09; goal difference −0.15 [−0.42, +0.12]), and a direct-method check, which uses the randomised random-switching data to estimate each action's effect on danger-zone steps per situation, predicts no benefit from Q's choices (+0.015 [−0.008, +0.037]; negative would be better). That check ignores lingering effects of earlier actions and does not include uncertainty in the estimated table, so the intervals are too narrow; treat it as directional only.

## 5. Limitations

- Exploratory; 17 statistics × 10 comparisons, no pre-registration. Unadjusted intervals are shown in the tables, adjusted only for the Q-versus-native family.
- 150 matches per policy; standard errors of 0.1 goals and 0.2 recoveries mean only larger effects can be detected.
- Shot statistics are heuristic with 85% recall on calibration goals and unknown precision; the possession and recovery definitions involve choices (5-step debounce, 150-step cap) that were fixed in advance but are not unique.
- Opponent advancement and shape statistics depend directly on the fixed policies' physical shape, so they mostly describe the settings, not coaching skill.
- One training run, one opponent and scenario.
- During the evaluation, two other jobs from a second session (`stats_collect.py`, `head_to_head_eval.py`) shared the machine and slowed it; results are unaffected because matches are deterministic per seed, but wall-clock time was longer.

## 6. Reproduce

```bash
bash grf_rl_coach/run.sh measured_eval.py --out secondary --seed0 1000000 --episodes 150 --workers 6 \
     --policies native,q,compact,stretched,hand,random,q_shuffled      # resumable, about 2 h on a loaded machine
bash grf_rl_coach/run.sh analyze_secondary.py                         # tables + figures
# shot-detector calibration (separate seeds): measured_eval.py --out calib --seed0 810000 --episodes 24 --policies native ; calibrate_shots.py ; check_metrics_on_calib.py
```

Figures for the supervisor discussion: `figs/secondary_overview.png` (nine key statistics by coach), `figs/secondary_q_vs_native_effects.png` (Q-learning versus native on every statistic), `figs/secondary_situation_dependence.png` (action mix by situation and the direct-method check).
