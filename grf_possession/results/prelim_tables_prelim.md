# PRELIMINARY evaluation: 20 fresh seed pairs, seeds 6,000,000 to 6,000,019 (folder eval_h2h_prelim)

Final iteration-100 policies, greedy (no exploration). Preliminary: small sample, intervals are wide; not for claims of improvement unless an interval excludes zero.

The two tables below use OVERLAPPING match data: the fixed-left matches are the left legs of the two-sided run (same seeds, deterministic engine).


## Table 1. Fixed-left comparison (20 matches per policy; policy coaches the LEFT team, opponent = unmodified native AI on the right)

| Policy | W-D-L (win/draw/loss %) | Goals for | Goals against | Mean goal diff [95% CI] | Possession % [95% CI] | Matches >50% possession |
|---|---|---|---|---|---|---|
| Native GRF | 5-11-4 (25/55/20%) | 0.80 | 0.70 | +0.10 [-0.30, +0.55] | 51.3 [48.1, 54.4] | 11/20 |
| A  Q-learning, original reward | 8-6-6 (40/30/30%) | 1.15 | 0.85 | +0.30 [-0.40, +1.00] | 51.1 [48.1, 53.8] | 11/20 |
| B  Q-learning, possession reward | 7-6-7 (35/30/35%) | 1.10 | 0.75 | +0.35 [-0.20, +1.00] | 54.7 [51.6, 57.6] | 16/20 |
| C  Double Q, possession reward | 8-3-9 (40/15/45%) | 1.10 | 0.95 | +0.15 [-0.50, +0.80] | 54.1 [51.2, 57.3] | 13/20 |

Paired differences, same seeds (positive = first policy higher), 95% bootstrap CI over seeds, unadjusted sign-flip p:

| Comparison | Goal diff difference [95% CI] | p | Possession difference, pts [95% CI] | p |
|---|---|---|---|---|
| A vs Native | +0.20 [-0.70, +1.15] | 0.760 | -0.2 [-4.5, +4.1] | 0.934 |
| B vs Native | +0.25 [-0.40, +0.95] | 0.584 | +3.4 [-1.0, +7.8] | 0.149 |
| C vs Native | +0.05 [-0.60, +0.70] | 1.000 | +2.9 [-0.7, +6.7] | 0.169 |
| B vs A | +0.05 [-0.80, +0.90] | 1.000 | +3.6 [-0.8, +8.1] | 0.142 |
| C vs A | -0.15 [-1.15, +0.85] | 0.849 | +3.1 [-1.7, +8.1] | 0.255 |
| C vs B | -0.20 [-0.95, +0.60] | 0.728 | -0.5 [-4.1, +3.0] | 0.772 |

Note: the right team has slightly weaker players in GRF (difficulty 0.95 vs 1.0), so the left side has a built-in advantage; fixed-left means all coached teams share it.

## Table 2. Two-sided evaluation (20 seed pairs = 40 matches per policy; policy coaches the left team in one leg and the right team in the other)

| Policy | Legs W-D-L (win/draw/loss %) | Left leg GD | Right leg GD | Pair-mean goal diff [95% CI] | Pair-mean possession % [95% CI] | Seed pairs won-drawn-lost (by combined GD) |
|---|---|---|---|---|---|---|
| Native GRF | 9-22-9 (22/55/22%) | +0.10 | -0.10 | +0.00 [+0.00, +0.00] | 50.0 [50.0, 50.0] | 0-20-0 |
| A  Q-learning, original reward | 14-12-14 (35/30/35%) | +0.30 | -0.40 | -0.05 [-0.55, +0.42] | 47.9 [45.8, 50.1] | 8-5-7 |
| B  Q-learning, possession reward | 10-12-18 (25/30/45%) | +0.35 | -1.00 | -0.33 [-0.82, +0.17] | 50.3 [48.0, 52.5] | 6-2-12 |
| C  Double Q, possession reward | 13-5-22 (32/12/55%) | +0.15 | -0.60 | -0.23 [-0.72, +0.30] | 50.2 [47.7, 52.7] | 7-3-10 |

Paired differences over seed pairs (pair-mean values; positive = first policy higher), 95% bootstrap CI, unadjusted sign-flip p:

| Comparison | Pair-mean goal diff difference [95% CI] | p | Possession difference, pts [95% CI] | p |
|---|---|---|---|---|
| A vs Native | -0.05 [-0.53, +0.42] | 0.920 | -2.1 [-4.2, +0.0] | 0.081 |
| B vs Native | -0.33 [-0.82, +0.17] | 0.274 | +0.3 [-2.0, +2.6] | 0.812 |
| C vs Native | -0.23 [-0.75, +0.30] | 0.469 | +0.2 [-2.3, +2.8] | 0.866 |
| B vs A | -0.28 [-0.97, +0.45] | 0.508 | +2.3 [-1.3, +6.1] | 0.231 |
| C vs A | -0.17 [-0.88, +0.55] | 0.688 | +2.3 [-1.2, +6.1] | 0.261 |
| C vs B | +0.10 [-0.60, +0.85] | 0.847 | -0.1 [-3.7, +3.4] | 0.973 |

Native vs native: both legs are the same match seen from the two sides, so its pair-mean goal difference is exactly 0 and its pair-mean possession exactly 50%; the two-sided differences against native therefore equal each policy's own pair mean. Left minus right leg shows side sensitivity.

No multiplicity adjustment is applied in this preliminary report; with 6 comparisons x 2 metrics x 2 tables, a few small p-values are expected by chance.