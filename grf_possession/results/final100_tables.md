# Final evaluation: 100 fresh seed pairs, seeds 7,000,000-7,000,099, every policy on both sides = 200 matches per policy, 800 in total

Frozen iteration-100 checkpoints, greedy, no exploration, opponent = unmodified native GRF AI. Same seeds, scenario and settings for every policy.

Fixed-left and two-sided results use OVERLAPPING match data (the fixed-left matches are the left legs of the two-sided matches). Native vs native: both legs are the same match seen from the two sides, so its two-sided figures are exactly neutral (goal difference 0, possession 50%).


## Table 1. FIXED-LEFT: each policy coaches the LEFT team (100 matches each) against unmodified native GRF on the right

Left-side players are slightly stronger in GRF (difficulty 1.0 vs 0.95), so all left-side results include a built-in left advantage.

| Result | Native GRF | Q-learning A | Q-learning B | Double Q-learning C |
|---|---|---|---|---|
| Matches | 100 | 100 | 100 | 100 |
| Wins | 49 | 39 | 41 | 40 |
| Draws | 26 | 27 | 20 | 34 |
| Losses | 25 | 34 | 39 | 26 |
| Win percentage | 49.0% | 39.0% | 41.0% | 40.0% |
| Loss percentage | 25.0% | 34.0% | 39.0% | 26.0% |
| Goals per match | 1.10 | 0.99 | 1.11 | 1.01 |
| Goals conceded per match | 0.72 | 0.90 | 1.01 | 0.78 |
| Average goal difference | +0.38 | +0.09 | +0.10 | +0.23 |
|   95% CI (goal difference) | [+0.13, +0.63] | [-0.20, +0.37] | [-0.22, +0.42] | [-0.04, +0.50] |
| Average possession percentage | 53.9% | 51.8% | 50.2% | 51.4% |
|   95% CI (possession) | [52.3, 55.5] | [50.4, 53.4] | [48.6, 51.7] | [49.8, 53.0] |
| Matches with >50% possession | 68/100 | 56/100 | 46/100 | 50/100 |

## Table 1b. FIXED-LEFT head-to-head, coach on the left vs native on the right (100 matches per policy)

| Result | Q-learning A | Native opponent | Q-learning B | Native opponent | Double Q-learning C | Native opponent |
|---|---|---|---|---|---|---|
| Wins | 39 (39.0%) | 34 (34.0%) | 41 (41.0%) | 39 (39.0%) | 40 (40.0%) | 26 (26.0%) |
| Draws | 27 (27.0%) | 27 (27.0%) | 20 (20.0%) | 20 (20.0%) | 34 (34.0%) | 34 (34.0%) |
| Losses | 34 (34.0%) | 39 (39.0%) | 39 (39.0%) | 41 (41.0%) | 26 (26.0%) | 40 (40.0%) |
| Goals per match | 0.99 | 0.90 | 1.11 | 1.01 | 1.01 | 0.78 |
| Goal difference | +0.09 | -0.09 | +0.10 | -0.10 | +0.23 | -0.23 |
| Possession % | 51.8 | 48.2 | 50.2 | 49.8 | 51.4 | 48.6 |

## Table 2a. Coach on the RIGHT (100 matches each) against native GRF on the left

Right side is the disadvantaged side in this scenario.

| Result | Native GRF | Q-learning A | Q-learning B | Double Q-learning C |
|---|---|---|---|---|
| Matches | 100 | 100 | 100 | 100 |
| Wins | 25 | 30 | 31 | 22 |
| Draws | 26 | 23 | 32 | 33 |
| Losses | 49 | 47 | 37 | 45 |
| Win percentage | 25.0% | 30.0% | 31.0% | 22.0% |
| Loss percentage | 49.0% | 47.0% | 37.0% | 45.0% |
| Goals per match | 0.72 | 0.93 | 0.86 | 0.74 |
| Goals conceded per match | 1.10 | 1.14 | 1.09 | 1.11 |
| Average goal difference | -0.38 | -0.21 | -0.23 | -0.37 |
|   95% CI (goal difference) | [-0.63, -0.13] | [-0.51, +0.10] | [-0.50, +0.03] | [-0.63, -0.11] |
| Average possession percentage | 46.1% | 45.7% | 47.9% | 47.0% |
|   95% CI (possession) | [44.5, 47.8] | [44.3, 47.2] | [46.4, 49.5] | [45.4, 48.7] |
| Matches with >50% possession | 32/100 | 33/100 | 40/100 | 30/100 |

## Table 2b. RIGHT-side head-to-head, coach on the right vs native on the left (100 matches per policy)

| Result | Q-learning A | Native opponent | Q-learning B | Native opponent | Double Q-learning C | Native opponent |
|---|---|---|---|---|---|---|
| Wins | 30 (30.0%) | 47 (47.0%) | 31 (31.0%) | 37 (37.0%) | 22 (22.0%) | 45 (45.0%) |
| Draws | 23 (23.0%) | 23 (23.0%) | 32 (32.0%) | 32 (32.0%) | 33 (33.0%) | 33 (33.0%) |
| Losses | 47 (47.0%) | 30 (30.0%) | 37 (37.0%) | 31 (31.0%) | 45 (45.0%) | 22 (22.0%) |
| Goals per match | 0.93 | 1.14 | 0.86 | 1.09 | 0.74 | 1.11 |
| Goal difference | -0.21 | +0.21 | -0.23 | +0.23 | -0.37 | +0.37 |
| Possession % | 45.7 | 54.3 | 47.9 | 52.1 | 47.0 | 53.0 |

## Table 3. COMBINED left + right (200 matches per policy)

Native vs native: the two legs are one match seen from both sides, so every native row is exactly neutral.

| Result | Native GRF | Q-learning A | Q-learning B | Double Q-learning C |
|---|---|---|---|---|
| Matches | 200 | 200 | 200 | 200 |
| Wins | 74 | 69 | 72 | 62 |
| Draws | 52 | 50 | 52 | 67 |
| Losses | 74 | 81 | 76 | 71 |
| Win percentage | 37.0% | 34.5% | 36.0% | 31.0% |
| Loss percentage | 37.0% | 40.5% | 38.0% | 35.5% |
| Goals per match | 0.91 | 0.96 | 0.98 | 0.88 |
| Goals conceded per match | 0.91 | 1.02 | 1.05 | 0.94 |
| Average goal difference | +0.00 | -0.06 | -0.07 | -0.07 |
|   95% CI (goal difference) | [-0.19, +0.18] | [-0.27, +0.15] | [-0.28, +0.14] | [-0.26, +0.12] |
| Average possession percentage | 50.0% | 48.8% | 49.0% | 49.2% |
|   95% CI (possession) | [48.7, 51.3] | [47.7, 49.9] | [47.9, 50.2] | [48.1, 50.4] |
| Matches with >50% possession | 100/200 | 89/200 | 86/200 | 80/200 |

## Table 3b. COMBINED head-to-head (left + right, 200 matches per policy)

| Result | Q-learning A | Native opponent | Q-learning B | Native opponent | Double Q-learning C | Native opponent |
|---|---|---|---|---|---|---|
| Wins | 69 (34.5%) | 81 (40.5%) | 72 (36.0%) | 76 (38.0%) | 62 (31.0%) | 71 (35.5%) |
| Draws | 50 (25.0%) | 50 (25.0%) | 52 (26.0%) | 52 (26.0%) | 67 (33.5%) | 67 (33.5%) |
| Losses | 81 (40.5%) | 69 (34.5%) | 76 (38.0%) | 72 (36.0%) | 71 (35.5%) | 62 (31.0%) |
| Goals per match | 0.96 | 1.02 | 0.98 | 1.05 | 0.88 | 0.94 |
| Goal difference | -0.06 | +0.06 | -0.07 | +0.07 | -0.07 | +0.07 |
| Possession % | 48.8 | 51.2 | 49.0 | 51.0 | 49.2 | 50.8 |

## Table 4. PAIRED comparisons, FIXED-LEFT (100 matched seeds)

Positive = first policy higher. Win/loss differences are in percentage points. p = unadjusted sign-flip permutation p; Holm column covers the primary family (B/C vs native, goal difference and possession).

| Comparison | Goal diff difference [95% CI] | p | Possession pts [95% CI] | p | Win-rate pts [95% CI] | Loss-rate pts [95% CI] |
|---|---|---|---|---|---|---|
| Q-learning B vs Native GRF (primary) | -0.28 [-0.65, +0.09] | 0.161 | -3.7 [-5.8, -1.5] | 0.002 | -8.0 [-21.0, +4.0] | +14.0 [+2.0, +26.0] |
| Double Q-learning C vs Native GRF (primary) | -0.15 [-0.51, +0.21] | 0.446 | -2.5 [-4.7, -0.2] | 0.033 | -9.0 [-23.0, +5.0] | +1.0 [-12.0, +13.0] |
| Q-learning B vs Q-learning A | +0.01 [-0.36, +0.39] | 1.000 | -1.7 [-3.9, +0.4] | 0.134 | +2.0 [-10.0, +14.0] | +5.0 [-8.0, +17.0] |
| Double Q-learning C vs Q-learning B | +0.13 [-0.27, +0.52] | 0.552 | +1.2 [-0.9, +3.3] | 0.262 | -1.0 [-14.0, +12.0] | -13.0 [-26.0, +0.0] |
| Q-learning A vs Native GRF | -0.29 [-0.64, +0.07] | 0.124 | -2.0 [-4.1, +0.1] | 0.062 | -10.0 [-24.0, +4.0] | +9.0 [-2.0, +20.0] |
| Double Q-learning C vs Q-learning A | +0.14 [-0.20, +0.49] | 0.465 | -0.5 [-2.6, +1.7] | 0.673 | +1.0 [-12.0, +14.0] | -8.0 [-19.0, +3.0] |

Holm-adjusted p (primary family): Q-learning B vs Native GRF / goal diff 0.323; Q-learning B vs Native GRF / possession 0.006; Double Q-learning C vs Native GRF / goal diff 0.446; Double Q-learning C vs Native GRF / possession 0.100

## Table 5. PAIRED comparisons, TWO-SIDED (pair-mean over 100 seed pairs)

Positive = first policy higher. Win/loss differences are in percentage points. p = unadjusted sign-flip permutation p; Holm column covers the primary family (B/C vs native, goal difference and possession).

| Comparison | Goal diff difference [95% CI] | p | Possession pts [95% CI] | p | Win-rate pts [95% CI] | Loss-rate pts [95% CI] |
|---|---|---|---|---|---|---|
| Q-learning B vs Native GRF (primary) | -0.07 [-0.27, +0.14] | 0.561 | -1.0 [-2.0, +0.1] | 0.089 | -1.0 [-8.5, +6.5] | +1.0 [-7.0, +9.5] |
| Double Q-learning C vs Native GRF (primary) | -0.07 [-0.23, +0.09] | 0.418 | -0.8 [-2.0, +0.4] | 0.217 | -6.0 [-13.5, +1.5] | -1.5 [-8.5, +5.5] |
| Q-learning B vs Q-learning A | -0.01 [-0.26, +0.24] | 1.000 | +0.3 [-1.4, +1.9] | 0.755 | +1.5 [-7.0, +10.0] | -2.5 [-11.5, +6.5] |
| Double Q-learning C vs Q-learning B | -0.01 [-0.27, +0.26] | 1.000 | +0.2 [-1.4, +1.7] | 0.831 | -5.0 [-14.0, +4.0] | -2.5 [-12.0, +7.0] |
| Q-learning A vs Native GRF | -0.06 [-0.23, +0.12] | 0.548 | -1.2 [-2.3, -0.2] | 0.029 | -2.5 [-10.5, +6.0] | +3.5 [-3.0, +10.0] |
| Double Q-learning C vs Q-learning A | -0.01 [-0.26, +0.23] | 0.964 | +0.4 [-1.1, +2.0] | 0.576 | -3.5 [-12.5, +5.5] | -5.0 [-13.5, +3.5] |

Holm-adjusted p (primary family): Q-learning B vs Native GRF / goal diff 0.837; Q-learning B vs Native GRF / possession 0.354; Double Q-learning C vs Native GRF / goal diff 0.837; Double Q-learning C vs Native GRF / possession 0.651

## Table 6. PAIRED comparisons, RIGHT side only (100 matched seeds)

Positive = first policy higher. Win/loss differences are in percentage points. p = unadjusted sign-flip permutation p; Holm column covers the primary family (B/C vs native, goal difference and possession).

| Comparison | Goal diff difference [95% CI] | p | Possession pts [95% CI] | p | Win-rate pts [95% CI] | Loss-rate pts [95% CI] |
|---|---|---|---|---|---|---|
| Q-learning B vs Native GRF (primary) | +0.15 [-0.23, +0.51] | 0.459 | +1.8 [-0.5, +4.0] | 0.118 | +6.0 [-7.0, +18.0] | -12.0 [-25.0, +1.0] |
| Double Q-learning C vs Native GRF (primary) | +0.01 [-0.36, +0.37] | 1.000 | +0.9 [-1.3, +3.2] | 0.433 | -3.0 [-15.0, +9.0] | -4.0 [-17.0, +9.0] |
| Q-learning B vs Q-learning A | -0.02 [-0.41, +0.38] | 0.959 | +2.2 [+0.1, +4.3] | 0.039 | +1.0 [-12.0, +13.0] | -10.0 [-23.0, +4.0] |
| Double Q-learning C vs Q-learning B | -0.14 [-0.48, +0.20] | 0.460 | -0.9 [-3.0, +1.2] | 0.409 | -9.0 [-21.0, +3.0] | +8.0 [-6.0, +22.0] |
| Q-learning A vs Native GRF | +0.17 [-0.16, +0.50] | 0.354 | -0.4 [-2.6, +1.7] | 0.690 | +5.0 [-5.0, +15.0] | -2.0 [-14.0, +10.0] |
| Double Q-learning C vs Q-learning A | -0.16 [-0.53, +0.21] | 0.418 | +1.3 [-0.9, +3.5] | 0.240 | -8.0 [-19.0, +3.0] | -2.0 [-16.0, +11.0] |

Holm-adjusted p (primary family): Q-learning B vs Native GRF / goal diff 1.000; Q-learning B vs Native GRF / possession 0.473; Double Q-learning C vs Native GRF / goal diff 1.000; Double Q-learning C vs Native GRF / possession 1.000

## Reading guide

- An improvement is called reliable only if the paired 95% interval excludes zero (and, for the primary family, the Holm-adjusted p is below 0.05). All comparisons are shown, favourable or not.
- Possession is part of the training reward of B and C (not A or native), so a possession gain for them is the trained-for target; goal difference and win/loss rates are the outcome measures.
- Side sensitivity: compare Tables 1 (left) and 2a (right). Two-sided pair means cancel the built-in left-side advantage.
