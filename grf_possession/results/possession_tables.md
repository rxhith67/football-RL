## Independent evaluation: each policy plays the native AI on the left; 3 matches per policy, seeds 3000000 to 3000002 (identical for all policies)

| Policy | Possession % (mean [95% CI]) | Matches >50% possession | Goal diff (mean [95% CI]) | W-D-L | Goals for / against | Points | Actions C/D/S |
|---|---|---|---|---|---|---|---|
| Native GRF AI | 51.1 [48.8, 54.7] | 1/3 (33%) | +1.000 [-1.00, +2.00] | 2-0-1 | 1.33 / 0.33 | 2.00 | 0.00/1.00/0.00 |
| smoke_dq | 53.3 [51.7, 56.1] | 3/3 (100%) | -0.333 [-1.00, +1.00] | 1-0-2 | 1.00 / 1.33 | 1.00 | 0.28/0.58/0.14 |

Other measured outcomes (means per match; contested = nobody owns the ball, dead = kick-off/throw-in etc.):

| Policy | Interval possession % | Contested % | Dead-ball % | Possession-reward return | Original-reward return | Live recoveries | Time to recover (RMST) | Opp. advancement | Shots conceded (heuristic) |
|---|---|---|---|---|---|---|---|---|---|
| Native GRF AI | 51.2 | 44.1 | 1.8 | +2.36 | +1.00 | 8.33 | 107.6 | 0.305 | 2.67 |
| smoke_dq | 54.4 | 41.8 | 1.7 | +0.65 | -0.34 | 10.67 | 76.2 | 0.107 | 1.67 |

Paired differences (same seeds; A minus B; mean [95% CI], sign-flip p; Holm-adjusted p for the pre-specified family where shown):

- **smoke_dq minus native**: possession +nan pts [+nan, +nan] (p = nan); goal diff +nan [+nan, +nan] (p = nan); goals against +nan [+nan, +nan]

## Head-to-head: RL team vs native AI on both sides, 2 seed pairs (4 matches per policy), seeds 4000000 to 4000001

| RL policy | Side | Matches | W-D-L | RL goals | Native goals | RL goal diff [95% CI] | RL possession % [95% CI] | Native possession % |
|---|---|---|---|---|---|---|---|---|
| Native GRF AI | left | 2 | 2-0-0 | 2.000 | 0.000 | +2.000 [+2.00, +2.00] | 61.5 [54.4, 68.6] | 38.5 |
| Native GRF AI | right | 2 | 0-0-2 | 0.000 | 2.000 | -2.000 [-2.00, -2.00] | 38.5 [31.4, 45.6] | 61.5 |
| Native GRF AI | both | 4 | 2-0-2 | 1.000 | 1.000 | +0.000 [+0.00, +0.00] | 50.0 [50.0, 50.0] | 50.0 |
| A_orig | left | 2 | 1-1-0 | 1.000 | 0.500 | +0.500 [+0.00, +1.00] | 51.2 [49.9, 52.4] | 48.8 |
| A_orig | right | 2 | 1-0-1 | 2.000 | 1.000 | +1.000 [-1.00, +3.00] | 39.3 [34.4, 44.1] | 60.7 |
| A_orig | both | 4 | 2-1-1 | 1.500 | 0.750 | +0.750 [-0.50, +2.00] | 45.2 [42.2, 48.3] | 54.8 |
| smoke_dq | left | 2 | 1-0-1 | 1.000 | 1.500 | -0.500 [-3.00, +2.00] | 60.0 [51.4, 68.6] | 40.0 |
| smoke_dq | right | 2 | 1-0-1 | 0.500 | 1.500 | -1.000 [-3.00, +1.00] | 33.5 [21.6, 45.5] | 66.5 |
| smoke_dq | both | 4 | 2-0-2 | 0.750 | 1.500 | -0.750 [-1.00, -0.50] | 46.8 [45.1, 48.5] | 53.2 |

Side contrast (left leg minus right leg, per seed pair). For `native` both legs are the same match seen from either side, so this equals twice the intrinsic left-side advantage of the scenario (right-side players have difficulty 0.95 versus 1.0 on the left):

- Native GRF AI: goal diff +4.000 [+4.00, +4.00]; possession +23.02 pts [+8.88, +37.16]
- A_orig: goal diff -0.500 [-2.00, +1.00]; possession +11.91 pts [+8.30, +15.52]
- smoke_dq: goal diff +0.500 [-4.00, +5.00]; possession +26.50 pts [+5.95, +47.04]

Pair-mean differences between RL policies and native / original Q (paired over seed pairs):

- **A_orig minus native (pair mean)**: goal diff +nan [+nan, +nan] (p = nan); possession +nan pts [+nan, +nan] (p = nan)
- **smoke_dq minus native (pair mean)**: goal diff +nan [+nan, +nan] (p = nan); possession +nan pts [+nan, +nan] (p = nan)
