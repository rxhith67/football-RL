## Independent evaluation: each policy plays the native AI on the left; 30 matches per policy, seeds 3100000 to 3100029 (identical for all policies)

| Policy | Possession % (mean [95% CI]) | Matches >50% possession | Goal diff (mean [95% CI]) | W-D-L | Goals for / against | Points | Actions C/D/S |
|---|---|---|---|---|---|---|---|
| A: Q-learning, original reward (existing) | 51.8 [48.4, 55.2] | 18/30 (60%) | +0.333 [-0.30, +0.97] | 12-10-8 | 1.20 / 0.87 | 1.53 | 0.29/0.40/0.31 |
| B: Q-learning, possession reward | 52.7 [49.7, 55.7] | 18/30 (60%) | +0.367 [-0.03, +0.77] | 13-10-7 | 1.13 / 0.77 | 1.63 | 0.39/0.31/0.29 |
| Native GRF AI | 54.6 [51.0, 58.2] | 21/30 (70%) | -0.133 [-0.63, +0.37] | 10-7-13 | 0.70 / 0.83 | 1.23 | 0.00/1.00/0.00 |

Other measured outcomes (means per match; contested = nobody owns the ball, dead = kick-off/throw-in etc.):

| Policy | Interval possession % | Contested % | Dead-ball % | Possession-reward return | Original-reward return | Live recoveries | Time to recover (RMST) | Opp. advancement | Shots conceded (heuristic) |
|---|---|---|---|---|---|---|---|---|---|
| A: Q-learning, original reward (existing) | 51.8 | 41.8 | 1.7 | +1.21 | +0.33 | 9.07 | 89.8 | 0.244 | 1.90 |
| B: Q-learning, possession reward | 53.2 | 42.0 | 1.8 | +1.68 | +0.36 | 8.47 | 92.2 | 0.180 | 1.83 |
| Native GRF AI | 54.9 | 42.4 | 1.8 | +1.20 | -0.13 | 9.10 | 88.2 | 0.259 | 1.80 |

Paired differences (same seeds; A minus B; mean [95% CI], sign-flip p; Holm-adjusted p for the pre-specified family where shown):

- **A_q_original minus native**: possession -2.80 pts [-6.61, +0.91] (p = 0.167); goal diff +0.467 [-0.27, +1.23] (p = 0.267); goals against +0.033 [-0.37, +0.43]
- **B_q_possession minus native**: possession -1.98 pts [-5.68, +1.75] (p = 0.321, Holm p 0.382); goal diff +0.500 [-0.17, +1.17] (p = 0.191, Holm p 0.382); goals against -0.067 [-0.53, +0.40]
- **B_q_possession minus A_q_original**: possession +0.83 pts [-2.52, +4.21] (p = 0.630); goal diff +0.033 [-0.67, +0.77] (p = 1.000); goals against -0.100 [-0.47, +0.27]
- **B_q_possession minus A_q_original  (RQ1: possession reward, Q-learning)**: possession +0.83 pts [-2.61, +4.07] (p = 0.640); goal diff +0.033 [-0.67, +0.80] (p = 1.000); goals against -0.100 [-0.43, +0.23]
