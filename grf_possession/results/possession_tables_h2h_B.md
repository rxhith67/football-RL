
## Head-to-head: RL team vs native AI on both sides, 30 seed pairs (60 matches per policy), seeds 3200000 to 3200029

| RL policy | Side | Matches | W-D-L | RL goals | Native goals | RL goal diff [95% CI] | RL possession % [95% CI] | Native possession % |
|---|---|---|---|---|---|---|---|---|
| B: Q-learning, possession reward | left | 30 | 12-7-11 (40/23/37%) | 0.867 | 0.900 | -0.033 [-0.47, +0.37] | 50.7 [47.7, 53.8] | 49.3 |
| B: Q-learning, possession reward | right | 30 | 8-7-15 (27/23/50%) | 0.633 | 1.233 | -0.600 [-1.13, -0.10] | 48.9 [45.5, 52.3] | 51.1 |
| B: Q-learning, possession reward | both | 60 | 20-14-26 (33/23/43%) | 0.750 | 1.067 | -0.317 [-0.67, +0.00] | 49.8 [47.7, 51.8] | 50.2 |

Seed-pair outcomes (combined goal difference of the two legs; win/draw/loss for the RL policy):

- B: Q-learning, possession reward: 10-6-14 of 30 seed pairs

Side contrast (left leg minus right leg, per seed pair). For `native` both legs are the same match seen from either side, so this equals twice the intrinsic left-side advantage of the scenario (right-side players have difficulty 0.95 versus 1.0 on the left):

- B: Q-learning, possession reward: goal diff +0.567 [-0.03, +1.17]; possession +1.83 pts [-3.21, +6.89]

Pair-mean differences between RL policies and native / original Q (paired over seed pairs):

