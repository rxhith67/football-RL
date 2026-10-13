"""Calibrate the shot detector on SEPARATE calibration matches (seeds 810000+, native AI) before the secondary evaluation.
Looks at what the ball does between the scorer's last touch and the goal."""
import pickle
import numpy as np

res = pickle.load(open("results/calib/native_unmodified.pkl", "rb"))
rows, speeds_all = [], []
n_goal_us = n_goal_opp = 0
for ep, st, dec in res:
    owner, bx, by, vx, vy, gm, su, so, _ = st.T
    for t in range(1, len(st)):
        for team, col in ((0, su), (1, so)):
            if col[t] > col[t - 1]:
                (n_goal_us, n_goal_opp) = (n_goal_us + (team == 0), n_goal_opp + (team == 1))
                # last step before the goal where the scoring team owned the ball
                k = max([i for i in range(t) if owner[i] == team], default=None)
                if k is None: continue
                j = min(k + 2, t)                                   # ball state two steps after release
                sp = float(np.hypot(vx[j], vy[j]))
                tgt_x = -1.0 if team == 1 else 1.0                  # goal line the ball is heading to
                dx = tgt_x - bx[j]
                yproj = by[j] + (vy[j] / vx[j]) * dx if abs(vx[j]) > 1e-6 and np.sign(vx[j]) == np.sign(dx) else np.nan
                rows.append((team, int(k), int(t - k), float(bx[k]), float(by[k]), sp, float(yproj), int(gm[k])))
rows = np.array(rows)
print("goals found: ours", n_goal_us, "opp", n_goal_opp, "| with a release point", len(rows))
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print("columns: team, release_step, steps_release_to_goal, ball_x, ball_y, speed(2 steps after), y_at_goal_line, game_mode_at_release")
print(rows)
if len(rows):
    for team in (0, 1):
        r = rows[rows[:, 0] == team]
        if len(r): print("team", team, "n", len(r), "speed percentiles 0/10/50/100:", np.percentile(r[:, 5], [0, 10, 50, 100]).round(4),
                         "| release |x| percentiles:", np.percentile(np.abs(r[:, 3]), [0, 50, 100]).round(3), "| |yproj| max", np.nanmax(np.abs(r[:, 6])).round(3))
# generic: ball speed right after each opponent release (any kind) for comparison
rel = []
for ep, st, dec in res:
    owner, bx, by, vx, vy, gm, su, so, _ = st.T
    for t in range(1, len(st) - 3):
        if owner[t - 1] == 1 and owner[t] != 1 and gm[t] == 0:
            rel.append((np.hypot(vx[t + 2], vy[t + 2]), bx[t]))
rel = np.array(rel)
print("all opponent releases n", len(rel), "speed percentiles 10/25/50/75/90/99:", np.percentile(rel[:, 0], [10, 25, 50, 75, 90, 99]).round(4))
print("overall ball |speed| during play percentiles 50/90/99:", np.percentile(np.hypot(res[0][1][:, 3], res[0][1][:, 4]), [50, 90, 99]).round(4))
