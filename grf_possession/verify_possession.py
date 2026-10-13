"""Verify GRF's ball-ownership label (the possession detector's input) against the actual player/ball geometry in real
matches. Native AI vs native AI, 4 matches on seeds 777101-777104 (not used anywhere else). Writes results/possession_verification.json.
Questions: when GRF says a team owns the ball, is its owner player right at the ball and the nearest player? When nobody owns it,
is the ball really free (nobody close, or in the air)? How long do ownership spells last (flicker)? What are dead-ball steps?"""
import json
import numpy as np

from coach_env import GRFDepthCoachEnv, AI

env = GRFDepthCoachEnv(noop_calls=True)
rows = []
for seed in (777101, 777102, 777103, 777104):
    env.reset(seed=seed)
    done = False
    while not done:
        obs, _, done, _ = env.env.step([AI] * 11)
        r = obs[0]
        ball = np.array(r["ball"], float); L = np.array(r["left_team"], float); R = np.array(r["right_team"], float)
        dl = np.hypot(*(L - ball[:2]).T); dr = np.hypot(*(R - ball[:2]).T)
        rows.append((int(r["ball_owned_team"]), int(r["ball_owned_player"]), int(r["game_mode"]), ball[2], dl.min(), dr.min(),
                     int(np.argmin(dl)), int(np.argmin(dr)), dl[int(r["ball_owned_player"])] if r["ball_owned_team"] == 0 else np.nan,
                     dr[int(r["ball_owned_player"])] if r["ball_owned_team"] == 1 else np.nan))
A = np.array(rows, float); n = len(A)
owner, op, gm, z, dminL, dminR, argL, argR, dOwnL, dOwnR = A.T
normal = gm == 0
out = {"matches": 4, "steps": int(n)}
out["step_classes_fraction"] = {"ours/left owned": float(((owner == 0) & normal).mean()), "opp/right owned": float(((owner == 1) & normal).mean()),
                                "contested (nobody owns, normal play)": float(((owner == -1) & normal).mean()), "dead ball (game_mode != 0)": float((~normal).mean())}
out["dead_ball_modes"] = {int(m): int((gm == m).sum()) for m in np.unique(gm) if m != 0}
for name, team, dmin_own, dmin_opp, arg_own, dist_owner in (("left", 0, dminL, dminR, argL, dOwnL), ("right", 1, dminR, dminL, argR, dOwnR)):
    m = (owner == team) & normal
    d = dist_owner[m]
    out[f"{name}_owned_steps"] = {"n": int(m.sum()), "owner_player_distance_to_ball_median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                                  "owner_is_nearest_own_player": float((arg_own[m] == op[m]).mean()),
                                  "own_team_nearest_overall": float((dmin_own[m] <= dmin_opp[m]).mean()),
                                  "opponent_strictly_closer_than_0.02": float(((dmin_opp[m] + 0.02) < dmin_own[m]).mean())}
m = (owner == -1) & normal
nearest = np.minimum(dminL, dminR)
out["contested_steps"] = {"n": int(m.sum()), "nearest_player_distance_median": float(np.median(nearest[m])), "p10": float(np.percentile(nearest[m], 10)),
                          "ball_airborne_z_gt_0.15": float((z[m] > 0.15).mean()),
                          "someone_within_owner_range_(<=owned p95)": float((nearest[m] <= max(out["left_owned_steps"]["p95"], out["right_owned_steps"]["p95"])).mean())}
# flicker: lengths of runs of constant owner label among normal-play steps
runs = []; cur = None; length = 0
for o, nm in zip(owner, normal):
    key = int(o) if nm else 9
    if key == cur: length += 1
    else:
        if cur in (0, 1): runs.append(length)
        cur, length = key, 1
runs = np.array(runs)
out["ownership_spells"] = {"count": int(len(runs)), "median_length": float(np.median(runs)), "share_shorter_than_5_steps": float((runs < 5).mean()),
                           "share_of_owned_time_in_spells_shorter_than_5": float(runs[runs < 5].sum() / runs.sum())}
# possession share per match on this data, native vs native
out["possession_share_controlled_left"] = float(((owner == 0) & normal).sum() / (((owner == 0) | (owner == 1)) & normal).sum())
json.dump(out, open("results/possession_verification.json", "w"), indent=1)
print(json.dumps(out, indent=1))
