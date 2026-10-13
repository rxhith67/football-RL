"""Whole-match sanity check that the native AI keeps playing normally under each depth setting.
usage: python full_match_check.py DEPTH OUT.json   (DEPTH -1 = default). All 11 left players on builtin AI."""
import json, sys, time
import numpy as np
from poc_lib import AI, make_env, set_depth, live_depth, row, line_metrics, has_setter
depth = float(sys.argv[1]); env = make_env(); env.reset(); assert has_setter(env)
set_depth(env, 0, depth); set_depth(env, 1, -1.0)
last_left_owner = None; prev = None; passes = gains = losses = 0; poss = {0: 0, 1: 0, -1: 0}; steps = 0; t0 = time.perf_counter()
back = []; length = []; lives = []; moved = np.zeros(11); last = None
done = False
while not done:
    obs, _, done, _ = env.step([AI] * 11); steps += 1
    s = row(obs); poss[s["owner"]] += 1
    if prev is not None:
        if s["owner"] == 0:
            if last_left_owner is not None and s["owner_player"] != last_left_owner: passes += 1
            last_left_owner = s["owner_player"]
        if s["owner"] == 1: last_left_owner = None
        if s["owner"] == 0 and prev["owner"] == 1: gains += 1
        if s["owner"] == 1 and prev["owner"] == 0: losses += 1
    if s["owner"] == 1:
        m = line_metrics(s["left"]); back.append(m["back_line_x"]); length.append(m["team_length"])
    if last is not None: moved += np.linalg.norm(s["left"] - last, axis=1)
    last = s["left"]; prev = s
    if steps % 100 == 0: lives.append(live_depth(env, 0))
    assert steps < 20000
res = {"depth": depth, "steps": steps, "final_score": s["score"], "possession_steps": {str(k): v for k, v in poss.items()},
       "left_pass_proxy": passes, "possession_gains": gains, "possession_losses": losses,
       "mean_back_line_x_opp_possession": float(np.mean(back)), "mean_team_length_opp_possession": float(np.mean(length)),
       "total_distance_moved_by_player": [round(float(x), 2) for x in moved],
       "live_depth_every_100_steps_unique": sorted(set(round(v, 4) for v in lives)), "seconds": time.perf_counter() - t0}
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res))
