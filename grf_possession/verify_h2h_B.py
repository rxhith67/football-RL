"""Readiness check for the Model-B head-to-head pilot (a few hundred engine steps; NOT the experiment).
Checks: checkpoint identity; both teams' strength settings; B's depth changes reach ONLY its own team on either side; possession
accounting and ownership mapping on both sides; the head-to-head code path reproduces an earlier independent evaluation exactly.
usage: bash run.sh verify_h2h_B.py"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from policies import q_action
from poss_env import PossessionCoachEnv
import possession as P

CK = Path("checkpoints/q_possession_run1_it25.npz")
out = {"checkpoint": str(CK), "sha256": hashlib.sha256(CK.read_bytes()).hexdigest()}
z = np.load(CK); q = z["q"]
zm = np.load("checkpoints/q_possession_run1.npz")        # main checkpoint carries the counters; the snapshot stores tables only
out["checkpoint_facts"] = {"main_checkpoint_iteration": int(zm["iter"]), "training_matches": int(zm["episodes"]), "decisions": int(zm["decisions"]),
                           "snapshot_table_equals_main_checkpoint_table": bool(np.array_equal(q, zm["q"])), "table_shape": list(q.shape),
                           "finite": bool(np.isfinite(q).all()), "algo": str(zm["algo"]), "reward": str(zm["reward"])}
env = PossessionCoachEnv(side=0, reward_mode="possession")
EXPECT = {0: 0.0, 1: 0.71, 2: 1.0}

# 1. player-strength settings as the engine sees them
env.side = 0; env.reset(seed=3200000)
cfg = env.env.unwrapped._env._env.config
dl, dr = float(cfg.left_team_difficulty), float(cfg.right_team_difficulty)
out["team_difficulty"] = {"left": dl, "right": dr, "stat_multiplier_left": round(0.3 + 0.7 * dl, 4), "stat_multiplier_right": round(0.3 + 0.7 * dr, 4),
                          "extra_reaction_time_ms_left": round((1 - dl) * 100, 1), "extra_reaction_time_ms_right": round((1 - dr) * 100, 1),
                          "meaning": "right-team players have 3.5% lower skill statistics and 5 ms slower reactions; difficulty scales BOTH rosters' players regardless of who is coached"}

# 2. B's actions reach only its own team, on each side; possession accounting and mapping
checks = {}
for side in (0, 1):
    env.side = side
    f = env.reset(seed=3200000)
    rows, ok_depth, ok_opp, ok_acct, ok_owner = [], True, True, True, True
    for k in range(6):
        a = int(q_action(q, env.discretize(f), None, 0.0))            # greedy, no exploration, ties -> native default
        f, r, done, info = env.step(a)
        own_expected = EXPECT[a]
        ok_depth &= abs(info["live_depth"] - own_expected) < 1e-3
        ok_opp &= abs(info["opp_live_depth"] - 0.71) < 1e-3
        pp = info["poss"]; ok_acct &= (pp["ours"] + pp["opp"] + pp["contested"] + pp["dead"] == info["engine_steps"])
        rows.append({"decision": k + 1, "action": a, "own_live_depth": round(info["live_depth"], 3), "opp_live_depth": round(info["opp_live_depth"], 3), "share": pp["share"]})
        core = env.env.unwrapped._env.observation(); go = int(core["ball_owned_team"])
        last_owner = int(env.stream[-1][0]); ok_owner &= (last_owner == (go if side == 0 else ((1 - go) if go > -1 else -1)))
    st = np.array(env.stream)
    ours = int(((st[:, 0] == 0) & (st[:, 5] == 0)).sum()); opp = int(((st[:, 0] == 1) & (st[:, 5] == 0)).sum())
    checks[f"side{side}"] = {"own_depth_matches_B_action_every_decision": bool(ok_depth), "opponent_depth_stays_native_0.71": bool(ok_opp),
                             "possession_classes_sum_to_steps": bool(ok_acct), "ownership_label_matches_raw_engine_owner": bool(ok_owner),
                             "controlled_share_first_600_steps": ours / (ours + opp) if ours + opp else None, "decisions": rows}
out["both_sides"] = checks

# 3. same code path as the earlier independent evaluation: B on the left, seed 3100000 -> identical actions and interval shares
old = {(ep["seed"], ep["side"]): ep for ep, _ in pickle.load(open("results/pilot_eval/B_q_possession.pkl", "rb"))}[(3100000, 0)]
env.side = 0; f = env.reset(seed=3100000); acts, shares = [], []
for k in range(10):
    a = int(q_action(q, env.discretize(f), None, 0.0)); f, r, done, info = env.step(a); acts.append(a); shares.append(info["poss"]["share"])
out["reproduces_earlier_independent_run_first_10_decisions"] = {"actions_identical": acts == old["actions"][:10], "interval_possession_identical": shares == old["interval_shares"][:10]}
out["all_checks_passed"] = bool(all(all(v for k, v in c.items() if k not in ("decisions", "controlled_share_first_600_steps")) for c in checks.values())
                                and out["reproduces_earlier_independent_run_first_10_decisions"]["actions_identical"] and out["reproduces_earlier_independent_run_first_10_decisions"]["interval_possession_identical"])
Path("results").mkdir(exist_ok=True); json.dump(out, open("results/h2h_B_readiness_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "both_sides"}, indent=1))
for sd, c in checks.items(): print(sd, {k: v for k, v in c.items() if k != "decisions"}); print("   B actions/live depth:", [(r["action"], r["own_live_depth"], r["opp_live_depth"]) for r in c["decisions"]])
