"""No-simulation checks for the pilot plan: checkpoints, policies, arithmetic, seed ranges.  usage: bash run.sh verify_pilot_setup.py"""
import hashlib
import json
from pathlib import Path

import numpy as np

FROZEN_A = "c54f3c63eae23dc8ed1d2ad79ecd5ddb5e1486917ff4ca858a43729918a34f17"
out = {}
# 1. every policy that will be evaluated can be loaded as a (243, 3) finite table
A = Path("../grf_rl_coach/checkpoints/q_it100_final.npz")
out["A_sha256_matches_frozen"] = hashlib.sha256(A.read_bytes()).hexdigest() == FROZEN_A
za = np.load(A); out["A"] = {"shape": list(za["q"].shape), "finite": bool(np.isfinite(za["q"]).all()), "iter": int(za["iter"])}
zb = np.load("checkpoints/q_possession_run1.npz")
out["B_current"] = {"shape": list(zb["q"].shape), "finite": bool(np.isfinite(zb["q"]).all()), "iter": int(zb["iter"]), "episodes": int(zb["episodes"]),
                    "q_is_plain_q_learning_table": bool(np.array_equal(zb["q"], zb["q_learn"])), "has_rng_state": "rng_state" in zb.files}
# the Double-Q checkpoint format was exercised by the smoke test: 'q' = mean of the two tables
zs = np.load("checkpoints/smoke_double_q_possession.npz")
out["C_format_from_smoke_test"] = {"shape": list(zs["q"].shape), "q_equals_mean_of_q1_q2": bool(np.allclose(zs["q"], (zs["q1"] + zs["q2"]) / 2)), "iter": int(zs["iter"])}
# 2. pilot arithmetic
per_iter = 12; stop_iter = 300 // per_iter
frac = lambda it: min(1.0, it / (0.7 * 100)); eps = lambda it: 0.5 + (0.05 - 0.5) * frac(it)
out["plan"] = {"matches_per_iteration": per_iter, "stop_iteration_for_300_matches": stop_iter, "stop_exact": stop_iter * per_iter == 300,
               "B_iterations_remaining": stop_iter - int(zb["iter"]), "B_matches_remaining": (stop_iter - int(zb["iter"])) * per_iter,
               "B_next_match_seeds": [2_010_000 + int(zb["episodes"]), 2_010_000 + 299], "C_match_seeds": [2_010_000, 2_010_000 + 299],
               "epsilon_at_start_of_iteration_20": round(eps(19), 4), "epsilon_at_last_iteration_25": round(eps(24), 4),
               "note": "exploration schedule follows --iters 100 for both B and C, so the pilot is an exact prefix of the full plan"}
# 3. evaluation seeds do not overlap any seed range used earlier
used = {"original training 100000+": (100000, 101200), "curves 500000+": (500000, 500006), "dev comparison 700000+": (700000, 700030), "demo 800001": (800001, 800001),
        "calibration 810000+": (810000, 810024), "final test": (900000, 900200), "head-to-head (other session)": (960000, 960100), "demo 970001": (970001, 970001),
        "secondary eval": (1000000, 1000150), "possession training run1/run2": (2010000, 2030000), "smoke evals": (3000000, 3000003), "smoke h2h": (4000000, 4000002),
        "tests": (777001, 777110)}
pilot = (3_100_000, 3_100_030)
out["pilot_eval_seeds"] = {"range": [pilot[0], pilot[1] - 1], "overlaps": [k for k, (a, b) in used.items() if not (pilot[1] <= a or b <= pilot[0])]}
print(json.dumps(out, indent=1))
Path("results").mkdir(exist_ok=True); json.dump(out, open("results/pilot_setup_check.json", "w"), indent=1)
