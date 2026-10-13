"""Diagnostic for the near-constant goal-difference sums in the first 9 training iterations.
Runs uniform-random tactical switching and the native AI on the SAME seeds those iterations used
(100000..100107, blocks of 12) and reports block sums, parity effects and goal timing.
usage: bash run.sh diag_first_iterations.py [n_workers]"""
import json
import sys
import numpy as np
from rollout import Runner

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seeds = [100000 + i for i in range(108)]
    r = Runner(workers)
    out = {}
    for name, kw in (("random_switching", dict(policy="random")), ("native", dict(policy="native"))):
        res = [e for e, _ in r.run([dict(seed=s, **kw) for s in seeds])]
        gd = np.array([e["goal_diff"] for e in res])
        out[name] = {"seeds": seeds, "goal_diff": gd.tolist(), "goal_events": [e["goal_events"] for e in res],
                     "goals_for": [e["goals_for"] for e in res], "goals_against": [e["goals_against"] for e in res]}
        blocks = gd.reshape(9, 12).sum(1)
        par = np.array(seeds) % 2
        print(name, "block sums", blocks.tolist(), "mean/match", round(gd.mean(), 3),
              "| even-seed mean", round(gd[par == 0].mean(), 3), "odd-seed mean", round(gd[par == 1].mean(), 3),
              "| SD per match", round(gd.std(ddof=1), 2), flush=True)
        first = [ev[0][0] if ev else None for ev in out[name]["goal_events"]]
        print("   first-goal decision index (None = no goal):", first[:24], flush=True)
    json.dump(out, open("results/diag_first_iterations.json", "w"))
    r.close()
