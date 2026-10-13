"""Reproduce logged training iterations in a fresh process to verify seed handling, resets, score tracking and
log aggregation: iteration 1 (Q = 0) and iteration 11 (Q = snapshot after iteration 10).
usage: bash run.sh diag_reproduce_iterations.py [n_workers]"""
import json
import sys
import numpy as np
from rollout import Runner

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    L = {json.loads(l)["iter"]: json.loads(l) for l in open("logs/q_main.jsonl")}
    r = Runner(workers)
    out = {}
    for it, qpath in ((1, None), (11, "checkpoints/q_main_it10.npz")):
        q = np.zeros((243, 3)) if qpath is None else np.load(qpath)["q"]
        eps = 0.5 + (0.05 - 0.5) * min(1.0, (it - 1) / (0.7 * 100))
        seeds = [100000 + (it - 1) * 12 + i for i in range(12)]
        res = [e for e, _ in r.run([dict(policy="q", q=q, epsilon=eps, seed=s, collect=True) for s in seeds])]
        gd = [e["goal_diff"] for e in res]
        ret = float(np.mean([e["return"] for e in res]))
        out[it] = {"goal_diff_per_match": gd, "sum": int(sum(gd)), "return_mean": ret, "logged_return_mean": L[it]["train_return_mean"],
                   "logged_sum": round(L[it]["train_goal_diff_mean"] * 12), "eps_used": eps, "logged_eps": L[it]["epsilon"]}
        print(f"iteration {it}: reproduced sum {sum(gd)} vs logged {out[it]['logged_sum']} | return {ret:.9f} vs logged {L[it]['train_return_mean']:.9f} | eps {eps:.4f} vs {L[it]['epsilon']}", flush=True)
        print("   per-match goal diff", gd, flush=True)
    json.dump(out, open("results/diag_reproduce_iterations.json", "w"), indent=1)
    r.close()
