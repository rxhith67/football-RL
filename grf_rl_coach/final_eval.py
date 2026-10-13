"""FINAL held-out evaluation. Seeds 900000.. were never used for training (100000+), curves (500000+),
preliminary comparison (700000+) or the demo (800001). No policy is modified or tuned here.
Each policy is cached after it finishes; rerun the same command to resume.
usage: bash run.sh final_eval.py --episodes 200 --workers 6"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rollout import Runner

OUT = Path("results/final_eval.json")
POOL = Path("results/final_eval_q_pool.npy")
CKPT = "checkpoints/q_it100_final.npz"
SEED0 = 900000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    seeds = [SEED0 + i for i in range(args.episodes)]
    store = json.load(open(OUT)) if OUT.exists() else {"seeds": seeds, "rows": {}, "wall_seconds": {}, "meta": {}}
    assert store["seeds"] == seeds, "cached results use a different seed set"
    store["meta"] = {"q_checkpoint": CKPT, "q_sha256": hashlib.sha256(Path(CKPT).read_bytes()).hexdigest(),
                     "decision_steps": 100, "workers": args.workers, "shaping": 0.1, "gamma": 0.99}
    q = np.load(CKPT)["q"]
    assert int(np.load(CKPT)["iter"]) == 100
    runner = Runner(args.workers)

    def run(label, **kw):
        if label in store["rows"]:
            print("cached", label, flush=True); return
        t0 = time.time()
        res = runner.run([dict(seed=s, label=label, **kw) for s in seeds])
        store["rows"][label] = [e for e, _ in res]
        store["wall_seconds"][label] = time.time() - t0
        if label == "q_learning":
            np.save(POOL, np.array([t[1] for _, tr in res for t in tr], np.float32))
        gd = np.array([e["goal_diff"] for e in store["rows"][label]])
        print(f"{label:26s} n={len(gd)} goal_diff {gd.mean():+.3f} (SD {gd.std(ddof=1):.2f})  decisions/match {sorted({e['decisions'] for e in store['rows'][label]})}  {time.time() - t0:.0f}s", flush=True)
        json.dump(store, open(OUT, "w"))

    run("native_unmodified", policy="native")                      # never calls the setter: the true unmodified engine
    run("fixed_compact", policy="fixed", action=0)
    run("fixed_stretched", policy="fixed", action=2)
    run("hand_coach", policy="hand")
    run("random_switching", policy="random")
    run("q_learning", policy="q", q=q, epsilon=0.0, collect=True)
    pool = list(np.load(POOL))
    run("q_learning_shuffled_obs", policy="q", q=q, epsilon=0.0, shuffle_pool=pool)
    runner.close()
    print("all policies complete", flush=True)


if __name__ == "__main__":
    main()
