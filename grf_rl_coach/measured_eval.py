"""Run matches with the per-step measurement layer and save the raw streams (results/<out>/<label>.npz + .json).
Each policy is cached after it finishes. usage:
  bash run.sh measured_eval.py --out calib --seed0 810000 --episodes 16 --policies native
  bash run.sh measured_eval.py --out secondary --seed0 1000000 --episodes 150 --policies native,q,compact,stretched,hand,random,q_shuffled"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from measure import run_measured
from rollout import Runner

CKPT = "checkpoints/q_it100_final.npz"
SPECS = {"native": dict(policy="native"), "compact": dict(policy="fixed", action=0), "stretched": dict(policy="fixed", action=2),
         "hand": dict(policy="hand"), "random": dict(policy="random")}
LABEL = {"native": "native_unmodified", "compact": "fixed_compact", "stretched": "fixed_stretched", "hand": "hand_coach",
         "random": "random_switching", "q": "q_learning", "q_shuffled": "q_learning_shuffled_obs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path("results") / args.out; out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed0 + i for i in range(args.episodes)]
    q = np.load(CKPT)["q"]
    runner = Runner(args.workers)
    pool = None
    for name in args.policies.split(","):
        label = LABEL[name]
        f = out / f"{label}.pkl"
        if f.exists():
            print("cached", label, flush=True)
            if name == "q": pool = [d[:14] for ep, st, dec in pickle.load(open(f, "rb")) for d in dec]
            continue
        if name == "q": spec = dict(policy="q", q=q, epsilon=0.0)
        elif name == "q_shuffled":
            if pool is None:
                pool = [d[:14] for ep, st, dec in pickle.load(open(out / f"{LABEL['q']}.pkl", "rb")) for d in dec]
            spec = dict(policy="q", q=q, epsilon=0.0, shuffle_pool=[np.asarray(p, np.float32) for p in pool])
        else: spec = SPECS[name]
        t0 = time.time()
        res = runner.run_fn(run_measured, [dict(spec, seed=s, label=label) for s in seeds])
        pickle.dump(res, open(f, "wb"))
        if name == "q": pool = [d[:14] for ep, st, dec in res for d in dec]
        gd = np.mean([ep["goal_diff"] for ep, _, _ in res])
        print(f"{label:26s} n={len(res)} goal_diff {gd:+.3f}  {time.time() - t0:.0f}s", flush=True)
    runner.close()
    json.dump({"seeds": seeds, "labels": [LABEL[n] for n in args.policies.split(",")]}, open(out / "meta.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
