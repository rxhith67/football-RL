"""Evaluation of all coaches on predetermined, unseen seeds. Raw per-match results and per-step streams are cached per policy.
  independent : each policy plays the native AI on the LEFT only (seeds 3,000,000+; same protocol as the earlier final evaluation)
  headtohead  : each policy plays the native AI on BOTH sides for every seed (seeds 4,000,000+); leg order irrelevant (deterministic)
usage:
  bash run.sh evaluate_poss.py --mode independent --episodes 150 --out eval_independent \
      --policy A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz --policy B_q_possession=checkpoints/q_possession_run1.npz ...
  bash run.sh evaluate_poss.py --mode headtohead --episodes 100 --out eval_h2h --policy ... --policy native
Built-in labels: native, fixed_compact, fixed_stretched, random_switching, hand_coach."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from poss_rollout import Runner

BUILTIN = {"native": dict(policy="native"), "fixed_compact": dict(policy="fixed", action=0), "fixed_stretched": dict(policy="fixed", action=2),
           "random_switching": dict(policy="random"), "hand_coach": dict(policy="hand")}
SEED0 = {"independent": 3_000_000, "headtohead": 4_000_000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["independent", "headtohead"], required=True)
    ap.add_argument("--episodes", type=int, required=True, help="seeds (independent) or seed pairs (headtohead)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", action="append", default=[], help="label=checkpoint.npz or a built-in label")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=None, help="first evaluation seed (default depends on mode)")
    ap.add_argument("--chunk", type=int, default=None, help="matches per batch; with --pause the machine rests between batches")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to rest between batches")
    args = ap.parse_args()
    out = Path("results") / args.out; out.mkdir(parents=True, exist_ok=True)
    seeds = [(args.seed0 if args.seed0 is not None else SEED0[args.mode]) + i for i in range(args.episodes)]
    sides = (0,) if args.mode == "independent" else (0, 1)
    meta = {"mode": args.mode, "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "sides": list(sides), "decision_steps": 100,
            "policies": {}}
    runner = Runner(args.workers)
    try:
        for item in args.policy:
            label, _, path = item.partition("=")
            f = out / f"{label}.pkl"
            if label in BUILTIN: spec = dict(BUILTIN[label])
            else: spec = dict(policy="q", q=np.load(path)["q"], epsilon=0.0)
            meta["policies"][label] = {"checkpoint": path or None, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None}
            if f.exists():
                print("cached", label, flush=True); continue
            t0 = time.time()
            specs = [dict(spec, seed=s, side=sd, reward_mode="possession", label=label, keep_stream=True) for s in seeds for sd in sides]
            res = []
            chunk = args.chunk or len(specs)
            for i in range(0, len(specs), chunk):
                res += runner.run(specs[i:i + chunk])
                if args.pause > 0 and i + chunk < len(specs): time.sleep(args.pause)
            if args.pause > 0: time.sleep(args.pause)
            pickle.dump([(ep, stream) for ep, _, stream in res], open(f, "wb"))
            gd = np.mean([ep["goal_diff"] for ep, _, _ in res]); ps = np.mean([ep["possession_share"] for ep, _, _ in res])
            print(f"{label:28s} n={len(res)} goal_diff {gd:+.3f} possession {ps:.3f}  {time.time() - t0:.0f}s", flush=True)
    finally:
        runner.close()
    old = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else {"policies": {}}
    meta["policies"] = {**old.get("policies", {}), **meta["policies"]}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
