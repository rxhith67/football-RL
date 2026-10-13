"""Frozen GRF Q coach versus unmodified native AI, both sides per seed.

Run: bash run.sh head_to_head_eval.py --seeds 100 --seed-start 960000 --workers 6
No seeds are selected based on results. Demonstration matches use a separate script
and seed range. Each seed pair runs in one engine process; leg order alternates.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from coach_env import GRFDepthCoachEnv
from head_to_head_common import CHECKPOINT, run_match


FROZEN_SHA256 = "c54f3c63eae23dc8ed1d2ad79ecd5ddb5e1486917ff4ca858a43729918a34f17"
_ENV = None
_Q = None


def init_worker():
    global _ENV, _Q
    _ENV = GRFDepthCoachEnv(decision_steps=100)
    _Q = np.load(CHECKPOINT)["q"]


def run_pair(seed):
    order = (0, 1) if seed % 2 == 0 else (1, 0)
    legs = [run_match(_ENV, side, seed, _Q) for side in order]
    assert {r["coached_side"] for r in legs} == {"left", "right"}
    return {"seed": int(seed), "leg_order": [r["coached_side"] for r in legs], "legs": legs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=960000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/head_to_head_eval"))
    args = ap.parse_args()
    assert args.seeds > 0 and args.workers > 0
    sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert sha == FROZEN_SHA256, "frozen iteration-100 checkpoint hash changed"
    assert int(np.load(CHECKPOINT)["iter"]) == 100
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"label": "head-to-head frozen Q versus native, no policy tuning",
                "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha,
                "scenario": "11_vs_11_hard_stochastic", "seed_start": args.seed_start,
                "n_seed_pairs": args.seeds, "n_full_matches": 2 * args.seeds,
                "decision_interval": 100, "full_match_engine_steps": 3001,
                "side_control": "same seed both coached-left and coached-right; leg order alternates by seed parity",
                "player_control": "11 left builtin_ai actions per step, 0 right external slots; opposing team no tactical override after reset",
                "analysis_unit": "seed pair; cross-process variation remains possible"}
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "existing protocol differs"
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    pending = [s for s in seeds if not (args.out / ("seed_%d.json" % s)).exists()]
    print("seed pairs", len(seeds), "pending", len(pending), "workers", args.workers, flush=True)
    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=init_worker) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_pair, pending, chunksize=1), 1):
                file = args.out / ("seed_%d.json" % row["seed"])
                temp = file.with_suffix(".json.tmp")
                temp.write_text(json.dumps(row) + "\n")
                temp.replace(file)
                if idx % 10 == 0 or idx == len(pending):
                    print("finished", idx, "of", len(pending), "pending pairs", flush=True)
    pairs = [json.loads((args.out / ("seed_%d.json" % s)).read_text()) for s in seeds]
    assert len(pairs) == args.seeds and all(len(p["legs"]) == 2 for p in pairs)
    legs = [leg for pair in pairs for leg in pair["legs"]]
    print("complete", len(legs), "matches",
          "W/D/L", [sum(r["result"] == x for r in legs) for x in ("win", "draw", "loss")],
          "mean RL goal difference", float(np.mean([r["rl_goal_difference"] for r in legs])), flush=True)


if __name__ == "__main__":
    main()
