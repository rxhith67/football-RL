"""Held-out evaluation of all coaches on identical seeds (paired).
usage: bash run.sh evaluate.py --episodes 120 --workers 12 --q checkpoints/q_main.npz --dqn checkpoints/dqn_main.pt --out results/eval_main.json
Seeds 900000.. are used for nothing else. Both kick-off parities occur equally (seed parity)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rollout import Runner
from train_dqn import np_weights, make_net

TEST_SEED0 = 900000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--q", default="checkpoints/q_main.npz")
    ap.add_argument("--dqn", default=None)
    ap.add_argument("--seed0", type=int, default=TEST_SEED0)
    ap.add_argument("--out", default="results/eval_main.json")
    args = ap.parse_args()
    seeds = [args.seed0 + i for i in range(args.episodes)]
    runner = Runner(args.workers)
    allrows, pools = {}, {}
    def run(label, **kw):
        res = runner.run([dict(seed=s, label=label, **kw) for s in seeds])
        allrows[label] = [e for e, _ in res]
        pools[label] = [t[1] for _, tr in res for t in tr]          # features seen (for shuffle control)
        print(label, "goal_diff", np.mean([e["goal_diff"] for e in allrows[label]]), flush=True)
    run("native_unmodified", policy="native", collect=True)
    run("fixed_default_clear", policy="fixed", action=1)
    run("fixed_compact", policy="fixed", action=0)
    run("fixed_stretched", policy="fixed", action=2)
    run("hand_coach", policy="hand")
    q = np.load(args.q)["q"]
    run("q_learning", policy="q", q=q, epsilon=0.0, collect=True)
    if args.dqn:
        net = make_net(); net.load_state_dict(torch.load(args.dqn)["net"])
        w = np_weights(net)
        run("dqn", policy="dqn", weights=w, epsilon=0.0, collect=True)
    # policy-dependence control: the trained policy sees states drawn at random from its own evaluation visits
    run("q_learning_shuffled_obs", policy="q", q=q, epsilon=0.0, shuffle_pool=pools["q_learning"])
    if args.dqn:
        run("dqn_shuffled_obs", policy="dqn", weights=w, epsilon=0.0, shuffle_pool=pools["dqn"])
    runner.close()
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"seeds": seeds, "rows": allrows}, open(args.out, "w"))
    print("saved", args.out)


if __name__ == "__main__":
    main()
