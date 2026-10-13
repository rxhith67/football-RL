"""Train the tabular coach with Q-learning or Double Q-learning under the original or the possession reward.
Same state space (243), actions (compact/native default/stretched), decision interval (100 engine steps), exploration schedule,
alpha, gamma and interaction budget (12 matches per update batch, 100 batches = 1,200 matches) for every configuration.
usage: bash run.sh train_poss.py --algo double_q --reward possession --run-id 1 --workers 6
Match seeds: 2,000,000 + run_id * 10,000 + match index (never reused by evaluation); learner RNG: run_id."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from learners import q_update, double_q_update, behaviour_table
from poss_rollout import Runner

N_STATES = 243
DEV_SEEDS = [500000 + i for i in range(6)]          # greedy-policy curve seeds (same as the original run); never final-test seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["q", "double_q"], required=True)
    ap.add_argument("--reward", choices=["original", "possession"], required=True)
    ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes-per-iter", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps-start", type=float, default=0.5)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-frac", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None, help="stop cleanly after this many iterations (iterations x episodes-per-iter = matches); the exploration schedule still follows --iters, so a stopped run equals a prefix of the full run")
    ap.add_argument("--dev-seeds", type=int, default=6, help="number of greedy dev-evaluation matches at each evaluation point")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to sleep after each iteration (limits sustained heat)")
    args = ap.parse_args()
    tag = args.tag or f"{args.algo}_{args.reward}_run{args.run_id}"
    out = Path("checkpoints"); out.mkdir(exist_ok=True); Path("logs").mkdir(exist_ok=True)
    ckpt, logf = out / f"{tag}.npz", Path("logs") / f"{tag}.jsonl"
    seed0 = 2_000_000 + args.run_id * 10_000
    rng = np.random.default_rng(args.run_id)
    q = np.zeros((N_STATES, 3)); q1 = np.zeros((N_STATES, 3)); q2 = np.zeros((N_STATES, 3)); visits = np.zeros((N_STATES, 3), int)
    it0, episodes, decisions, engine_steps, wall = 0, 0, 0, 0, 0.0
    upd1 = upd2 = 0                                   # Double Q-learning: how often each table was the one updated
    if args.resume:
        if not ckpt.exists(): raise SystemExit(f"--resume but {ckpt} does not exist")
        z = np.load(ckpt)
        q, q1, q2, visits = z["q_learn"], z["q1"], z["q2"], z["visits"]
        it0, episodes, decisions, engine_steps = (int(z[k]) for k in ("iter", "episodes", "decisions", "engine_steps"))
        upd1 = int(z["upd1"]) if "upd1" in z.files else 0; upd2 = int(z["upd2"]) if "upd2" in z.files else 0
        wall = float(z["wall"]); rng.bit_generator.state = json.loads(str(z["rng_state"]))
        print(f"resumed {ckpt}: iter {it0}", flush=True)
    elif ckpt.exists() or logf.exists():
        raise SystemExit(f"{ckpt} or {logf} exists: use --resume or a new --tag/--run-id")
    stop_it = min(args.iters, args.stop_after or args.iters)
    if it0 >= stop_it:
        print(f"already at iteration {it0} (>= stop {stop_it}); nothing to do", flush=True); return
    runner = Runner(args.workers)
    t_start = time.time()

    def save_npz(path, **kw):
        """atomic: write a temp file, then rename, so an interruption can never leave a half-written checkpoint"""
        tmp = path.with_name(path.stem + ".tmp.npz"); np.savez(tmp, **kw); os.replace(tmp, path)

    def table():
        return behaviour_table(args.algo, q=q, q1=q1, q2=q2)

    try:
        for it in range(it0, stop_it):
            frac = min(1.0, it / max(1, args.eps_frac * args.iters))
            eps = args.eps_start + (args.eps_end - args.eps_start) * frac
            tb = table().copy()
            res = runner.run([dict(policy="q", q=tb, epsilon=eps, seed=seed0 + episodes + i, side=0, reward_mode=args.reward,
                                   collect=True, label="train") for i in range(args.episodes_per_iter)])
            for ep, trans, _ in res:
                if args.algo == "q": q_update(q, visits, trans, args.alpha, args.gamma)
                else:
                    chosen = double_q_update(q1, q2, visits, trans, args.alpha, args.gamma, rng)
                    upd1 += chosen.count(0); upd2 += chosen.count(1)
                decisions += ep["decisions"]; engine_steps += ep["engine_steps"]
            episodes += len(res)
            eps_list = [e for e, _, _ in res]
            rec = {"iter": it + 1, "episodes": episodes, "epsilon": round(eps, 4), "algo": args.algo, "reward": args.reward,
                   "train_return_trained_on": float(np.mean([e["return_trained_on"] for e in eps_list])),
                   "train_return_original": float(np.mean([e["return_original"] for e in eps_list])),
                   "train_return_possession": float(np.mean([e["return_possession"] for e in eps_list])),
                   "train_possession_share": float(np.mean([e["possession_share"] for e in eps_list])),
                   "train_goal_diff": float(np.mean([e["goal_diff"] for e in eps_list])),
                   "train_action_frac": (np.sum([e["action_counts"] for e in eps_list], 0) / sum(e["decisions"] for e in eps_list)).round(3).tolist(),
                   "states_visited": int((visits.sum(1) > 0).sum()), "double_q_table_updates_q1_q2": [upd1, upd2], "decisions": decisions, "engine_steps": engine_steps,
                   "wall_s": wall + time.time() - t_start}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters or it + 1 == stop_it:
                tb = table().copy()
                ev = [e for e, _, _ in runner.run([dict(policy="q", q=tb, epsilon=0.0, seed=s, side=0, reward_mode=args.reward, label="dev") for s in DEV_SEEDS[:args.dev_seeds]])]
                rec["dev_goal_diff"] = float(np.mean([e["goal_diff"] for e in ev]))
                rec["dev_possession_share"] = float(np.mean([e["possession_share"] for e in ev]))
                rec["dev_action_frac"] = (np.sum([e["action_counts"] for e in ev], 0) / sum(e["decisions"] for e in ev)).round(3).tolist()
                save_npz(out / f"{tag}_it{it + 1}.npz", q=table(), q_learn=q, q1=q1, q2=q2, visits=visits)
            save_npz(ckpt, q=table(), q_learn=q, q1=q1, q2=q2, visits=visits, iter=it + 1, episodes=episodes, decisions=decisions,
                     engine_steps=engine_steps, wall=wall + time.time() - t_start, algo=args.algo, reward=args.reward,
                     run_id=args.run_id, seed0=seed0, upd1=upd1, upd2=upd2, rng_state=json.dumps(rng.bit_generator.state))
            with logf.open("a") as fh: fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if args.cooldown > 0 and it + 1 < stop_it: time.sleep(args.cooldown)
        print(f"STOPPED at iteration {stop_it} = {episodes} training matches, {decisions} decisions", flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
