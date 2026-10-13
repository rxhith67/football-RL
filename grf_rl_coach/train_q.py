"""Tabular Q-learning for the GRF depth coach (sequential, 30 decisions per match).
usage: bash run.sh train_q.py --iters 100 --workers 6 --tag q_main
Each iteration: `workers` full matches in parallel with the current epsilon-greedy policy; then ordered
TD(0) updates over all collected transitions: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') * (1-done) - Q(s,a))."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from coach_env import N_DISCRETE
from rollout import Runner

DEV_SEEDS = [500000 + i for i in range(6)]       # learning-curve evaluation seeds (never used for training/test)
TRAIN_SEED0 = 100000


def td_update(q, visits, trans, alpha, gamma):
    """Ordered one-step Q-learning updates over one episode's coach-decision transitions.
    Each transition is ONE coach decision (an interval of `decision_steps` engine steps), so gamma is applied
    once per decision, not per engine frame. A terminal transition has no bootstrap term."""
    for (s, f, a, r, s2, f2, done) in trans:
        target = r if done else r + gamma * q[s2].max()
        q[s, a] += alpha * (target - q[s, a])
        visits[s, a] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes-per-iter", type=int, default=None, help="matches per update batch (default = workers); the planned design uses 12")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps-start", type=float, default=0.5)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-frac", type=float, default=0.7, help="fraction of iterations over which epsilon decays")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--shaping", type=float, default=0.1)
    ap.add_argument("--decision-steps", type=int, default=100)
    ap.add_argument("--tag", default="q_main")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = Path("checkpoints"); out.mkdir(exist_ok=True); Path("logs").mkdir(exist_ok=True)
    ckpt, logf = out / f"{args.tag}.npz", Path("logs") / f"{args.tag}.jsonl"
    q = np.zeros((N_DISCRETE, 3)); visits = np.zeros((N_DISCRETE, 3), int)
    it0, episodes, decisions, engine_steps, wall = 0, 0, 0, 0, 0.0
    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume given but {ckpt} does not exist")
        z = np.load(ckpt)
        q, visits = z["q"], z["visits"]
        it0, episodes, decisions, engine_steps = (int(z[k]) for k in ("iter", "episodes", "decisions", "engine_steps"))
        wall = float(z["wall"])
        print(f"resumed {ckpt}: iter {it0}, episodes {episodes}, decisions {decisions}", flush=True)
    elif ckpt.exists() or logf.exists():
        # never overwrite earlier work silently (the old code deleted the log here)
        raise SystemExit(f"{ckpt} or {logf} already exists: use --resume, or choose a new --tag")
    runner = Runner(args.workers, args.decision_steps, args.shaping, args.gamma)
    t_start = time.time()
    try:
        for it in range(it0, args.iters):
            frac = min(1.0, it / max(1, args.eps_frac * args.iters))
            eps = args.eps_start + (args.eps_end - args.eps_start) * frac
            specs = [dict(policy="q", q=q.copy(), epsilon=eps, seed=TRAIN_SEED0 + episodes + i, collect=True, label="train")
                     for i in range(args.episodes_per_iter or args.workers)]
            res = runner.run(specs)
            for ep, trans in res:                                    # ordered TD(0) updates
                td_update(q, visits, trans, args.alpha, args.gamma)
                decisions += ep["decisions"]; engine_steps += ep["engine_steps"]
            episodes += len(res)
            rec = {"iter": it + 1, "episodes": episodes, "epsilon": round(eps, 4),
                   "train_return_mean": float(np.mean([e["return"] for e, _ in res])),
                   "train_goal_diff_mean": float(np.mean([e["goal_diff"] for e, _ in res])),
                   "train_action_frac": (np.sum([e["action_counts"] for e, _ in res], 0) / max(1, sum(e["decisions"] for e, _ in res))).round(3).tolist(),
                   "states_visited": int((visits.sum(1) > 0).sum()), "decisions": decisions, "engine_steps": engine_steps,
                   "wall_s": wall + time.time() - t_start}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                ev = runner.run([dict(policy="q", q=q.copy(), epsilon=0.0, seed=s, label="dev") for s in DEV_SEEDS])
                rec["dev_goal_diff_mean"] = float(np.mean([e["goal_diff"] for e, _ in ev]))
                rec["dev_return_mean"] = float(np.mean([e["return"] for e, _ in ev]))
                rec["dev_action_frac"] = (np.sum([e["action_counts"] for e, _ in ev], 0) / max(1, sum(e["decisions"] for e, _ in ev))).round(3).tolist()
                np.savez(out / f"{args.tag}_it{it + 1}.npz", q=q, visits=visits)
            np.savez(ckpt, q=q, visits=visits, iter=it + 1, episodes=episodes, decisions=decisions,
                     engine_steps=engine_steps, wall=wall + time.time() - t_start)
            with logf.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
