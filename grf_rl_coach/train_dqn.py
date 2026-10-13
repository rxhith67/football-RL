"""DQN for the GRF depth coach: same environment, actions, reward, seeds and episode budget as train_q.py.
Network 14 -> 64 -> 64 -> 3 (ReLU), replay buffer, hard-updated target network, Huber loss, Adam.
State normalisation: the 14 features are bounded by construction (see coach_env.FEATURE_NAMES); no running
statistics, so training workers, learner and evaluation all see identical inputs.
usage: bash run.sh train_dqn.py --iters 100 --workers 12 --tag dqn_main"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from coach_env import OBS_DIM
from rollout import Runner
from train_q import DEV_SEEDS, TRAIN_SEED0

torch.set_num_threads(2)


def make_net():
    return nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))


def np_weights(net):
    out = []
    for m in net:
        if isinstance(m, nn.Linear):
            out += [m.weight.detach().numpy().T.copy().astype(np.float32), m.bias.detach().numpy().copy().astype(np.float32)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=50000)
    ap.add_argument("--updates-per-transition", type=float, default=1.0)
    ap.add_argument("--target-sync", type=int, default=200, help="gradient updates between hard target copies")
    ap.add_argument("--eps-start", type=float, default=0.5)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-frac", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--shaping", type=float, default=0.1)
    ap.add_argument("--decision-steps", type=int, default=100)
    ap.add_argument("--tag", default="dqn_main")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    out = Path("checkpoints"); out.mkdir(exist_ok=True); Path("logs").mkdir(exist_ok=True)
    logf = Path("logs") / f"{args.tag}.jsonl"
    if logf.exists(): logf.unlink()
    net, tgt = make_net(), make_net(); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = {k: [] for k in ("f", "a", "r", "f2", "d")}
    updates, episodes, decisions, engine_steps = 0, 0, 0, 0
    runner = Runner(args.workers, args.decision_steps, args.shaping, args.gamma)
    t_start = time.time()
    try:
        for it in range(args.iters):
            frac = min(1.0, it / max(1, args.eps_frac * args.iters))
            eps = args.eps_start + (args.eps_end - args.eps_start) * frac
            W = np_weights(net)
            res = runner.run([dict(policy="dqn", weights=W, epsilon=eps, seed=TRAIN_SEED0 + episodes + i, collect=True, label="train")
                              for i in range(args.workers)])
            n_new = 0
            for ep, trans in res:
                for (s, f, a, r, s2, f2, done) in trans:
                    for k, v in zip(("f", "a", "r", "f2", "d"), (f, a, r, f2, float(done))): buf[k].append(v)
                    n_new += 1
                decisions += ep["decisions"]; engine_steps += ep["engine_steps"]
            for k in buf: buf[k] = buf[k][-args.buffer:]
            episodes += len(res)
            losses = []
            n_up = int(n_new * args.updates_per_transition)
            N = len(buf["a"])
            F = torch.tensor(np.array(buf["f"]), dtype=torch.float32); A = torch.tensor(buf["a"]); R = torch.tensor(buf["r"], dtype=torch.float32)
            F2 = torch.tensor(np.array(buf["f2"]), dtype=torch.float32); D = torch.tensor(buf["d"], dtype=torch.float32)
            for _ in range(n_up):
                idx = torch.as_tensor(rng.integers(0, N, args.batch))
                with torch.no_grad():
                    y = R[idx] + args.gamma * (1 - D[idx]) * tgt(F2[idx]).max(1).values
                q = net(F[idx]).gather(1, A[idx][:, None])[:, 0]
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
                updates += 1; losses.append(float(loss))
                if updates % args.target_sync == 0: tgt.load_state_dict(net.state_dict())
            rec = {"iter": it + 1, "episodes": episodes, "epsilon": round(eps, 4), "updates": updates, "buffer": N,
                   "loss_mean": float(np.mean(losses)) if losses else None,
                   "train_return_mean": float(np.mean([e["return"] for e, _ in res])),
                   "train_goal_diff_mean": float(np.mean([e["goal_diff"] for e, _ in res])),
                   "train_action_frac": (np.sum([e["action_counts"] for e, _ in res], 0) / max(1, sum(e["decisions"] for e, _ in res))).round(3).tolist(),
                   "decisions": decisions, "engine_steps": engine_steps, "wall_s": time.time() - t_start}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                ev = runner.run([dict(policy="dqn", weights=np_weights(net), epsilon=0.0, seed=s, label="dev") for s in DEV_SEEDS])
                rec["dev_goal_diff_mean"] = float(np.mean([e["goal_diff"] for e, _ in ev]))
                rec["dev_return_mean"] = float(np.mean([e["return"] for e, _ in ev]))
                rec["dev_action_frac"] = (np.sum([e["action_counts"] for e, _ in ev], 0) / max(1, sum(e["decisions"] for e, _ in ev))).round(3).tolist()
                torch.save({"net": net.state_dict(), "iter": it + 1}, out / f"{args.tag}_it{it + 1}.pt")
            torch.save({"net": net.state_dict(), "iter": it + 1, "episodes": episodes, "updates": updates}, out / f"{args.tag}.pt")
            with logf.open("a") as fh: fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
