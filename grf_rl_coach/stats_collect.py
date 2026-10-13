"""EVALUATION-ONLY trace collector. Records raw GRF observations at every engine step while the *unchanged*
`rollout.run_episode` plays full matches, so football statistics can be computed afterwards (stats_metrics.py).

Nothing here changes the reward, tactical controls, trained checkpoint, policies or engine: the recorder is a read-only
wrapper around the gfootball env's step(); run_episode / coach_env / policies are imported as they are.

usage (WSL):  bash run.sh stats_collect.py --set dev   --episodes 30  --workers 8
              bash run.sh stats_collect.py --set final --episodes 200 --workers 8
              bash run.sh stats_collect.py --set verify          # replays seeds 900000-900003 and checks they equal results/final_eval.json
Results are cached per policy label (resumable): results/stats_<set>/<label>.pkl
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

import rollout

CKPT = "checkpoints/q_it100_final.npz"
SETS = {"dev": 1_100_000, "final": 1_000_000, "verify": 900_000}      # disjoint from train 100000+, curves 500000+, dev_compare 700000+, demo 800001, old test 900000+
LABELS = ("native_unmodified", "fixed_compact", "fixed_stretched", "hand_coach", "random_switching", "q_learning", "q_learning_shuffled_obs")

_TRACE = {}


def _fresh():
    return {k: [] for k in ("ball", "dir", "owner", "owned_player", "mode", "score", "steps_left", "near_team", "near_idx", "near_d")}


def _record(obs):
    r = obs[0]
    t = _TRACE
    t["ball"].append(np.asarray(r["ball"], np.float32)); t["dir"].append(np.asarray(r["ball_direction"], np.float32))
    t["owner"].append(int(r["ball_owned_team"])); t["owned_player"].append(int(r["ball_owned_player"]))
    t["mode"].append(int(r["game_mode"])); t["score"].append([int(x) for x in r["score"]]); t["steps_left"].append(int(r["steps_left"]))
    # nearest player to the ball in the xy plane (kicker attribution for balls struck from a loose state, e.g. volleys)
    pl = np.concatenate([np.asarray(r["left_team"], float), np.asarray(r["right_team"], float)])
    d = np.linalg.norm(pl - np.asarray(r["ball"], float)[:2], axis=1)
    j = int(np.argmin(d))
    t["near_team"].append(0 if j < 11 else 1); t["near_idx"].append(j % 11); t["near_d"].append(float(d[j]))


def _init(decision_steps=100, shaping=0.1, gamma=0.99):
    rollout._init(decision_steps, shaping, gamma)
    env = rollout._ENV.env
    orig = env.step

    def step(action):                                    # read-only tap: same call, same return value
        res = orig(action)
        _record(res[0])
        return res
    env.step = step


def _run(spec):
    global _TRACE
    _TRACE = _fresh()
    out, trans = rollout.run_episode(spec)
    tr = {"ball": np.array(_TRACE["ball"]), "dir": np.array(_TRACE["dir"]), "owner": np.array(_TRACE["owner"], np.int8),
          "owned_player": np.array(_TRACE["owned_player"], np.int8), "mode": np.array(_TRACE["mode"], np.int8),
          "score": np.array(_TRACE["score"], np.int8), "steps_left": np.array(_TRACE["steps_left"], np.int16),
          "near_team": np.array(_TRACE["near_team"], np.int8), "near_idx": np.array(_TRACE["near_idx"], np.int8), "near_d": np.array(_TRACE["near_d"], np.float32),
          "dec_state": np.array([t[0] for t in trans], np.int16), "dec_action": np.array([t[2] for t in trans], np.int8),
          "feats": np.array([t[1] for t in trans], np.float32)}
    return out, tr


class Runner:
    def __init__(self, workers):
        import multiprocessing as mp
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init)

    def run(self, specs):
        return self.pool.map(_run, specs, chunksize=1)

    def close(self):
        self.pool.close(); self.pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()
    n = args.episodes or {"dev": 30, "final": 200, "verify": 4}[args.set]
    seeds = [SETS[args.set] + i for i in range(n)]
    outdir = Path("results") / f"stats_{args.set}"; outdir.mkdir(parents=True, exist_ok=True)
    qz = np.load(CKPT); assert int(qz["iter"]) == 100
    q = qz["q"]
    meta = {"q_checkpoint": CKPT, "q_sha256": hashlib.sha256(Path(CKPT).read_bytes()).hexdigest(), "set": args.set, "seeds": [seeds[0], seeds[-1]], "n": n}
    if args.set == "final":                               # pre-registration guard: the analysis spec must exist (and is hashed) BEFORE any final match is played
        spec = Path("eval_spec.json")
        assert spec.exists(), "write and freeze eval_spec.json before collecting the final set"
        meta["spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
        old = outdir / "meta.json"
        assert not old.exists() or json.load(open(old)).get("spec_sha256") == meta["spec_sha256"], "eval_spec.json changed since final collection began"
    json.dump(meta, open(outdir / "meta.json", "w"))
    runner = Runner(args.workers)
    labels = args.labels or LABELS

    def run(label, **kw):
        f = outdir / f"{label}.pkl"
        if f.exists():
            print("cached", label, flush=True); return
        t0 = time.time()
        res = runner.run([dict(seed=s, label=label, collect=True, **kw) for s in seeds])
        pickle.dump({"seeds": seeds, "outs": [o for o, _ in res], "traces": [t for _, t in res], "meta": meta}, open(f, "wb"), protocol=4)
        gd = np.array([o["goal_diff"] for o, _ in res])
        print(f"{label:26s} n={n} goal_diff {gd.mean():+.3f}  {time.time() - t0:.0f}s", flush=True)
        return res

    specs = {"native_unmodified": dict(policy="native"), "fixed_compact": dict(policy="fixed", action=0), "fixed_stretched": dict(policy="fixed", action=2),
             "hand_coach": dict(policy="hand"), "random_switching": dict(policy="random"), "q_learning": dict(policy="q", q=q, epsilon=0.0)}
    for label in labels:
        if label == "q_learning_shuffled_obs":
            src = outdir / "q_learning.pkl"
            assert src.exists(), "run q_learning first (its visited states form the shuffle pool)"
            pool = list(np.concatenate([t["feats"] for t in pickle.load(open(src, "rb"))["traces"]]))   # features the Q coach itself saw
            run(label, policy="q", q=q, epsilon=0.0, shuffle_pool=pool)
        else:
            run(label, **specs[label])
    runner.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
