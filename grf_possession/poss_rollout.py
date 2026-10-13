"""Parallel full-match rollouts for the possession experiments (spawned workers, one GRF engine each).
A match is a deterministic function of (seed, side, action sequence)."""
import multiprocessing as mp
import time

import numpy as np

import policies
import possession as P
from poss_env import PossessionCoachEnv

_ENV = None


def _init(decision_steps=100, shaping=0.1, gamma=0.99):
    global _ENV
    _ENV = PossessionCoachEnv(decision_steps=decision_steps, shaping=shaping, gamma=gamma)


def run_episode(spec):
    """spec: policy in {fixed, native, hand, random, q}; seed; side (0/1); reward_mode; q; epsilon; action;
    shuffle_pool; collect (return transitions); keep_stream (return per-step stream)."""
    e = _ENV
    kind = spec["policy"]
    e.side = spec.get("side", 0)
    e.reward_mode = spec.get("reward_mode", "possession")
    e.noop_calls = (kind == "native")                         # 'native' never touches either team's tactics
    rng = np.random.default_rng(spec["seed"] + 12345 + 7 * e.side)
    eps = spec.get("epsilon", 0.0)
    pool = spec.get("shuffle_pool")
    t0 = time.perf_counter()
    f = e.reset(seed=spec["seed"])
    done = False
    trans, acts, ivs = [], [], []
    while not done:
        s = e.discretize(f)
        fi, s_in = f, s
        if pool is not None:                                  # policy-dependence control
            fi = pool[int(rng.integers(len(pool)))]
            s_in = e.discretize(fi)
        if kind == "fixed": a = spec["action"]
        elif kind == "native": a = 1
        elif kind == "random": a = int(rng.integers(3))
        elif kind == "hand": a = policies.hand_coach(fi)
        elif kind == "q": a = policies.q_action(spec["q"], s_in, rng, eps, tie_default=(eps == 0.0))
        else: raise ValueError(kind)
        f2, r, done, info = e.step(a)
        if spec.get("collect"):
            trans.append((s, f.copy(), a, r, e.discretize(f2), f2.copy(), done))
        acts.append(a); ivs.append(info)
        f = f2
    gf, ga = info["score"]
    match = P.match_possession([i["poss"] for i in ivs])
    shares = [i["poss"]["share"] for i in ivs]
    ep = {"seed": spec["seed"], "label": spec.get("label", kind), "side": e.side, "policy": kind,
          "goals_for": gf, "goals_against": ga, "goal_diff": gf - ga, "points": 3 if gf > ga else (1 if gf == ga else 0),
          "return_original": sum(i["reward_original"] for i in ivs), "return_possession": sum(i["reward_possession"] for i in ivs),
          "return_trained_on": sum(i["reward_possession" if e.reward_mode == "possession" else "reward_original"] for i in ivs),
          "possession_share": match["share"], "contested_frac": match["contested_frac"], "dead_frac": match["dead_frac"],
          "controlled_steps": match["ours"] + match["opp"], "ours_steps": match["ours"], "opp_steps": match["opp"],
          "interval_shares": [None if x is None else float(x) for x in shares],
          "decisions": len(acts), "engine_steps": e.engine_steps, "actions": acts, "action_counts": [acts.count(i) for i in range(3)],
          "interval_engine_steps": [i["engine_steps"] for i in ivs], "interval_goals": [(i["goals_for"], i["goals_against"]) for i in ivs],
          "live_depth_end": [i["live_depth"] for i in ivs], "team_length": float(np.mean([i["team_length"] for i in ivs])),
          "seconds": time.perf_counter() - t0}
    stream = np.array(e.stream, np.float32) if spec.get("keep_stream") else None
    return ep, trans, stream


class Runner:
    def __init__(self, workers=6, **env_kw):
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init, initargs=(env_kw.get("decision_steps", 100), env_kw.get("shaping", 0.1), env_kw.get("gamma", 0.99)))

    def run(self, specs):
        return self.pool.map(run_episode, specs, chunksize=1)

    def close(self):
        self.pool.close(); self.pool.join()
