"""Parallel full-match rollouts. One GRF engine per worker process (spawn). Episodes are
deterministic functions of (seed, action sequence), which makes paired evaluation valid."""
import multiprocessing as mp
import time

import numpy as np

from coach_env import GRFDepthCoachEnv, ACTION_NAMES
import policies

_ENV = None


def _init(decision_steps, shaping, gamma):
    global _ENV
    _ENV = GRFDepthCoachEnv(decision_steps=decision_steps, shaping=shaping, gamma=gamma)


def run_episode(spec):
    """spec keys: policy ('fixed'|'native'|'hand'|'q'|'dqn'), seed, epsilon, action, q, weights,
    shuffle_pool (list of discrete states or feature rows used to destroy observation dependence),
    collect (bool)"""
    e = _ENV
    kind = spec["policy"]
    e.noop_calls = (kind == "native")                         # 'native' never calls the setter at all
    rng = np.random.default_rng(spec["seed"] + 12345)
    eps = spec.get("epsilon", 0.0)
    pool = spec.get("shuffle_pool")
    t0 = time.perf_counter()
    f = e.reset(seed=spec["seed"])
    done = False
    trans, acts = [], []
    tot = dict(r=0.0, r_goal=0.0, r_shape=0.0, gains=0, danger=0, poss0=0, poss1=0, possn=0,
               len_sum=0.0, back_sum=0.0, min_ball=1.0)
    n_dec = 0
    goal_events = []                                           # (decision index, goals_for, goals_against) per interval
    while not done:
        s = e.discretize(f)
        fi = f
        if pool is not None:                                   # policy-dependence control
            fi = pool[int(rng.integers(len(pool)))]
            s_in = e.discretize(fi)
        else:
            s_in = s
        if kind == "fixed":
            a = spec["action"]
        elif kind == "native":
            a = 1
        elif kind == "random":                                 # uniform random tactical switching every decision
            a = int(rng.integers(3))
        elif kind == "hand":
            a = policies.hand_coach(fi)
        elif kind == "q":
            a = policies.q_action(spec["q"], s_in, rng, eps, tie_default=(eps == 0.0))
        elif kind == "dqn":
            a = policies.dqn_action(spec["weights"], fi, rng, eps)
        else:
            raise ValueError(kind)
        f2, r, done, info = e.step(a)
        s2 = e.discretize(f2)
        if spec.get("collect"):
            trans.append((s, f.copy(), a, r, s2, f2.copy(), done))
        acts.append(a)
        tot["r"] += r; tot["r_goal"] += info["r_goal"]; tot["r_shape"] += info["r_shape"]
        tot["gains"] += info["possession_gains"]; tot["danger"] += info["danger_steps"]
        p = info["possession_steps"]
        tot["poss0"] += p[0]; tot["poss1"] += p[1]; tot["possn"] += p[-1]
        tot["len_sum"] += info["team_length"]; tot["back_sum"] += info["back_line_x"]
        tot["min_ball"] = min(tot["min_ball"], info["min_ball_x_opp_possession"])
        n_dec += 1
        if info["goals_for"] or info["goals_against"]:
            goal_events.append((n_dec, info["goals_for"], info["goals_against"]))
        f = f2
    gf, ga = info["score"]
    owned = tot["poss0"] + tot["poss1"]
    out = {"seed": spec["seed"], "policy": kind, "label": spec.get("label", kind), "goals_for": gf, "goals_against": ga,
           "goal_diff": gf - ga, "points": 3 if gf > ga else (1 if gf == ga else 0),
           "return": tot["r"], "return_goal": tot["r_goal"], "return_shape": tot["r_shape"],
           "decisions": n_dec, "engine_steps": e.engine_steps,
           "possession_share": tot["poss0"] / max(1, owned), "possession_gains": tot["gains"],
           "danger_steps": tot["danger"], "opp_max_progress": -tot["min_ball"],
           "mean_team_length": tot["len_sum"] / n_dec, "mean_back_line_x": tot["back_sum"] / n_dec,
           "action_counts": [acts.count(i) for i in range(3)], "actions": acts, "goal_events": goal_events, "seconds": time.perf_counter() - t0}
    return out, trans


class Runner:
    def __init__(self, workers=6, decision_steps=100, shaping=0.1, gamma=0.99):
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init, initargs=(decision_steps, shaping, gamma))

    def run(self, specs):
        return self.pool.map(run_episode, specs, chunksize=1)

    def run_fn(self, fn, specs):
        return self.pool.map(fn, specs, chunksize=1)

    def close(self):
        self.pool.close(); self.pool.join()
