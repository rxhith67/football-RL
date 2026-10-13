"""Profiling only (no training): where does CPU time go?  One engine process, 300 engine steps, plus learner-update timing.
usage: bash run.sh profile_run.py"""
import cProfile
import io
import pstats
import time

import numpy as np

from learners import q_update, double_q_update
from poss_env import PossessionCoachEnv

env = PossessionCoachEnv(side=0, reward_mode="possession")
env.reset(seed=777777)
pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable()
for a in (0, 1, 2):
    env.step(a)
pr.disable(); wall = time.perf_counter() - t0
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative"); ps.print_stats(14)
print(f"300 engine steps in {wall:.2f} s ({300 / wall:.0f} steps/s)")
tot = ps.total_tt
rows = {f"{k[2]}": v for k, v in ps.stats.items()}
def cum(name, file_hint=""):
    for (fn, ln, nm), (cc, nc, tt, ct, callers) in ps.stats.items():
        if nm == name and file_hint in fn: return ct
    return 0.0
eng = cum("step", "football_env_core") or cum("step_python") or cum("step")
print(f"time inside the native engine call chain (football_env_core.step -> C++ GameEnv.step): {eng:.2f} s of {wall:.2f} s = {100 * eng / wall:.1f}%")
print("python-side per-step observation handling in this wrapper is the remainder")
print(s.getvalue()[:2500])
# learner cost for one 12-match batch (360 transitions)
rng = np.random.default_rng(0)
trans = [(int(rng.integers(243)), None, int(rng.integers(3)), float(rng.normal()), int(rng.integers(243)), None, i % 30 == 29) for i in range(360)]
q = np.zeros((243, 3)); q1 = np.zeros((243, 3)); q2 = np.zeros((243, 3)); v = np.zeros((243, 3), int)
t = time.perf_counter()
for _ in range(100): q_update(q, v, trans, 0.1, 0.99)
tq = (time.perf_counter() - t) / 100
t = time.perf_counter(); r = np.random.default_rng(1)
for _ in range(100): double_q_update(q1, q2, v, trans, 0.1, 0.99, r)
td = (time.perf_counter() - t) / 100
print(f"Q-learning update for one 12-match batch (360 decisions): {tq * 1e3:.2f} ms; Double Q: {td * 1e3:.2f} ms; one batch of simulation is about 60,000 ms")
