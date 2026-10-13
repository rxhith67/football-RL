import sys
import numpy as np
from coach_env import *

burn = int(sys.argv[1]) if len(sys.argv) > 1 else 1
e = GRFDepthCoachEnv(noop_calls=True)
def run(seed, n=200):
    e.reset(seed=seed)
    L = []
    for _ in range(n):
        obs, *_ = e.env.step([AI] * 11)
        s = raw(obs); L.append(np.concatenate([s["ball"], s["left"].ravel(), s["right"].ravel()]))
    return np.array(L)
for _ in range(burn):
    run(99, 20)
out = []
for seed in (5, 7, 5, 5):
    T = run(seed)
    out.append(round(float(T.sum()), 4))
print("burn", burn, "checksums seeds (5,7,5,5):", out)
