import time
import numpy as np
from coach_env import *

e = GRFDepthCoachEnv()
t = time.time()
f = e.reset(seed=1)
print("obs", dict(zip(FEATURE_NAMES, np.round(f, 3))), "disc", e.discretize(f))
for a in (0, 1, 2, 2):
    f, r, d, i = e.step(a)
    print(a, round(r, 4), d, {k: i[k] for k in ("r_goal", "r_shape", "live_depth", "team_length", "back_line_x", "score", "steps_left")})
print("sec", time.time() - t, "engine steps", e.engine_steps)
