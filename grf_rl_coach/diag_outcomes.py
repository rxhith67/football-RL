import sys
import numpy as np
from rollout import Runner

if __name__ == "__main__":
    seeds = [100000 + i for i in range(12)]
    r = Runner(12)
    for name, kw in (("native", dict(policy="native")), ("compact", dict(policy="fixed", action=0)), ("stretched", dict(policy="fixed", action=2))):
        res = r.run([dict(seed=s, **kw) for s in seeds])
        print(name, "gd", [e["goal_diff"] for e, _ in res], "GF", [e["goals_for"] for e, _ in res], "GA", [e["goals_against"] for e, _ in res], flush=True)
    r.close()
