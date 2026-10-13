"""Dump default-setting trajectories (builtin AI for all 11) so patched vs original engines can be compared."""
import sys, numpy as np
from poc_lib import make_env, load_states, rollout, engine_module_path, has_setter
env = make_env(); env.reset()
states = load_states("tactical_states_independent")[:6]
out = {}
for meta, snap in states:
    r = rollout(env, snap, meta, -1.0, horizon=150, trace=True)
    out[meta["id"]] = r["traj"]
np.savez(sys.argv[1], **out)
print(engine_module_path(), "setter" if has_setter(env) else "original", list(out))
