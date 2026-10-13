"""Stage 2: paired defensive rollouts for several native-depth settings. No RL, builtin AI for all 11 players.
usage: python stage2_paired.py STATES_DIR[,STATES_DIR2] OUT.json REPLICATE [HORIZON]
-1 = native default (override cleared explicitly before every rollout)."""
import json, sys
from poc_lib import make_env, load_states, rollout, has_setter
dirs = sys.argv[1].split(","); out = sys.argv[2]; rep = int(sys.argv[3]); horizon = int(sys.argv[4]) if len(sys.argv) > 4 else 200
SETTINGS = [-1.0, 0.0, 0.35, 0.95, 1.0]
rot = rep % len(SETTINGS); order = SETTINGS[rot:] + SETTINGS[:rot]      # vary order across replicates
env = make_env(); env.reset(); assert has_setter(env)
rows = []
for d in dirs:
    for meta, snap in load_states(d):
        for dep in order:
            r = rollout(env, snap, meta, dep, horizon)
            r.pop("traj"); r["replicate"] = rep; r["set"] = d
            rows.append(r)
json.dump({"horizon": horizon, "settings": SETTINGS, "rows": rows}, open(out, "w"))
print("done", out, len(rows))
