"""Stage 1: runtime verification of the one-parameter native depth control.
Run with the patched overlay:  PYTHONPATH=$HOME/grf_native_poc/overlay python stage1_verify.py out.json
"""
import json
import sys
import time

import numpy as np

from poc_lib import AI, engine_module_path, has_setter, live_depth, load_states, make_env, rollout, set_depth, row, line_metrics

out = {"engine_module_path": engine_module_path()}
env = make_env()
env.reset()
assert has_setter(env), "patched engine not loaded"
out["setter_available"] = True
states = load_states("tactical_states_independent")
meta, snap = states[0]

# --- 1. parameter reaches the native TeamAIController and is applied immediately
env.set_state(snap)
set_depth(env, 0, -1.0); set_depth(env, 1, -1.0)
d0, d1 = live_depth(env, 0), live_depth(env, 1)
set_depth(env, 0, 0.95)
a0, a1 = live_depth(env, 0), live_depth(env, 1)
out["live_default_left_right"] = [d0, d1]
out["live_after_set_0.95_left_right"] = [a0, a1]

# --- 2. persistence over >= 10 native UpdateTactics() cycles (1000 ms match time each; 1 env step = 100 ms?)
samples = []
for step in range(300):
    env.step([AI] * 11)
    if step % 10 == 0:
        samples.append(live_depth(env, 0))
out["persistence_samples_every_10_steps_set_0.95"] = samples
out["persistence_unique_values"] = sorted(set(round(v, 6) for v in samples))
out["opponent_live_unchanged_after_300_steps"] = live_depth(env, 1)

# --- 3. clear override -> returns to native default within one update cycle
set_depth(env, 0, -1.0)
cleared_now = live_depth(env, 0)
for _ in range(20):
    env.step([AI] * 11)
out["after_clear_immediate_and_after_20_steps"] = [cleared_now, live_depth(env, 0)]

# --- 4. does set_state restore the tactical state?  (override member is intentionally NOT in ProcessState)
env.set_state(snap)
set_depth(env, 0, -1.0)
snap_default = env.get_state()                      # snapshot taken with native default
set_depth(env, 0, 0.95)
for _ in range(5):
    env.step([AI] * 11)
snap_override = env.get_state()                     # snapshot taken with override 0.95 active
env.set_state(snap_default)
after_restore_default_live = live_depth(env, 0)     # live value restored from snapshot (default)?
for _ in range(15):
    env.step([AI] * 11)
after_restore_then_15_steps = live_depth(env, 0)    # does the old override re-assert itself after restore?
set_depth(env, 0, -1.0)
env.set_state(snap_override)
live_after_restoring_override_snapshot = live_depth(env, 0)
for _ in range(15):
    env.step([AI] * 11)
live_after_restore_override_snapshot_15 = live_depth(env, 0)
out["set_state_probe"] = {
    "restore_default_snapshot_while_override_was_0.95__live_immediately": after_restore_default_live,
    "restore_default_snapshot_while_override_was_0.95__after_15_steps": after_restore_then_15_steps,
    "restore_override_snapshot_while_override_cleared__live_immediately": live_after_restoring_override_snapshot,
    "restore_override_snapshot__after_15_steps": live_after_restore_override_snapshot_15}

# --- 5. repeated commands every step do not change the trajectory (vs one command)
def run(mode, depth=0.95, n=120):
    env.set_state(snap)
    set_depth(env, 0, -1.0)
    if mode == "once":
        set_depth(env, 0, depth)
    traj = []
    for _ in range(n):
        if mode == "every_step":
            set_depth(env, 0, depth)
        obs, *_ = env.step([AI] * 11)
        s = row(obs)
        traj.append(np.concatenate([s["ball"], s["left"].ravel(), s["right"].ravel()]))
    return np.array(traj)
t0 = time.perf_counter(); once = run("once"); t_once = time.perf_counter() - t0
t0 = time.perf_counter(); every = run("every_step"); t_every = time.perf_counter() - t0
out["repeat_command_trajectory_max_abs_diff_once_vs_every_step"] = float(np.abs(once - every).max())
out["repeat_command_seconds_once_vs_every"] = [t_once, t_every]
once2 = run("once")
out["determinism_same_command_twice_max_abs_diff"] = float(np.abs(once - once2).max())

json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
env.close()
