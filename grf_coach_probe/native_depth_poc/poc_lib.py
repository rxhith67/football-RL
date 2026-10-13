"""Helpers for the one-parameter native-depth proof of concept (run inside WSL, py39 env).

The patched engine is selected ONLY by PYTHONPATH=$HOME/grf_native_poc/overlay; without it the
original, untouched install is used (needed for the regression comparison).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import gfootball.env as football_env
from gfootball.env import football_action_set

AI = football_action_set.action_set_v2.index(football_action_set.action_builtin_ai)
PROBE = Path(os.environ.get("PROBE_DIR", Path(__file__).resolve().parent.parent))
SCENARIO = "11_vs_11_hard_stochastic"


def engine_module_path():
    import gfootball_engine
    return os.path.dirname(gfootball_engine.__file__)


def make_env():
    return football_env.create_environment(
        env_name=SCENARIO, representation="raw", rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"})


def engine(env):
    """The native GameEnv object (boost.python)."""
    return env.unwrapped._env._env


def has_setter(env):
    return hasattr(engine(env), "set_coach_defense_depth")


def set_depth(env, team, value):
    """value in [0,1]; value < 0 clears the override. No-op on the unpatched engine."""
    if has_setter(env):
        engine(env).set_coach_defense_depth(int(team), float(value))


def live_depth(env, team):
    return engine(env).get_live_defense_depth(int(team)) if has_setter(env) else None


def load_states(dirname):
    d = PROBE / dirname
    manifest = json.loads((d / "manifest.json").read_text())
    out = []
    for m in manifest["states"]:
        out.append((m, (d / (m["id"] + ".state")).read_bytes()))
    return out


def row(obs):
    r = obs[0]
    return {"ball": [float(x) for x in r["ball"]],
            "ball_dir": [float(x) for x in r["ball_direction"]],
            "score": [int(x) for x in r["score"]],
            "owner": int(r["ball_owned_team"]), "owner_player": int(r["ball_owned_player"]),
            "left": np.array([[float(x) for x in p] for p in r["left_team"]]),
            "right": np.array([[float(x) for x in p] for p in r["right_team"]])}


def line_metrics(left):
    """left: (11,2) with index 0 = goalkeeper. Our goal is at x = -1."""
    out = left[1:, 0]
    xs = np.sort(out)
    return {"back_line_x": float(xs[:4].mean()),      # mean x of the 4 deepest outfield players
            "front_x": float(xs[-3:].mean()),
            "team_length": float(xs.max() - xs.min()),
            "team_width": float(left[1:, 1].max() - left[1:, 1].min())}


def rollout(env, snapshot, meta, depth, horizon=200, team=0, trace=False):
    """Restore snapshot, set the coach depth for `team` (explicit; depth<0 = native default),
    always make the opponent default, and play `horizon` steps with builtin AI for all 11."""
    env.set_state(snapshot)
    set_depth(env, 1, -1.0)
    set_depth(env, team, depth)
    obs = env.unwrapped.observation()
    s = row(obs)
    start_score = s["score"]
    init_ball_x = s["ball"][0]
    first_rec = first_conc = None
    closest = init_ball_x
    dang = 0
    was_d = False
    poss = {"left": 0, "right": 0, "free": 0}
    shape = {"back_line_x": [], "front_x": [], "team_length": [], "team_width": []}
    all_shape = {"back_line_x": [], "team_length": []}
    traj = []
    live = []
    for step in range(horizon):
        obs, _, done, _ = env.step([AI] * 11)
        s = row(obs)
        if depth is not None and step % 10 == 0:
            live.append(live_depth(env, team))
        if s["score"][1] > start_score[1] and first_conc is None:
            first_conc = step + 1
        if s["owner"] == 0:
            poss["left"] += 1
            if first_rec is None and first_conc is None:
                first_rec = step + 1
        elif s["owner"] == 1:
            poss["right"] += 1
            lm = line_metrics(s["left"])
            for k in shape:
                shape[k].append(lm[k])
        else:
            poss["free"] += 1
        lm = line_metrics(s["left"])
        for k in all_shape:
            all_shape[k].append(lm[k])
        if first_rec is None and first_conc is None:
            closest = min(closest, s["ball"][0])
        d = (s["ball"][0] < -0.76 and abs(s["ball"][1]) < 0.22 and s["ball_dir"][0] < -0.004 and s["owner"] != 0)
        if d and not was_d:
            dang += 1
        was_d = d
        if trace:
            traj.append(np.concatenate([[s["ball"][0], s["ball"][1]], s["left"].ravel(), s["right"].ravel()]))
        if done:
            break
    mean = lambda v: (float(np.mean(v)) if len(v) else None)
    return {"state_id": meta["id"], "category": meta["category"], "depth": depth, "steps": step + 1,
            "goals_conceded": s["score"][1] - start_score[1], "goals_scored": s["score"][0] - start_score[0],
            "recovered": first_rec is not None, "time_to_recovery": first_rec,
            "recovery_censored": first_rec if first_rec is not None else horizon + 1,
            "time_to_concede": first_conc,
            "opp_progress": init_ball_x - closest, "dangerous_entries": dang, "possession": poss,
            "shape_opp_possession": {k: mean(v) for k, v in shape.items()},
            "shape_all_frames": {k: mean(v) for k, v in all_shape.items()},
            "live_depth_samples": live,
            "traj": np.array(traj) if trace else None}
