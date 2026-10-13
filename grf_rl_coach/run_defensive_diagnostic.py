"""Paired exploratory 200-step defensive rollouts from fresh native-GRF states.

Run via: bash run.sh run_defensive_diagnostic.py
Three independent engine processes repeat every state/policy comparison. The
same snapshot is restored before each policy; the coach override is explicitly
cleared because GRF snapshots do not restore its external override member.
"""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from coach_env import GRFDepthCoachEnv, AI, apply_depth, live_depth, raw
from policies import hand_coach, q_action


ROOT = Path("results/defensive_diagnostic")
PROTOCOL = json.loads(Path("defensive_diagnostic_protocol.json").read_text())
KINDS = list(PROTOCOL["policies"])


def policy_action(kind, f, e, q, rng):
    if kind == "native":
        return 1
    if kind == "compact":
        return 0
    if kind == "stretched":
        return 2
    if kind == "hand":
        return hand_coach(f)
    if kind == "random":
        return int(rng.integers(3))
    if kind == "q":
        return q_action(q, e.discretize(f), rng, 0.0, tie_default=True)
    raise ValueError(kind)


def one_rollout(e, snapshot, meta, kind, q, rng, rep):
    e.env.set_state(snapshot)
    apply_depth(e.env, 0, None)
    apply_depth(e.env, 1, None)
    e.noop_calls = kind == "native"
    e.last_action = 1
    e.share_last = float(meta["feature"][12])
    e.prev = raw(e.env.unwrapped.observation())
    e.last_features = e._features(e.prev, e.share_last)
    e.engine_steps = e.decisions = 0
    e.cum = {"goals_for": 0, "goals_against": 0}
    e._started = True
    assert e.last_features.shape == (14,)
    assert np.allclose(e.last_features, np.asarray(meta["feature"], np.float32), atol=1e-6), meta["id"]
    assert abs(live_depth(e.env, 0) - 0.71) < 1e-3
    start_score = e.prev["score"][:]
    start_x = float(e.prev["ball"][0])
    metrics = {"step": 0, "first_recovery": None, "first_concede": None,
               "min_ball_x_before_outcome": start_x, "dangerous_entries": 0,
               "dangerous_steps": 0, "shot_like_releases": 0, "all_ai_actions": True}
    prev_owner = int(e.prev["owner"])
    prev_danger = False
    prev_ball = np.asarray(e.prev["ball"])
    real_step = e.env.step

    def spy(actions):
        nonlocal prev_owner, prev_danger, prev_ball
        metrics["all_ai_actions"] &= len(actions) == 11 and all(a == AI for a in actions)
        obs, reward, done, info = real_step(actions)
        r = obs[0]
        metrics["step"] += 1
        owner = int(r["ball_owned_team"])
        score = r["score"]
        ball = np.asarray(r["ball"], float)
        if metrics["first_recovery"] is None and metrics["first_concede"] is None:
            metrics["min_ball_x_before_outcome"] = min(metrics["min_ball_x_before_outcome"], float(ball[0]))
            if score[1] > start_score[1]:
                metrics["first_concede"] = metrics["step"]
            elif owner == 0:
                metrics["first_recovery"] = metrics["step"]
        danger = owner != 0 and ball[0] < -0.76 and abs(ball[1]) < 0.22
        if danger:
            metrics["dangerous_steps"] += 1
        if danger and not prev_danger:
            metrics["dangerous_entries"] += 1
        direction = np.asarray(r["ball_direction"], float)
        if prev_owner == 1 and owner == -1 and prev_ball[0] < -0.45 and direction[0] < -0.004:
            # A transparent proxy: fast opponent release toward the left goal,
            # including possible crosses/passes. GRF does not expose shot events.
            y_goal = float(ball[1] + (-1.0 - ball[0]) * direction[1] / direction[0])
            if abs(y_goal) < 0.25:
                metrics["shot_like_releases"] += 1
        prev_owner, prev_danger, prev_ball = owner, danger, ball
        return obs, reward, done, info

    e.env.step = spy
    f = e.last_features
    actions = []
    live = []
    try:
        while metrics["step"] < PROTOCOL["rollout_horizon_engine_steps"]:
            a = policy_action(kind, f, e, q, rng)
            f, _, done, info = e.step(a)
            actions.append(int(a))
            live.append(float(info["live_depth"]) if info["live_depth"] is not None else float(live_depth(e.env, 0)))
            if done:
                break
    finally:
        e.env.step = real_step
    assert metrics["all_ai_actions"]
    assert metrics["step"] == PROTOCOL["rollout_horizon_engine_steps"]
    expected = {0: 0.0, 1: 0.71, 2: 1.0}
    assert all(abs(depth - expected[action]) < 1e-3 for action, depth in zip(actions, live))
    end_score = e.prev["score"]
    rec = metrics["first_recovery"]
    return {"state_id": meta["id"], "category": meta["category"],
            "source_seed": meta["source_seed"], "replicate": rep, "policy": kind,
            "initial_feature": meta["feature"], "initial_state_id": meta["state_id"],
            "actions": actions, "live_depths": live, "engine_steps": metrics["step"],
            "recovered": int(rec is not None), "steps_to_recovery": rec,
            "recovery_censored": rec if rec is not None else 201,
            "opponent_progress": start_x - metrics["min_ball_x_before_outcome"],
            "dangerous_entries": metrics["dangerous_entries"],
            "dangerous_steps": metrics["dangerous_steps"],
            "shot_like_releases": metrics["shot_like_releases"],
            "goals_conceded": int(end_score[1] - start_score[1]),
            "goals_scored": int(end_score[0] - start_score[0]),
            "goal_difference": int((end_score[0] - start_score[0]) - (end_score[1] - start_score[1]))}


def worker(rep):
    out = ROOT / ("replicate_%d.json" % rep)
    if out.exists():
        return str(out)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    q = np.load(PROTOCOL["checkpoint"])["q"]
    e = GRFDepthCoachEnv(decision_steps=100)
    e.reset(seed=PROTOCOL["replicate_seed_start"] + rep)
    rows = []
    try:
        for index, meta in enumerate(manifest["states"]):
            snapshot = (ROOT / (meta["id"] + ".state")).read_bytes()
            assert hashlib.sha256(snapshot).hexdigest() == meta["state_sha256"]
            offset = rep % len(KINDS)
            order = KINDS[offset:] + KINDS[:offset]
            for kind in order:
                rng = np.random.default_rng(PROTOCOL["replicate_seed_start"] + rep * 1000 + index)
                rows.append(one_rollout(e, snapshot, meta, kind, q, rng, rep))
            if index % 5 == 4:
                print("replicate", rep, "states", index + 1, flush=True)
    finally:
        e.close()
    out.write_text(json.dumps({"replicate": rep, "rows": rows}) + "\n")
    return str(out)


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert all(v == PROTOCOL["snapshots_per_category"] for v in manifest["counts"].values())
    assert hashlib.sha256(Path(PROTOCOL["checkpoint"]).read_bytes()).hexdigest() == PROTOCOL["checkpoint_sha256"]
    assert len({x["source_seed"] for x in manifest["states"]}) == len(manifest["states"])
    ctx = mp.get_context("spawn")
    with ctx.Pool(PROTOCOL["replicate_processes"]) as pool:
        files = pool.map(worker, range(PROTOCOL["replicate_processes"]))
    print("completed", files)


if __name__ == "__main__":
    main()
