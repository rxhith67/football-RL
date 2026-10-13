"""Paired, finite-horizon evaluation of GRF tactical movement instructions."""

import argparse
import json
import math
import time
from pathlib import Path

import gfootball.env as football_env

from tactical_translator_py39 import AI, TacticalTranslator, distance


def snapshot_row(obs):
    row = obs[0]
    return {
        "ball": [float(x) for x in row["ball"]],
        "ball_direction": [float(x) for x in row["ball_direction"]],
        "score": [int(x) for x in row["score"]],
        "owner_team": int(row["ball_owned_team"]),
        "owner_player": int(row["ball_owned_player"]),
        "left": [[float(x) for x in p] for p in row["left_team"]],
        "right": [[float(x) for x in p] for p in row["right_team"]],
    }


def defensive_shape(state):
    ball = state["ball"]
    teammates = state["left"][1:]
    carrier = state["right"][state["owner_player"]]
    distances = sorted(distance(player, carrier) for player in teammates)
    goal_side_near = sum(player[0] < ball[0] and distance(player, ball) < 0.36
                         for player in teammates)
    return {"nearest_two_carrier_distance": sum(distances[:2]) / 2.0,
            "goal_side_near_ball": goal_side_near}


def evaluate(env, snapshot, meta, policy, horizon, full_trace,
             translator_factory=TacticalTranslator):
    env.set_state(snapshot)
    obs = env.unwrapped.observation()
    state = snapshot_row(obs)
    assert state["owner_team"] == 1
    assert distance(state["ball"], meta["ball"]) < 1e-5
    translator = translator_factory(policy)
    start_score = state["score"]
    initial_left = state["left"]
    initial_ball_x = state["ball"][0]
    first_recovery = None
    first_conceded = None
    closest_goal_x = initial_ball_x
    dangerous_entries = 0
    was_dangerous = False
    possession_frames = {"left": 0, "right": 0, "free": 0}
    goalie_overrides = 0
    action_ai_counts = []
    player_override_frames = {}
    directed_progress = 0.0
    directed_samples = 0
    shape_samples = []
    trajectory = []
    last_state = state
    start_time = time.perf_counter()
    done = False
    for step in range(horizon):
        actions, action_meta = translator.actions(obs)
        assert len(actions) == 11
        assert len(action_meta["overridden"]) <= 2
        assert 0 not in action_meta["overridden"]
        if int(obs[0]["ball_owned_team"]) == 0:
            assert all(action == AI for action in actions)
        goalie_overrides += 0 in action_meta["overridden"]
        action_ai_counts.append(sum(action == AI for action in actions))
        for player in action_meta["overridden"]:
            key = str(player)
            player_override_frames[key] = player_override_frames.get(key, 0) + 1

        obs, reward, done, info = env.step(actions)
        state = snapshot_row(obs)
        for player in action_meta["overridden"]:
            target = action_meta["targets"][str(player)]
            before = last_state["left"][player]
            after = state["left"][player]
            dx, dy = target[0] - before[0], target[1] - before[1]
            length = math.hypot(dx, dy)
            if length > 0.022:
                directed_progress += ((after[0] - before[0]) * dx
                                      + (after[1] - before[1]) * dy) / length
                directed_samples += 1

        if state["score"][1] > start_score[1] and first_conceded is None:
            first_conceded = step + 1
        if state["owner_team"] == 0:
            possession_frames["left"] += 1
            if first_recovery is None and first_conceded is None:
                first_recovery = step + 1
        elif state["owner_team"] == 1:
            possession_frames["right"] += 1
            shape_samples.append(defensive_shape(state))
        else:
            possession_frames["free"] += 1

        if first_recovery is None and first_conceded is None:
            closest_goal_x = min(closest_goal_x, state["ball"][0])
        dangerous = (state["ball"][0] < -0.76
                     and abs(state["ball"][1]) < 0.22
                     and state["ball_direction"][0] < -0.004
                     and state["owner_team"] != 0)
        if dangerous and not was_dangerous:
            dangerous_entries += 1
        was_dangerous = dangerous

        if full_trace or step == 0 or (step + 1) % 5 == 0 or done:
            trajectory.append({"step": step + 1, "ball": state["ball"],
                               "score": state["score"],
                               "owner_team": state["owner_team"],
                               "owner_player": state["owner_player"],
                               "left": state["left"], "right": state["right"],
                               "overridden": action_meta["overridden"],
                               "targets": action_meta["targets"],
                               "commands": action_meta["commands"],
                               "effective_policy": action_meta["effective_policy"]})
        last_state = state
        if done:
            break
    elapsed = time.perf_counter() - start_time
    translator.finish()
    ever_overridden = {int(p) for p in player_override_frames}
    never_overridden = [i for i in range(11) if i not in ever_overridden]
    never_overridden_moved = sum(
        distance(initial_left[i], last_state["left"][i]) > 0.01
        for i in never_overridden)
    mean_shape = {
        key: (sum(sample[key] for sample in shape_samples) / len(shape_samples)
              if shape_samples else None)
        for key in ("nearest_two_carrier_distance", "goal_side_near_ball")}
    return {
        "state_id": meta["id"], "category": meta["category"],
        "policy": policy, "horizon": horizon, "steps": step + 1,
        "done": bool(done), "seconds": elapsed,
        "steps_per_second": (step + 1) / elapsed,
        "initial_ball": meta["ball"],
        "initial_score": start_score, "final_score": last_state["score"],
        "goals_conceded": last_state["score"][1] - start_score[1],
        "goals_scored": last_state["score"][0] - start_score[0],
        "recovered_before_conceding": first_recovery is not None,
        "time_to_recovery": first_recovery,
        "time_to_concede": first_conceded,
        "opponent_progress_toward_goal": initial_ball_x - closest_goal_x,
        "dangerous_ball_entries_proxy": dangerous_entries,
        "possession_frames": possession_frames,
        "mean_shape_while_opponent_possesses": mean_shape,
        "override_frames_by_player": player_override_frames,
        "override_engagements": translator.engagements,
        "goalkeeper_override_frames": goalie_overrides,
        "minimum_ai_players_per_step": min(action_ai_counts),
        "mean_ai_players_per_step": sum(action_ai_counts) / len(action_ai_counts),
        "never_overridden_players": never_overridden,
        "never_overridden_players_moved": never_overridden_moved,
        "mean_target_directed_progress_per_override_frame": (
            directed_progress / directed_samples if directed_samples else None),
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--policies", default="normal,press,cover")
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--full-trace", action="store_true")
    args = parser.parse_args()
    states_dir = Path(args.states)
    manifest = json.loads((states_dir / "manifest.json").read_text())
    ids = set(args.ids.split(",")) if args.ids else None
    metas = [m for m in manifest["states"] if ids is None or m["id"] in ids]
    policies = args.policies.split(",")
    env = football_env.create_environment(
        env_name=manifest["scenario"], representation="raw", rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"},
    )
    results = []
    try:
        env.reset()
        for meta in metas:
            snapshot = (states_dir / (meta["id"] + ".state")).read_bytes()
            for policy in policies:
                result = evaluate(env, snapshot, meta, policy,
                                  args.horizon, args.full_trace)
                results.append(result)
                print(json.dumps({k: v for k, v in result.items()
                                  if k not in ("trajectory", "override_engagements",
                                               "never_overridden_players")}),
                      flush=True)
    finally:
        env.close()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"scenario": manifest["scenario"],
                                  "state_ids": [m["id"] for m in metas],
                                  "policies": policies,
                                  "horizon": args.horizon,
                                  "results": results}, indent=2) + "\n")
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
