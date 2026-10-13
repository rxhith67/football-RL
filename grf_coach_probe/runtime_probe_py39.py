"""Exercise real GRF engine physics and the V2 built-in-AI control boundary."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import gfootball.env as football_env
from gfootball.env import football_action_set as action_set


AI = action_set.action_set_v2.index(action_set.action_builtin_ai)
RIGHT = action_set.action_set_v2.index(action_set.action_right)


def scalar_state(observation):
    row = observation[0] if isinstance(observation, list) else observation
    return {
        "ball": [float(x) for x in row["ball"]],
        "score": [int(x) for x in row["score"]],
        "ball_owned_team": int(row["ball_owned_team"]),
        "ball_owned_player": int(row["ball_owned_player"]),
        "left_team": [[float(x) for x in p] for p in row["left_team"]],
        "right_team": [[float(x) for x in p] for p in row["right_team"]],
        "active": [int(o["active"]) for o in observation]
        if isinstance(observation, list) else [int(row["active"])],
    }


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def run_full_match(scenario):
    env = football_env.create_environment(
        env_name=scenario,
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"},
    )
    try:
        observation = env.reset()
        start = scalar_state(observation)
        last = start
        possession = Counter()
        owner_switches = 0
        transfer_candidates = []
        previous_owner = None
        previous_owner_step = None
        possession_side_changes = 0
        previous_possession_side = None
        ball_distance = 0.0
        goals = []
        samples = []
        start_time = time.perf_counter()
        for step in range(3001):
            observation, reward, done, info = env.step([AI] * 11)
            state = scalar_state(observation)
            possession[str(state["ball_owned_team"])] += 1
            if (state["ball_owned_team"] == last["ball_owned_team"]
                    and state["ball_owned_team"] in (0, 1)
                    and state["ball_owned_player"] != last["ball_owned_player"]):
                owner_switches += 1
            current_owner = (state["ball_owned_team"], state["ball_owned_player"])
            if current_owner[0] in (0, 1) and current_owner != previous_owner:
                if (previous_possession_side is not None
                        and current_owner[0] != previous_possession_side):
                    possession_side_changes += 1
                previous_possession_side = current_owner[0]
                if (previous_owner is not None
                        and current_owner[0] == previous_owner[0]
                        and current_owner[1] != previous_owner[1]
                        and step + 1 - previous_owner_step <= 30):
                    transfer_candidates.append({
                        "step": step + 1, "team": current_owner[0],
                        "from_player": previous_owner[1],
                        "to_player": current_owner[1],
                        "gap_frames": step + 1 - previous_owner_step})
                previous_owner = current_owner
                previous_owner_step = step + 1
            elif current_owner[0] in (0, 1):
                previous_owner_step = step + 1
            ball_distance += distance(state["ball"], last["ball"])
            if state["score"] != last["score"]:
                goals.append({"step": step + 1, "score": state["score"]})
            if step % 100 == 0 or done:
                samples.append({"step": step + 1,
                                "ball": state["ball"],
                                "score": state["score"],
                                "ball_owned_team": state["ball_owned_team"]})
            last = state
            if done:
                break
        seconds = time.perf_counter() - start_time
        return {
            "scenario": scenario,
            "control": "all 11 left players action_builtin_ai; right team game AI",
            "action_space": str(env.action_space),
            "steps": step + 1,
            "episode_done": bool(done),
            "elapsed_seconds": seconds,
            "steps_per_second": (step + 1) / seconds,
            "initial_score": start["score"],
            "final_score": last["score"],
            "goals": goals,
            "ball_distance": ball_distance,
            "possession_frames": dict(possession),
            "possession_side_changes": possession_side_changes,
            "same_team_owner_switches": owner_switches,
            "intra_team_transfer_candidates": transfer_candidates[:20],
            "intra_team_transfer_candidate_count": len(transfer_candidates),
            "left_player_displacements": [distance(a, b) for a, b in zip(start["left_team"], last["left_team"])],
            "right_player_displacements": [distance(a, b) for a, b in zip(start["right_team"], last["right_team"])],
            "samples": samples,
        }
    finally:
        env.close()


def run_mixed_control():
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"},
    )
    try:
        observation = env.reset()
        states = [scalar_state(observation)]
        target_action_slot = 4
        target_player = states[0]["active"][target_action_slot]
        for step in range(150):
            actions = [AI] * 11
            if 50 <= step < 75:
                actions[target_action_slot] = RIGHT
            observation, _, done, _ = env.step(actions)
            states.append(scalar_state(observation))
            if done:
                break
        slices = {}
        for name, a, b in (("ai_before", 25, 50),
                           ("external_override", 50, 75),
                           ("ai_return", 75, 100)):
            if b >= len(states):
                continue
            before, after = states[a], states[b]
            slices[name] = {
                "target_player": target_player,
                "target_start": before["left_team"][target_player],
                "target_end": after["left_team"][target_player],
                "target_delta_x": after["left_team"][target_player][0] - before["left_team"][target_player][0],
                "other_left_players_moved": sum(
                    distance(before["left_team"][i], after["left_team"][i]) > 0.001
                    for i in range(11) if i != target_player),
                "ball_distance": sum(distance(states[k]["ball"], states[k - 1]["ball"])
                                     for k in range(a + 1, b + 1)),
            }
        return {
            "scenario": "11_vs_11_easy_stochastic",
            "steps": len(states) - 1,
            "target_action_slot": target_action_slot,
            "target_player": target_player,
            "override_action": RIGHT,
            "builtin_ai_action": AI,
            "segments": slices,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("full", "mixed"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenario", default="11_vs_11_easy_stochastic")
    args = parser.parse_args()
    result = run_full_match(args.scenario) if args.mode == "full" else run_mixed_control()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
