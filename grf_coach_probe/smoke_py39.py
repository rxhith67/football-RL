"""Minimal native-engine runtime check for the isolated WSL GRF build."""

import json
import time

import gfootball.env as football_env
from gfootball.env import football_action_set


def main():
    actions = football_action_set.action_set_v2
    ai_id = actions.index(football_action_set.action_builtin_ai)
    print("action_set_v2:", len(actions), "builtin_ai:", ai_id, flush=True)
    env = football_env.create_environment(
        env_name="academy_empty_goal_close",
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=1,
        other_config_options={"action_set": "v2"},
    )
    try:
        obs = env.reset()
        print("action_space:", env.action_space, flush=True)
        print("observation_type:", type(obs).__name__, flush=True)
        first = obs[0] if isinstance(obs, list) else obs
        print("keys:", list(first), flush=True)
        print("initial:", json.dumps({
            "ball": list(first["ball"]),
            "score": list(first["score"]),
            "ball_owned_team": int(first["ball_owned_team"]),
            "ball_owned_player": int(first["ball_owned_player"]),
            "left_team": [list(p) for p in first["left_team"]],
        }), flush=True)
        t0 = time.perf_counter()
        done = False
        for step in range(400):
            obs, reward, done, info = env.step(ai_id)
            if step in (0, 9, 49, 99) or done:
                first = obs[0] if isinstance(obs, list) else obs
                print("step:", step + 1, "ball:", first["ball"],
                      "score:", first["score"], "reward:", reward,
                      "done:", done, flush=True)
            if done:
                break
        print("steps:", step + 1, "elapsed_s:", round(time.perf_counter() - t0, 3), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
