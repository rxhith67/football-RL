"""Test whether GRF snapshots support paired tactical rollouts."""

import json
import math

import gfootball.env as football_env
from gfootball.env import football_action_set


def state_vector(obs):
    row = obs[0]
    return ([float(x) for x in row["ball"]]
            + [float(x) for team in (row["left_team"], row["right_team"])
               for player in team for x in player]
            + [int(x) for x in row["score"]]
            + [int(row["ball_owned_team"]), int(row["ball_owned_player"])])


def main():
    ai = football_action_set.action_set_v2.index(
        football_action_set.action_builtin_ai)
    env = football_env.create_environment(
        env_name="11_vs_11_stochastic",
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"},
    )
    try:
        obs = env.reset()
        for _ in range(150):
            obs, _, done, _ = env.step([ai] * 11)
            assert not done
        snapshot = env.get_state()
        initial = state_vector(obs)
        first = []
        for _ in range(50):
            obs, _, done, _ = env.step([ai] * 11)
            first.append(state_vector(obs))
        env.set_state(snapshot)
        restored = state_vector(env.unwrapped.observation())
        second = []
        for _ in range(50):
            obs, _, done, _ = env.step([ai] * 11)
            second.append(state_vector(obs))
        max_initial_difference = max(abs(a - b) for a, b in zip(initial, restored))
        max_replay_difference = max(abs(a - b)
                                    for row_a, row_b in zip(first, second)
                                    for a, b in zip(row_a, row_b))
        print(json.dumps({"snapshot_bytes": len(snapshot),
                          "max_initial_difference": max_initial_difference,
                          "max_replay_difference": max_replay_difference,
                          "replay_exact": math.isclose(max_replay_difference, 0.0,
                                                       abs_tol=1e-10)}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
