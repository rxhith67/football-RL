"""Save a short native 2D replay from an actual GRF 11v11 match."""

import argparse
import json
from pathlib import Path

import gfootball.env as football_env
from gfootball.env import football_action_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    ai = football_action_set.action_set_v2.index(
        football_action_set.action_builtin_ai)
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"},
        write_video=True,
        render=False,
        logdir=str(logdir),
    )
    try:
        obs = env.reset()
        goals = []
        last_score = list(obs[0]["score"])
        for step in range(850):
            obs, _, done, _ = env.step([ai] * 11)
            score = list(obs[0]["score"])
            if score != last_score:
                goals.append({"step": step + 1, "score": score})
            last_score = score
            if done:
                break
        dump_prefix = env.write_dump("shutdown")
        print(json.dumps({"steps": step + 1, "score": last_score,
                          "goals": goals, "dump_prefix": dump_prefix,
                          "files": [str(p) for p in logdir.iterdir()]},
                         indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
