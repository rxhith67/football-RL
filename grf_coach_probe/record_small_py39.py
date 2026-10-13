"""Record a native GRF academy episode using its own dump/video writer."""

import argparse
import json
import time
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
        env_name="academy_pass_and_shoot_with_keeper",
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=2,
        other_config_options={"action_set": "v2"},
        write_full_episode_dumps=True,
        write_video=True,
        render=False,
        logdir=str(logdir),
    )
    try:
        observation = env.reset()
        print("action_space", env.action_space, flush=True)
        events = []
        last_score = list(observation[0]["score"])
        last_owner = (int(observation[0]["ball_owned_team"]),
                      int(observation[0]["ball_owned_player"]))
        t0 = time.perf_counter()
        for step in range(401):
            observation, reward, done, _ = env.step([ai, ai])
            row = observation[0]
            score = list(row["score"])
            owner = (int(row["ball_owned_team"]),
                     int(row["ball_owned_player"]))
            if score != last_score or (owner != last_owner and owner[0] != -1):
                events.append({"step": step + 1, "score": score,
                               "owner": owner, "ball": list(row["ball"])})
            last_score, last_owner = score, owner
            if done:
                break
        print(json.dumps({"steps": step + 1, "done": bool(done),
                          "score": last_score, "events": events,
                          "elapsed_seconds": time.perf_counter() - t0,
                          "files": [str(p) for p in logdir.iterdir()]},
                         indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
