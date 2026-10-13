"""Validate that saved defensive states are live play with a nearby carrier."""

import argparse
import json
import math
from pathlib import Path

import gfootball.env as football_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    args = parser.parse_args()
    folder = Path(args.states)
    manifest = json.loads((folder / "manifest.json").read_text())
    env = football_env.create_environment(
        env_name=manifest["scenario"], representation="raw", rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"})
    result = []
    try:
        env.reset()
        for item in manifest["states"]:
            env.set_state((folder / (item["id"] + ".state")).read_bytes())
            row = env.unwrapped.observation()[0]
            carrier = int(row["ball_owned_player"])
            ball = row["ball"]
            player = row["right_team"][carrier]
            result.append({"id": item["id"], "game_mode": int(row["game_mode"]),
                           "owner_team": int(row["ball_owned_team"]),
                           "owner_player": carrier,
                           "carrier_ball_distance": math.hypot(float(player[0]-ball[0]),
                                                                float(player[1]-ball[1])),
                           "ball_height": float(ball[2])})
    finally:
        env.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
