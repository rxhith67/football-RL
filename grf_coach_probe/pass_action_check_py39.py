"""Trigger supported pass and shot actions in GRF's existing academy scenario."""

import argparse
import json

import gfootball.env as football_env
from gfootball.env import football_action_set as action_set


AI = action_set.action_set_v2.index(action_set.action_builtin_ai)
PASS = action_set.full_action_set.index(action_set.action_short_pass)
RELEASE_PASS = action_set.full_action_set.index(action_set.action_release_short_pass)
SHOT = action_set.full_action_set.index(action_set.action_shot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="academy_pass_and_shoot_with_keeper")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--hold", type=int, default=1)
    args = parser.parse_args()
    env = football_env.create_environment(
        env_name=args.scenario,
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=args.players,
        other_config_options={"action_set": "full"},
    )
    results = []
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            pass_step = None
            pass_slot = None
            after_pass = []
            last_ball = list(obs[0]["ball"])
            shot_step = None
            for step in range(400):
                actions = [AI] * args.players
                owner_team = int(obs[0]["ball_owned_team"])
                owner_player = int(obs[0]["ball_owned_player"])
                if owner_team == 0:
                    slots = [i for i, row in enumerate(obs)
                             if int(row["active"]) == owner_player]
                    if slots and pass_step is None:
                        actions[slots[0]] = PASS
                        pass_step = step + 1
                        pass_slot = slots[0]
                    elif (slots and pass_step is not None
                          and step > pass_step + args.hold + 8
                          and shot_step is None):
                        actions[slots[0]] = SHOT
                        shot_step = step + 1
                if pass_step is not None:
                    if pass_step < step + 1 < pass_step + args.hold:
                        actions[pass_slot] = PASS
                    elif step + 1 == pass_step + args.hold:
                        actions[pass_slot] = RELEASE_PASS
                obs, reward, done, _ = env.step(actions)
                row = obs[0]
                if pass_step is not None and len(after_pass) < 80:
                    after_pass.append({"step": step + 1,
                                       "ball": [float(x) for x in row["ball"]],
                                       "owner_team": int(row["ball_owned_team"]),
                                       "owner_player": int(row["ball_owned_player"])})
                last_ball = list(row["ball"])
                if done:
                    break
            results.append({"episode": episode, "steps": step + 1,
                            "pass_action_step": pass_step,
                            "shot_action_step": shot_step,
                            "final_score": list(obs[0]["score"]),
                            "after_pass": after_pass})
    finally:
        env.close()
    print(json.dumps({"scenario": args.scenario, "players": args.players,
                      "hold_frames": args.hold,
                      "pass_action_id": PASS, "release_pass_action_id": RELEASE_PASS,
                      "shot_action_id": SHOT,
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
