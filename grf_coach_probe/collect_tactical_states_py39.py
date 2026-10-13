"""Save naturally occurring 11v11 defensive snapshots for paired rollouts."""

import argparse
import json
from pathlib import Path

import gfootball.env as football_env
from gfootball.env import football_action_set


AI = football_action_set.action_set_v2.index(football_action_set.action_builtin_ai)


def category(ball):
    x, y, z = float(ball[0]), float(ball[1]), float(ball[2])
    if z > 0.20:
        return None
    if x <= -0.55 and abs(y) <= 0.24:
        return "penalty"
    if -0.50 <= x <= 0.15 and abs(y) >= 0.23:
        return "wide"
    if -0.30 <= x <= 0.30 and abs(y) <= 0.18:
        return "midfield"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenario", default="11_vs_11_hard_stochastic")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--min-gap", type=int, default=130)
    parser.add_argument("--max-per-episode", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    env = football_env.create_environment(
        env_name=args.scenario,
        representation="raw",
        rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"},
    )
    targets = ("midfield", "penalty", "wide")
    counts = {key: 0 for key in targets}
    items = []
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            last_saved = -args.min_gap
            episode_counts = {key: 0 for key in targets}
            for step in range(3001):
                obs, _, done, _ = env.step([AI] * 11)
                row = obs[0]
                kind = category(row["ball"]) if int(row["ball_owned_team"]) == 1 else None
                if (step >= 50 and kind and counts[kind] < args.per_category
                        and episode_counts[kind] < args.max_per_episode
                        and step - last_saved >= args.min_gap):
                    snapshot = env.get_state()
                    name = "%s_%02d" % (kind, counts[kind])
                    (output / (name + ".state")).write_bytes(snapshot)
                    item = {
                        "id": name, "category": kind, "scenario": args.scenario,
                        "episode": episode, "step": step + 1,
                        "ball": [float(x) for x in row["ball"]],
                        "ball_owned_team": int(row["ball_owned_team"]),
                        "ball_owned_player": int(row["ball_owned_player"]),
                        "score": [int(x) for x in row["score"]],
                        "state_bytes": len(snapshot),
                    }
                    items.append(item)
                    counts[kind] += 1
                    episode_counts[kind] += 1
                    last_saved = step
                    print(json.dumps(item), flush=True)
                if done or all(counts[k] >= args.per_category for k in targets):
                    break
            print("episode", episode, "steps", step + 1, "counts", counts,
                  flush=True)
            if all(counts[k] >= args.per_category for k in targets):
                break
    finally:
        env.close()
    manifest = {"scenario": args.scenario, "episodes_attempted": episode + 1,
                "min_gap": args.min_gap, "counts": counts, "states": items}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest", output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
