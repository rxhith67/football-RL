"""Check that GRF's normal football skills survive sparse tactical overrides."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import gfootball.env as football_env

from tactical_translator_py39 import AI, TacticalTranslator, distance


def row_from(obs):
    row = obs[0]
    return {"ball": [float(x) for x in row["ball"]],
            "score": [int(x) for x in row["score"]],
            "owner_team": int(row["ball_owned_team"]),
            "owner_player": int(row["ball_owned_player"]),
            "left": [[float(x) for x in p] for p in row["left_team"]],
            "right": [[float(x) for x in p] for p in row["right_team"]]}


def run(env, snapshot, policy):
    env.reset()
    env.set_state(snapshot)
    obs = env.unwrapped.observation()
    start = row_from(obs)
    last = start
    coach = TacticalTranslator(policy)
    possession = Counter()
    side_changes = 0
    last_nonfree_team = None
    last_nonfree_owner = None
    last_nonfree_step = None
    transfers = 0
    override_frames = Counter()
    ai_counts = []
    goals = []
    t0 = time.perf_counter()
    done = False
    for step in range(3001):
        actions, meta = coach.actions(obs)
        if int(obs[0]["ball_owned_team"]) == 0:
            assert all(action == AI for action in actions)
        ai_counts.append(sum(action == AI for action in actions))
        assert 0 not in meta["overridden"]
        override_frames.update(meta["overridden"])
        obs, _, done, _ = env.step(actions)
        state = row_from(obs)
        possession[str(state["owner_team"])] += 1
        if state["owner_team"] in (0, 1):
            if last_nonfree_team is not None and last_nonfree_team != state["owner_team"]:
                side_changes += 1
            if (last_nonfree_owner is not None
                    and last_nonfree_team == state["owner_team"]
                    and last_nonfree_owner != state["owner_player"]
                    and step + 1 - last_nonfree_step <= 30):
                transfers += 1
            last_nonfree_team = state["owner_team"]
            last_nonfree_owner = state["owner_player"]
            last_nonfree_step = step + 1
        if state["score"] != last["score"]:
            goals.append({"step": step + 1, "score": state["score"]})
        last = state
        if done:
            break
    seconds = time.perf_counter() - t0
    coach.finish()
    return {"policy": policy, "steps": step + 1, "done": bool(done),
            "seconds": seconds, "steps_per_second": (step + 1) / seconds,
            "start_score": start["score"], "final_score": last["score"],
            "goals": goals, "possession_frames": dict(possession),
            "possession_side_changes": side_changes,
            "same_team_transfer_candidates": transfers,
            "override_frames_by_player": {str(k): v for k, v in override_frames.items()},
            "override_engagements": coach.engagements,
            "minimum_ai_players_per_step": min(ai_counts),
            "mean_ai_players_per_step": sum(ai_counts) / len(ai_counts),
            "left_players_moved": sum(distance(a, b) > 0.01
                                      for a, b in zip(start["left"], last["left"])),
            "right_players_moved": sum(distance(a, b) > 0.01
                                       for a, b in zip(start["right"], last["right"]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--policies", default="normal,press,cover,adaptive")
    args = parser.parse_args()
    env = football_env.create_environment(
        env_name="11_vs_11_hard_stochastic", representation="raw",
        rewards="scoring", number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"})
    try:
        env.reset()
        snapshot = env.get_state()
        results = []
        for policy in args.policies.split(","):
            result = run(env, snapshot, policy)
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    output = Path(args.out)
    output.write_text(json.dumps({"scenario": "11_vs_11_hard_stochastic",
                                  "same_initial_snapshot": True,
                                  "results": results}, indent=2) + "\n")
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
