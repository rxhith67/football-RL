"""Summarize paired GRF rollout outcomes by saved situation, not by frame."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    rows = data["results"]
    horizon = data["horizon"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["policy"])].append(row)
        groups[("all", row["policy"])].append(row)

    summaries = []
    for (category, policy), group in sorted(groups.items()):
        summaries.append({
            "category": category, "policy": policy, "states": len(group),
            "recovered": sum(bool(r["recovered_before_conceding"]) for r in group),
            "mean_censored_recovery_steps": statistics.mean(
                r["time_to_recovery"] if r["time_to_recovery"] is not None
                else horizon + 1 for r in group),
            "goals_conceded": sum(r["goals_conceded"] for r in group),
            "goals_scored": sum(r["goals_scored"] for r in group),
            "mean_opponent_progress": statistics.mean(
                r["opponent_progress_toward_goal"] for r in group),
            "mean_dangerous_entries_proxy": statistics.mean(
                r["dangerous_ball_entries_proxy"] for r in group),
            "mean_goal_side_near_ball": statistics.mean(
                r["mean_shape_while_opponent_possesses"]["goal_side_near_ball"]
                for r in group if r["mean_shape_while_opponent_possesses"]["goal_side_near_ball"] is not None),
            "mean_steps_per_second": statistics.mean(r["steps_per_second"] for r in group),
        })

    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_id"]].append(row)
    best_recovery = {}
    execution_comparisons = []
    for state_id, group in by_state.items():
        normal = next((r for r in group if r["policy"] == "normal"), None)
        normal_by_step = ({item["step"]: item for item in normal["trajectory"]}
                          if normal else {})
        for tactic in group:
            if tactic["policy"] not in ("press", "cover"):
                continue
            first = tactic["trajectory"][0]
            sample = next((item for item in tactic["trajectory"]
                           if item["step"] == 20), None)
            baseline = normal_by_step.get(20)
            selected = first["overridden"]
            if not (sample and baseline and len(selected) == 2
                    and all(player in sample["overridden"]
                            for player in selected)):
                continue
            target = first["targets"]
            def mean_distance(item):
                return statistics.mean(
                    math.hypot(item["left"][p][0] - target[str(p)][0],
                               item["left"][p][1] - target[str(p)][1])
                    for p in selected)
            execution_comparisons.append({
                "state_id": state_id, "policy": tactic["policy"],
                "selected_players": selected,
                "distance_to_initial_targets_at_20": mean_distance(sample),
                "normal_players_distance_to_same_targets_at_20": mean_distance(baseline),
                "tactic_closer_by": mean_distance(baseline) - mean_distance(sample),
            })
        eligible = [r for r in group if r["goals_conceded"] == 0
                    and r["time_to_recovery"] is not None]
        if eligible:
            best_time = min(r["time_to_recovery"] for r in eligible)
            best_recovery[state_id] = [r["policy"] for r in eligible
                                       if r["time_to_recovery"] == best_time]
        else:
            best_recovery[state_id] = []

    issues = {
        "goalkeeper_override_frames": sum(r["goalkeeper_override_frames"] for r in rows),
        "minimum_ai_players_per_step": min(r["minimum_ai_players_per_step"] for r in rows),
        "negative_target_progress_rollouts": [
            {"state": r["state_id"], "policy": r["policy"]}
            for r in rows
            if r["mean_target_directed_progress_per_override_frame"] is not None
            and r["mean_target_directed_progress_per_override_frame"] <= 0],
    }
    result = {"source": args.input, "horizon": horizon,
              "rollouts": len(rows), "summaries": summaries,
              "best_recovery_without_goal_conceded": best_recovery,
              "execution_comparisons": execution_comparisons,
              "execution_checks": issues}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("zone policy n recovery goals_against censored_time progress danger coverage fps")
    for s in summaries:
        print("{category:9s} {policy:8s} {states:2d} {recovered:2d} {goals_conceded:2d} "
              "{mean_censored_recovery_steps:6.1f} {mean_opponent_progress:.3f} "
              "{mean_dangerous_entries_proxy:.2f} {mean_goal_side_near_ball:.2f} "
              "{mean_steps_per_second:.1f}".format(**s))
    print("best by first recovery without conceded goal:", best_recovery)
    print("execution checks:", issues)
    print("first-20-step tactic closer than normal:",
          sum(x["tactic_closer_by"] > 0 for x in execution_comparisons),
          "/", len(execution_comparisons))


if __name__ == "__main__":
    main()
