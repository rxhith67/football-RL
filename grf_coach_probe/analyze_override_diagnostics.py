"""State-level mechanism checks from the original 48 paired GRF rollouts."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = (("midfield_02", "press"), ("penalty_02", "cover"),
         ("penalty_01", "cover"), ("wide_00", "press"))


def mean(values):
    return sum(values) / len(values) if values else None


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    rows = json.loads((ROOT / "tactical_final_comparison.json").read_text())["results"]
    lookup = {(row["state_id"], row["policy"]): row for row in rows}
    output = []
    for state_id, policy in CASES:
        base, test = lookup[state_id, "normal"], lookup[state_id, policy]
        engagement = test["override_engagements"][0]
        selected = engagement["players"]
        others = [i for i in range(1, 11) if i not in selected]
        base_trace = {r["step"]: r for r in base["trajectory"]}
        test_trace = {r["step"]: r for r in test["trajectory"]}
        first = base_trace[1]
        nearest_others = sorted(
            others, key=lambda i: distance(first["left"][i], first["ball"]))[:2]
        steps = [s for s in (1, 5, 10, 15, 20, 30)
                 if s in base_trace and s in test_trace]
        comparisons = []
        for step in steps:
            a, b = base_trace[step], test_trace[step]
            comparisons.append({
                "step": step,
                "selected_mean_divergence": mean([
                    distance(a["left"][i], b["left"][i]) for i in selected]),
                "other_outfield_mean_divergence": mean([
                    distance(a["left"][i], b["left"][i]) for i in others]),
                "nearest_other_two_mean_divergence": mean([
                    distance(a["left"][i], b["left"][i])
                    for i in nearest_others]),
                "other_outfield_above_0_02": sum(
                    distance(a["left"][i], b["left"][i]) > 0.02
                    for i in others),
                "ball_divergence": distance(a["ball"], b["ball"]),
                "selected_goal_side_normal": sum(
                    a["left"][i][0] < a["ball"][0] for i in selected),
                "selected_goal_side_intervention": sum(
                    b["left"][i][0] < b["ball"][0] for i in selected),
                "selected_carrier_distance_normal": (
                    mean([distance(a["left"][i], a["right"][a["owner_player"]])
                          for i in selected]) if a["owner_team"] == 1 else None),
                "selected_carrier_distance_intervention": (
                    mean([distance(b["left"][i], b["right"][b["owner_player"]])
                          for i in selected]) if b["owner_team"] == 1 else None),
                "normal_owner": a["owner_team"],
                "intervention_owner": b["owner_team"],
            })
        output.append({
            "state_id": state_id, "intervention": policy,
            "selected_players": selected,
            "nearest_nonoverridden_players": nearest_others,
            "first_engagement_duration_steps": engagement["duration_steps"],
            "normal_recovery": base["time_to_recovery"],
            "intervention_recovery": test["time_to_recovery"],
            "normal_progress": base["opponent_progress_toward_goal"],
            "intervention_progress": test["opponent_progress_toward_goal"],
            "comparisons": comparisons,
        })
    path = ROOT / "override_mechanism_cases.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    for case in output:
        print(case["state_id"], case["intervention"], "players",
              case["selected_players"], "recovery",
              case["normal_recovery"], case["intervention_recovery"])
        for row in case["comparisons"]:
            if row["step"] in (5, 10, 20):
                print(" ", row["step"], "selected divergence",
                      round(row["selected_mean_divergence"], 3),
                      "other divergence",
                      round(row["other_outfield_mean_divergence"], 3),
                      "nearby divergence",
                      round(row["nearest_other_two_mean_divergence"], 3),
                      "ball divergence", round(row["ball_divergence"], 3),
                      "goal-side", row["selected_goal_side_normal"],
                      row["selected_goal_side_intervention"])
    print("saved", path)


if __name__ == "__main__":
    main()
