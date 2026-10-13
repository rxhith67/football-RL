"""Validate and summarize the fixed minimal-intervention GRF investigation."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def summary(rows):
    return {
        "states": len(rows),
        "recoveries": sum(r["time_to_recovery"] is not None for r in rows),
        "mean_censored_recovery_steps": mean(
            r["time_to_recovery"] or 201 for r in rows),
        "mean_opponent_progress": mean(
            r["opponent_progress_toward_goal"] for r in rows),
        "goals_conceded": sum(r["goals_conceded"] for r in rows),
        "mean_dangerous_ball_entries_proxy": mean(
            r["dangerous_ball_entries_proxy"] for r in rows),
    }


def check_results(data, heldout=False):
    by_state = defaultdict(list)
    for row in data["results"]:
        by_state[row["state_id"]].append(row)
        assert row["goalkeeper_override_frames"] == 0
        assert row["minimum_ai_players_per_step"] >= 9
        if "_1_" in row["policy"]:
            assert row["minimum_ai_players_per_step"] >= 10
            assert len(row["override_engagements"]) <= 1
        if row["policy"].endswith("_12"):
            assert len(row["override_engagements"]) <= 1
            assert all(e["duration_steps"] <= 12
                       for e in row["override_engagements"])
        if row["policy"].endswith("_36"):
            assert len(row["override_engagements"]) <= 1
            assert all(e["duration_steps"] <= 36
                       for e in row["override_engagements"])
    assert len(by_state) == 12
    assert Counter(rows[0]["category"] for rows in by_state.values()) == {
        "midfield": 4, "penalty": 4, "wide": 4}
    if heldout:
        for rows in by_state.values():
            normal = [r for r in rows if r["policy"] == "normal"]
            assert len(normal) == 2
            assert normal[0]["trajectory"] == normal[1]["trajectory"]
            assert sorted(r["policy"] for r in rows) == [
                "cover", "normal", "normal", "press", "press_2_12"]
    else:
        assert all(len(rows) == 11 for rows in by_state.values())
    return by_state


def group_summaries(by_state):
    groups = defaultdict(list)
    for rows in by_state.values():
        seen = set()
        for row in rows:
            if row["policy"] in seen:
                continue
            seen.add(row["policy"])
            groups[("all", row["policy"])].append(row)
            groups[(row["category"], row["policy"])].append(row)
    return [{"category": category, "policy": policy, **summary(rows)}
            for (category, policy), rows in sorted(groups.items())]


def paired_counts(by_state, policy, category="all"):
    better = worse = tied = 0
    for rows in by_state.values():
        if category != "all" and rows[0]["category"] != category:
            continue
        baseline = next(r for r in rows if r["policy"] == "normal")
        test = next(r for r in rows if r["policy"] == policy)
        a, b = baseline["time_to_recovery"] or 201, test["time_to_recovery"] or 201
        if b < a:
            better += 1
        elif b > a:
            worse += 1
        else:
            tied += 1
    return {"faster": better, "slower": worse, "tied": tied}


def main():
    train = load("minimal_intervention_12states.json")
    heldout = load("minimal_intervention_holdout.json")
    train_states = ROOT / "tactical_states_independent"
    heldout_states = ROOT / "tactical_states_holdout_minimal"
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in train_states.glob("*.state")}
    heldout_hashes = {hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in heldout_states.glob("*.state")}
    assert len(train_hashes) == len(heldout_hashes) == 12
    assert not train_hashes.intersection(heldout_hashes)
    a = check_results(train)
    b = check_results(heldout, heldout=True)
    output = {
        "training": group_summaries(a),
        "heldout": group_summaries(b),
        "training_paired_short_press": paired_counts(a, "press_2_12"),
        "heldout_paired_short_press": paired_counts(b, "press_2_12"),
        "training_wide_paired_short_press": paired_counts(
            a, "press_2_12", "wide"),
        "heldout_wide_paired_short_press": paired_counts(
            b, "press_2_12", "wide"),
        "state_hash_overlap": 0,
        "heldout_exact_normal_repeats": 12,
    }
    path = ROOT / "minimal_intervention_summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print("PASS: 12 original and 12 disjoint held-out states; "
          "all 12 held-out normal repeats exact; limits and goalkeeper checks pass")
    print("training wide short press", output["training_wide_paired_short_press"])
    print("heldout wide short press", output["heldout_wide_paired_short_press"])
    print("saved", path)


if __name__ == "__main__":
    main()
