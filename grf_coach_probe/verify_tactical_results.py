"""Integrity checks for the saved tactical feasibility experiment."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def main():
    data = load("tactical_final_comparison.json")
    rows = data["results"]
    assert len(rows) == 48
    assert data["horizon"] == 200
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_id"]].append(row["policy"])
        assert row["goalkeeper_override_frames"] == 0
        assert row["minimum_ai_players_per_step"] >= 9
        if row["policy"] == "normal":
            assert not row["override_frames_by_player"]
        else:
            assert row["mean_target_directed_progress_per_override_frame"] > 0
            assert all(0 < int(player) <= 10
                       for player in row["override_frames_by_player"])
    assert len(by_state) == 12
    assert all(sorted(policies) == ["adaptive", "cover", "normal", "press"]
               for policies in by_state.values())

    manifest = load("tactical_states_independent/manifest.json")
    assert len(manifest["states"]) == 12
    counts = Counter((s["episode"], s["category"]) for s in manifest["states"])
    assert set(counts.values()) == {1}
    assert len({s["episode"] for s in manifest["states"]}) == 4
    inspection = load("tactical_states_inspection.json")
    assert all(s["game_mode"] == 0 and s["owner_team"] == 1
               and s["carrier_ball_distance"] < 0.025 for s in inspection)

    pairing = load("pairing_repeat_final_code.json")["results"]
    assert pairing[0]["policy"] == pairing[2]["policy"] == "normal"
    assert pairing[0]["trajectory"] == pairing[2]["trajectory"]

    matches = load("tactical_final_matches.json")["results"]
    assert len(matches) == 4
    assert all(m["done"] and m["steps"] == 3001
               and m["minimum_ai_players_per_step"] >= 9 for m in matches)
    print("PASS: 48 paired rollouts, 12 live-play snapshots from 4 matches, "
          "no goalkeeper overrides, >=9 built-in AI players, exact repeat, "
          "and 4 complete matches")


if __name__ == "__main__":
    main()
