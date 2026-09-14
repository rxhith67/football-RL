"""Cross-check published claims against raw experiment evidence (stdlib only)."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    source = root / "data/line_coach"
    protocol = json.loads((source / "protocol.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    evaluation = json.loads((source / "evaluation.json").read_text())
    assert summary["complete"]
    assert len(evaluation) == summary["evaluation_episodes"]
    assert len(evaluation) == (len(protocol["training_seeds"]) + 4) * protocol["eval_episodes_per_policy"]
    keys = [(r["policy"], r["model_seed"], r["episode"]) for r in evaluation]
    assert len(keys) == len(set(keys)), "Duplicate evaluation episode"
    for filename, expected in protocol["sources"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected, f"Source changed: {filename}"
    for row in evaluation:
        assert row["success"]
        assert row["native_seed"] in protocol["evaluation_seeds"]
        assert row["goal_difference"] == row["score_left"] - row["score_right"]
        assert sum(row["action_counts"]) == math.ceil(protocol["horizon"] / 20)
    for training in summary["training"]:
        assert training["complete"] and training["parameters_finite"] and training["parameters_changed"]
        assert training["timesteps"] == protocol["requested_timesteps_per_seed"]
        with zipfile.ZipFile(source / f"ppo_seed_{training['seed']}.zip") as checkpoint:
            assert checkpoint.testzip() is None
            assert "policy.pth" in checkpoint.namelist()
    for row in summary["results"]:
        group = [r for r in evaluation if r["policy"] == row["policy"] and r["model_seed"] == row["seed"]]
        assert len(group) == row["episodes"]
        assert math.isclose(statistics.mean(r["goal_difference"] for r in group), row["mean_goal_difference"], abs_tol=1e-12)
        assert row["wins"] + row["draws"] + row["losses"] == row["episodes"]
        if row["replay"]:
            original = source / row["replay"]
            published = root / "line_coach_demo/data" / row["replay"]
            assert original.read_bytes() == published.read_bytes()
            replay = json.loads(original.read_text())
            assert replay["frames"][-1]["step"] == protocol["horizon"]
            assert replay["frames"][0]["step"] == 0
            assert len(replay["decisions"]) == math.ceil(protocol["horizon"] / 20)
            previous = -1
            for frame in replay["frames"]:
                assert frame["step"] > previous
                previous = frame["step"]
                assert len(frame["players"]) == 22
                assert frame["action"] in (0, 1, 2)
                assert all(math.isfinite(v) for p in frame["players"] for v in p)
            first_result = next(r for r in group if r["episode"] == 0)
            assert replay["score"] == [first_result["score_left"], first_result["score_right"]]
    assert (source / "summary.json").read_bytes() == (root / "line_coach_demo/data/summary.json").read_bytes()
    archive_path = root / "artifacts/GAFFER_Line_Coach_Submission.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "line_coach_demo/index.html" in names
        assert not any(n.startswith("tmp/") or n.endswith(".env") or "/native-logs/" in n for n in names)
    return {"success": True, "training_seeds": protocol["training_seeds"],
            "evaluation_episodes": len(evaluation), "replays": len(summary["results"]),
            "source_hashes_match": True, "published_data_matches_raw": True,
            "archive_integrity": True, "archive_bytes": archive_path.stat().st_size}


if __name__ == "__main__":
    result = verify()
    (ROOT / "data/line_coach_delivery_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
