"""Offline checks for result reporting and package boundaries."""
import importlib.util
from pathlib import Path


def test_report_does_not_turn_a_negative_result_into_success():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_line_coach.py"
    spec = importlib.util.spec_from_file_location("publish_line_coach", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"name": name, "policy": policy, "episodes": 12, "wins": 0, "draws": 12,
             "losses": 0, "mean_goal_difference": difference, "goal_difference_ci95": [-1, 1],
             "points_per_episode": 1, "actions": [720, 0, 0]}
            for name, policy, difference in [("PPO seed 31", "ppo", -1), ("Normal HELIOS", "normal", 0)]]
    report = module.report_text({"results": rows, "training": [{"seed": 31, "timesteps": 2304,
            "episodes": [], "seconds": 120}], "selected_demo_seed": 31, "horizon": 1200,
            "evaluation_episodes": 24, "limitations": ["Limited budget"]})
    assert "did not outperform" in report
    assert "not regulation football matches" in report
    assert "Limited budget" in report
    assert "did not demonstrate adaptive switching" in report


def test_demo_uses_no_external_web_assets():
    root = Path(__file__).resolve().parents[1] / "line_coach_demo"
    for filename in ("index.html", "style.css", "app.js"):
        content = (root / filename).read_text(encoding="utf-8")
        assert "https://" not in content
        assert "http://" not in content
