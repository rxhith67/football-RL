"""Round-robin Elo across self-play generations (Gate 2, Stage 2 PFSP league).

Each "generation" is a Stage-2 checkpoint trained from the common Stage-1
parent against a progressively larger PFSP pool (train.py enforces that every
Stage-2 run's parent is a Stage-1 checkpoint, so generations are siblings that
face a growing opponent pool, not a linear fine-tuning chain). This module
plays every generation against every other and fits Elo via
`analysis.evaluate.elo_from_payoff`, so the question "does Elo rise across
generations" has a number behind it rather than an impression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.smdp_ppo import SMDPMaskablePPO
from analysis import evaluate as ev
from envs.coach_env import VecCoachEnv

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"


def _load(tag: str) -> SMDPMaskablePPO:
    return SMDPMaskablePPO.load(CHECKPOINTS / tag, device="cpu")


def round_robin(tags: list[str], n_envs: int = 16, n_matches: int = 20,
                seed: int = 0) -> dict:
    """Play every ordered pair in `tags` as home vs away; fit Elo from the result."""
    models = {tag: _load(tag) for tag in tags}
    n = len(tags)
    points = np.full((n, n), 1.5)  # diagonal placeholder, unused by elo_from_payoff

    for i, home_tag in enumerate(tags):
        for j, away_tag in enumerate(tags):
            if i == j:
                continue
            env = VecCoachEnv(n_envs=n_envs, belief_mode="bayesian",
                              opponent="policy", seed=seed)

            def opponent_policy(obs, _m=models[away_tag]):
                actions, _ = _m.predict(obs, action_masks=env.action_masks(),
                                        deterministic=True)
                return actions

            env.opponent_policy = opponent_policy
            outcomes = ev.play(models[home_tag], env, n_matches, collect_belief=False)
            points[i, j] = float(outcomes.points.mean())

    ratings = ev.elo_from_payoff(points, tags)
    return {
        "tags": tags,
        "n_matches_per_pairing": n_matches,
        "points_matrix": points.tolist(),
        "ratings": ratings,
    }


def generations_report(path: Path | None = None) -> tuple[bool, str]:
    """Gate 2: does Elo rise across successive self-play generations?"""
    path = path or (ROOT / "data" / "self_play_elo.json")
    if not path.exists():
        return False, "no round-robin evidence; run: python -m analysis.self_play_elo"
    blob = json.loads(path.read_text(encoding="utf-8"))
    tags, ratings = blob["tags"], blob["ratings"]
    if len(tags) < 3:
        return False, f"only {len(tags)} generation(s) evaluated; need >= 3 to show a trend"

    series = [ratings[tag] for tag in tags]
    rising = all(b >= a - 1e-6 for a, b in zip(series, series[1:])) and series[-1] > series[0]
    detail = "; ".join(f"{tag}={r:.0f}" for tag, r in zip(tags, series))
    return rising, detail


def main() -> None:
    import sys

    tags = sys.argv[1:] or [
        "stage1_bayesian_smdp_seed0",
        "stage2_bayesian_smdp_seed1",
        "stage2_bayesian_smdp_seed2",
    ]
    result = round_robin(tags)
    print(json.dumps(result["ratings"], indent=2))
    (ROOT / "data" / "self_play_elo.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
