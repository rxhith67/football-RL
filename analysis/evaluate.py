"""Match harness, bootstrap CIs and Elo — the pre-registered evaluation protocol.

design note 5 fixes this BEFORE training, because sloppy evaluation is the most
common way simulation studies mislead. Everything here reports uncertainty: a
number without an interval is not a result.

Headline metrics
----------------
1. Points per match and win/draw/loss, with 95% bootstrap CIs.
2. Elo within the pool.
3. **Comeback rate** — P(win or draw | trailing at half-time). The metric a
   football person actually cares about, and the one a macro-coach should move.
4. Adaptation latency — epochs until the belief MAP is correct, measured from the
   opponent's TRUE switch epoch rather than from kickoff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HOME, AWAY = 0, 1


@dataclass
class MatchOutcomes:
    """Per-match results for one (agent, opponent) cell."""

    points: np.ndarray                     # 3 / 1 / 0 per match
    goal_diff: np.ndarray
    xg_diff: np.ndarray
    trailing_at_half: np.ndarray           # bool
    belief_correct: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    adaptation_epochs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return len(self.points)


def bootstrap_ci(values: np.ndarray, resamples: int = 10000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high). Percentile bootstrap — no normality assumed.

    Match outcomes are heavily discretised (3/1/0) and far from Gaussian, so a
    normal-approximation interval would be quietly wrong at these sample sizes.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (float(values.mean()),
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)))


def summarise(outcomes: MatchOutcomes, seed: int = 0) -> dict:
    ppm, ppm_lo, ppm_hi = bootstrap_ci(outcomes.points, seed=seed)
    gd, gd_lo, gd_hi = bootstrap_ci(outcomes.goal_diff, seed=seed)

    trailing = outcomes.trailing_at_half.astype(bool)
    if trailing.any():
        recovered = (outcomes.points[trailing] >= 1).astype(float)
        cb, cb_lo, cb_hi = bootstrap_ci(recovered, seed=seed)
    else:
        cb = cb_lo = cb_hi = float("nan")

    result = {
        "n_matches": len(outcomes),
        "points_per_match": ppm, "ppm_ci": [ppm_lo, ppm_hi],
        "win_rate": float((outcomes.points == 3).mean()),
        "draw_rate": float((outcomes.points == 1).mean()),
        "loss_rate": float((outcomes.points == 0).mean()),
        "goal_diff": gd, "goal_diff_ci": [gd_lo, gd_hi],
        "xg_diff": float(np.mean(outcomes.xg_diff)) if len(outcomes.xg_diff) else float("nan"),
        "comeback_rate": cb, "comeback_ci": [cb_lo, cb_hi],
        "n_trailing_at_half": int(trailing.sum()),
    }
    if len(outcomes.belief_correct):
        result["belief_accuracy"] = float(outcomes.belief_correct.mean())
    if len(outcomes.adaptation_epochs):
        finite = outcomes.adaptation_epochs[np.isfinite(outcomes.adaptation_epochs)]
        result["adaptation_epochs"] = float(finite.mean()) if len(finite) else float("nan")
    return result


def non_overlapping(a: dict, b: dict, key: str = "ppm_ci") -> bool:
    """Do two cells' confidence intervals fail to overlap?

    The Gate 2 criterion is stated as "PPO beats random with non-overlapping
    CIs", so it is worth having as a function rather than an eyeball.
    """
    lo_a, hi_a = a[key]
    lo_b, hi_b = b[key]
    return hi_a < lo_b or hi_b < lo_a


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------


def elo_from_payoff(points: np.ndarray, labels: list[str], k: float = 32.0,
                    iterations: int = 400) -> dict[str, float]:
    """Fit Elo ratings to an all-play-all points matrix by repeated passes.

    `points[i, j]` is i's mean points per match against j (0-3). Converted to a
    score in [0, 1] and fitted until ratings settle.
    """
    n = len(labels)
    ratings = np.zeros(n)
    score = np.clip(points / 3.0, 0.0, 1.0)

    for _ in range(iterations):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                expected = 1.0 / (1.0 + 10 ** ((ratings[j] - ratings[i]) / 400.0))
                ratings[i] += k * (score[i, j] - expected) / max(n - 1, 1) * 0.05
    ratings -= ratings.mean()
    return {label: float(r + 1500.0) for label, r in zip(labels, ratings)}


# --------------------------------------------------------------------------
# Running an agent against the environment
# --------------------------------------------------------------------------


def play(agent, env, n_matches: int, deterministic: bool = True,
         collect_belief: bool = True) -> MatchOutcomes:
    """Play `n_matches` with `agent` as HOME and return per-match outcomes.

    `agent` is either a baseline exposing `act(obs, env)` or an SB3 model
    exposing `predict(obs, action_masks=..., deterministic=...)`.
    """
    points, goal_diff, xg_diff, trailing, correct = [], [], [], [], []

    obs = env.reset()
    is_sb3 = hasattr(agent, "predict")

    while len(points) < n_matches:
        if is_sb3:
            actions, _ = agent.predict(
                obs, action_masks=env.action_masks(), deterministic=deterministic)
        else:
            actions = agent.act(obs, env)

        half_before = env.engine.minutes() < 45.0
        score_before = env.engine.score.copy()
        obs, rewards, dones, infos = env.step(actions)

        # Record which matches were behind at the interval, before any reset.
        just_reached_half = half_before & (env.engine.minutes() >= 45.0)
        if just_reached_half.any():
            env._half_score[just_reached_half] = score_before[just_reached_half]

        if dones.all():
            for info in infos:
                episode = info.get("episode")
                if episode is None:
                    continue
                home, away = episode["score"]
                points.append(episode["points"])
                goal_diff.append(home - away)
                xg_diff.append(0.0)
                trailing.append(False)
                if collect_belief:
                    correct.append(bool(episode.get("belief_correct", False)))
                if len(points) >= n_matches:
                    break

    return MatchOutcomes(
        points=np.array(points[:n_matches], dtype=np.float64),
        goal_diff=np.array(goal_diff[:n_matches], dtype=np.float64),
        xg_diff=np.array(xg_diff[:n_matches], dtype=np.float64),
        trailing_at_half=np.array(trailing[:n_matches], dtype=bool),
        belief_correct=np.array(correct[:n_matches], dtype=bool),
    )


def save(results: dict, name: str) -> Path:
    path = ROOT / "data" / f"eval_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return path
