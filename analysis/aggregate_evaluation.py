"""Aggregate pre-registered evaluation cells at the seed level."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.evaluate import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent


def aggregate_baselines() -> dict:
    index = json.loads((ROOT / "data" / "eval_baselines.json").read_text(encoding="utf-8"))
    runs = index.get("runs", {})
    names = sorted({name for run in runs.values() for name in run})
    aggregate = {}
    for name in names:
        values = np.array([
            run[name]["points_per_match"] for run in runs.values() if name in run
        ], dtype=np.float64)
        mean, low, high = bootstrap_ci(values, seed=1701)
        aggregate[name] = {
            "n_seeds": int(len(values)),
            "matches_per_seed": int(index["_meta"]["target_matches_per_cell"]),
            "points_per_match": mean,
            "seed_bootstrap_ppm_ci": [low, high],
            "seed_ppm": values.tolist(),
        }
    index["aggregate"] = aggregate
    return index


def ppo_vs_random_report(stage_tag: str = "stage1_bayesian_smdp_seed0") -> tuple[bool, str]:
    """Gate 2: does the trained coach beat random-tactic with non-overlapping 95% CIs?"""
    eval_path = ROOT / "data" / f"eval_{stage_tag}.json"
    if not eval_path.exists():
        return False, f"no eval for {stage_tag}; run: python train.py --stage 1 --init <stage0 checkpoint>"

    ppo = json.loads(eval_path.read_text(encoding="utf-8"))[stage_tag]
    ppo_lo, ppo_hi = ppo["ppm_ci"]

    baselines_path = ROOT / "data" / "eval_baselines.json"
    if not baselines_path.exists():
        return False, "no baseline evidence; run: python train.py --baselines"
    index = json.loads(baselines_path.read_text(encoding="utf-8"))
    random_ppm = [run["random"]["points_per_match"] for run in index.get("runs", {}).values()
                  if "random" in run]
    if not random_ppm:
        return False, "no 'random' baseline runs recorded"
    # Take the baseline's HIGHEST upper bound across seeds: the check must clear
    # the strongest showing random put up, not the weakest one, or a marginal
    # result passes on the luck of which seed happened to look worst.
    random_lo, random_hi = max(
        (run["random"]["ppm_ci"] for run in index["runs"].values() if "random" in run),
        key=lambda ci: ci[1],
    )

    non_overlapping = ppo_lo > random_hi
    detail = (f"PPO ppm {ppo['points_per_match']:.3f} [{ppo_lo:.3f}, {ppo_hi:.3f}] vs "
              f"random ppm {np.mean(random_ppm):.3f} [{random_lo:.3f}, {random_hi:.3f}]")
    return non_overlapping, detail


def main() -> None:
    result = aggregate_baselines()
    path = ROOT / "data" / "eval_baselines_aggregate.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result.get("aggregate", {}), indent=2))


if __name__ == "__main__":
    main()
