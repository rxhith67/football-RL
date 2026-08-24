"""Auditable Gate-2 belief adaptation measurement.

The clock starts at a realised opponent tactic change, not at kickoff and not
whenever a switching rule remains active. Progress is saved after every batch
so the pre-registered evaluation survives short local process limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from agents.baselines import RandomCoach
from analysis.evaluate import play
from envs.coach_env import VecCoachEnv
from pitchsim.controller import config_fingerprint

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "belief_adaptation.json"
PROGRESS = ROOT / "data" / "belief_adaptation_progress.json"


def evidence_fingerprint() -> str:
    """Fingerprint the simulator plus every module that defines this audit."""
    digest = hashlib.sha256(config_fingerprint().encode("ascii"))
    for relative in (
        "envs/archetypes.py", "envs/belief.py", "envs/coach_env.py",
        "analysis/evaluate.py", "analysis/belief_adaptation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:16]


def _criteria() -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
    return cfg["evaluation"]


def run(matches: int | None = None, batch: int = 48, seed: int = 0,
        verbose: bool = True) -> dict:
    cfg = _criteria()
    target = int(matches or cfg["belief_adaptation_matches"])
    fingerprint = evidence_fingerprint()
    state = {
        "evidence_fingerprint": fingerprint,
        "target_matches": target,
        "seed": seed,
        "completed_matches": 0,
        "n_switches": 0,
        "adaptation_epochs": [],
    }
    if PROGRESS.exists():
        candidate = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if (candidate.get("evidence_fingerprint") == fingerprint
                and candidate.get("target_matches") == target
                and candidate.get("seed") == seed):
            state = candidate

    while state["completed_matches"] < target:
        n = min(batch, target - int(state["completed_matches"]))
        batch_index = int(state["completed_matches"]) // batch
        batch_seed = seed + 1009 * batch_index
        env = VecCoachEnv(n_envs=n, belief_mode="bayesian",
                          opponent="archetype", seed=batch_seed)
        outcomes = play(RandomCoach(batch_seed), env, n, collect_belief=True)
        state["completed_matches"] += n
        state["n_switches"] += int(outcomes.n_switches)
        state["adaptation_epochs"].extend(outcomes.adaptation_epochs.tolist())
        PROGRESS.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if verbose:
            print(f"belief adaptation: {state['completed_matches']}/{target} matches", flush=True)

    latencies = np.asarray(state["adaptation_epochs"], dtype=np.float64)
    n_switches = int(state["n_switches"])
    mean_latency = float(latencies.mean()) if len(latencies) else float("nan")
    success_rate = float(len(latencies) / n_switches) if n_switches else 0.0
    max_epochs = float(cfg["belief_adaptation_max_epochs"])
    min_success = float(cfg["belief_adaptation_min_success_rate"])
    min_switches = int(cfg["belief_adaptation_min_switches"])
    passed = (n_switches >= min_switches
              and success_rate >= min_success
              and mean_latency <= max_epochs)
    result = {
        "evidence_fingerprint": fingerprint,
        "n_matches": int(state["completed_matches"]),
        "n_switches": n_switches,
        "n_adapted": int(len(latencies)),
        "adaptation_success_rate": success_rate,
        "mean_adaptation_epochs": mean_latency,
        "median_adaptation_epochs": (
            float(np.median(latencies)) if len(latencies) else float("nan")),
        "p90_adaptation_epochs": (
            float(np.quantile(latencies, 0.90)) if len(latencies) else float("nan")),
        "criteria": {
            "max_mean_epochs": max_epochs,
            "min_success_rate": min_success,
            "min_switches": min_switches,
        },
        "passed": bool(passed),
    }
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(result, indent=2))
    return result


def adaptation_report() -> tuple[bool, str]:
    if not RESULTS.exists():
        return False, "not yet run: python -m analysis.belief_adaptation"
    blob = json.loads(RESULTS.read_text(encoding="utf-8"))
    if blob.get("evidence_fingerprint") != evidence_fingerprint():
        return False, "stale evidence: rerun python -m analysis.belief_adaptation"
    return bool(blob.get("passed")), (
        f"mean {blob['mean_adaptation_epochs']:.2f} epochs, "
        f"{blob['adaptation_success_rate']:.0%} recovered, "
        f"{blob['n_switches']} true switches"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-2 belief adaptation audit")
    parser.add_argument("--matches", type=int)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.matches, args.batch, args.seed)


if __name__ == "__main__":
    main()
