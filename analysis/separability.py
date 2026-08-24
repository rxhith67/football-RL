"""Archetype separability — a Gate 1 condition (design note 3.2).

Can the K opponent archetypes be told apart from their observable footprint
alone? If not, the belief filter has nothing to infer and **Pillar 1 collapses**,
so this is verified on scripted archetypes long before any RL exists.

The check deliberately runs at the same time as the non-transitivity check in
`analysis.egta`, because both need the same scripted round-robin and both void a
headline contribution if they fail. Finding out in Phase 1 costs a few CPU
minutes; finding out in Phase 3 costs the pillar.

CLI::

    python -m analysis.separability --windows 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from envs.archetypes import N_ARCHETYPES, archetype_table
from pitchsim import controller as ctrl
from pitchsim.engine import BatchedEngine

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "separability.json"
PROGRESS = ROOT / "data" / "separability_progress.json"


def collect_windows(
    windows_per_archetype: int = 60,
    window_minutes: float | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate each archetype and record its observable-statistics vectors.

    Returns (X (n, 8), y (n,)) where each row is one window's `WindowStats`.
    """
    tactics_cfg, physics = ctrl.load_config()
    if window_minutes is None:
        window_minutes = float(tactics_cfg["separability"]["window_minutes"])

    _, _, base = archetype_table()
    rng = np.random.default_rng(seed)

    features, labels = [], []
    first_archetype = 0
    if PROGRESS.exists():
        saved = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if (saved.get("config_fingerprint") == ctrl.config_fingerprint()
                and saved.get("windows_per_archetype") == windows_per_archetype
                and saved.get("window_minutes") == window_minutes
                and saved.get("seed") == seed):
            features = [np.asarray(saved["features"], dtype=np.float64)]
            labels = [np.asarray(saved["labels"], dtype=np.int64)]
            first_archetype = int(saved.get("completed_archetypes", 0))
            if verbose and first_archetype:
                print(f"  resuming after {first_archetype} archetypes", flush=True)

    for k in range(first_archetype, N_ARCHETYPES):
        engine = BatchedEngine(batch=windows_per_archetype, config=physics)

        # The archetype under test is HOME; the opponent is drawn at random so the
        # footprint measured is the archetype's own, not an artefact of one matchup.
        opponents = base[rng.integers(0, N_ARCHETYPES, size=windows_per_archetype)]
        tactics = np.stack(
            [np.tile(base[k], (windows_per_archetype, 1)), opponents], axis=1
        )
        engine.reset(seed=seed * 1000 + k, tactics=tactics)

        elapsed = 0.0
        while elapsed < 90.0:
            engine.run_until((elapsed + window_minutes) * 60.0, stop_on_goal=False)
            stats = engine.window_stats()[:, 0]        # home side only
            features.append(stats)
            labels.append(np.full(windows_per_archetype, k))
            elapsed += window_minutes

        if verbose:
            print(f"  {archetype_table()[1][k]:24} {len(features) * windows_per_archetype:6d} windows",
                  flush=True)
        PROGRESS.write_text(json.dumps({
            "config_fingerprint": ctrl.config_fingerprint(),
            "windows_per_archetype": windows_per_archetype,
            "window_minutes": window_minutes,
            "seed": seed,
            "completed_archetypes": k + 1,
            "features": np.concatenate(features).tolist(),
            "labels": np.concatenate(labels).tolist(),
        }), encoding="utf-8")

    return np.concatenate(features), np.concatenate(labels)


def gaussian_confusion(
    x: np.ndarray, y: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit the per-archetype Gaussians of design note 3.4 and score them.

    This is the *same* model the online belief filter uses, so the accuracy here
    is a direct estimate of how well that filter can possibly do — not a proxy
    from some stronger classifier that the filter could never match.

    Returns (confusion (K, K) row-normalised, overall accuracy, per-class recall).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    holdout = rng.random(n) < 0.3
    train, test = ~holdout, holdout

    _, physics = ctrl.load_config()
    min_var = 1e-4

    means, variances, priors = [], [], []
    for k in range(N_ARCHETYPES):
        rows = x[train & (y == k)]
        means.append(rows.mean(axis=0))
        variances.append(np.maximum(rows.var(axis=0), min_var))
        priors.append(len(rows) / max(train.sum(), 1))

    means = np.stack(means)
    variances = np.stack(variances)
    priors = np.array(priors)

    # Diagonal-Gaussian log-likelihood, exactly as the belief filter computes it.
    diff = x[test][:, None, :] - means[None, :, :]
    log_likelihood = -0.5 * np.sum(
        diff**2 / variances[None] + np.log(2 * np.pi * variances[None]), axis=-1
    )
    predicted = np.argmax(log_likelihood + np.log(priors + 1e-12), axis=1)
    actual = y[test]

    confusion = np.zeros((N_ARCHETYPES, N_ARCHETYPES))
    for a, p in zip(actual, predicted):
        confusion[a, p] += 1
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    accuracy = float((predicted == actual).mean())
    confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    return confusion, accuracy, recall


def run(windows_per_archetype: int = 60, seed: int = 0, verbose: bool = True) -> dict:
    tactics_cfg, _ = ctrl.load_config()
    criteria = tactics_cfg["separability"]
    min_accuracy = float(criteria["min_overall_accuracy"])
    min_recall = float(criteria["min_per_archetype_recall"])

    if verbose:
        print("simulating archetype windows...", flush=True)
    x, y = collect_windows(windows_per_archetype, seed=seed, verbose=verbose)
    confusion, accuracy, recall = gaussian_confusion(x, y, seed=seed)

    _, labels, _ = archetype_table()
    ok = accuracy >= min_accuracy and recall.min() >= min_recall

    if verbose:
        print(f"\nArchetype separability — {len(y)} windows, "
              f"diagonal-Gaussian MAP (the belief filter's own model)")
        print("-" * 78)
        header = "".join(f"{l.split()[0][:9]:>10}" for l in labels)
        corner = "true \\ pred"
        print(f"{corner:>18}{header}   recall")
        for i, label in enumerate(labels):
            row = "".join(f"{confusion[i, j]:10.2f}" for j in range(N_ARCHETYPES))
            print(f"{label[:18]:>18}{row}   {recall[i]:.2f}")
        print("-" * 78)
        print(f"  overall accuracy   {accuracy:.3f}  (need >= {min_accuracy})")
        print(f"  worst-case recall  {recall.min():.3f}  (need >= {min_recall})")
        print(f"  GATE 1 separability: {'PASS' if ok else 'FAIL'}")
        if not ok:
            worst = int(np.argmin(recall))
            print(f"  weakest archetype: {labels[worst]} — its footprint overlaps "
                  f"{labels[int(np.argmax(confusion[worst] - np.eye(N_ARCHETYPES)[worst]))]}")
            print("  Fix by widening the tactic modifiers in config/tactics.yaml, "
                  "NOT by adding features the online filter will not have.")

    result = {
        "config_fingerprint": ctrl.config_fingerprint(),
        "accuracy": accuracy,
        "recall": recall.tolist(),
        "labels": labels,
        "confusion": confusion.tolist(),
        "n_windows": int(len(y)),
        "passed": bool(ok),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # The online filter must be the model that was actually audited above.
    # Previously separability regenerated its own classifier but left the
    # committed belief_model.json stale, so training and the reported gate used
    # different distributions.
    from envs.belief import BeliefFilter
    means = np.stack([x[y == k].mean(axis=0) for k in range(N_ARCHETYPES)])
    variances = np.stack([x[y == k].var(axis=0) for k in range(N_ARCHETYPES)])
    BeliefFilter(means, variances).save()
    return result


def separability_report() -> tuple[bool, str]:
    """Gate 1 hook for scripts/check_gate.py."""
    if not RESULTS.exists():
        return False, "not yet run: python -m analysis.separability"
    blob = json.loads(RESULTS.read_text(encoding="utf-8"))
    if blob.get("config_fingerprint") != ctrl.config_fingerprint():
        return False, "stale evidence: rerun python -m analysis.separability"
    return blob["passed"], (
        f"accuracy {blob['accuracy']:.3f}, worst recall {min(blob['recall']):.3f}, "
        f"{blob['n_windows']} windows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Archetype separability (Gate 1)")
    parser.add_argument("--windows", type=int, default=60,
                        help="matches simulated per archetype")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(windows_per_archetype=args.windows, seed=args.seed)


if __name__ == "__main__":
    main()
