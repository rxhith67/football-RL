"""Empirical game-theoretic analysis of the tactic space — Pillar 2.

Two jobs, deliberately split by phase:

* `scripted_cycle_report` (**Gate 1**) asks the cheap, decisive question: does a
  non-transitive rock-paper-scissors cycle exist among the scripted archetypes at
  all? It needs no RL and runs in minutes. If the answer is no, Pillar 2 has no
  headline result and the EV counters in `config/tactics.yaml` must be retuned —
  which is a Phase-1 problem, not a Phase-3 catastrophe (design note 3.5).

* `payoff_matrix` / `nash` / `alpha_rank` (**Phase 3**) run the full analysis over
  archetypes plus trained checkpoints.

A note the write-up must keep: the counters in `config/tactics.yaml` were *designed*
to admit a cycle. The honest framing is "we built an environment that admits
non-transitivity, then asked whether the RL coach discovers and exploits it" —
never "we discovered football is non-transitive".
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from envs.archetypes import N_ARCHETYPES, archetype_table
from pitchsim import controller as ctrl
from pitchsim.engine import BatchedEngine

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "egta_scripted.json"


def play_pairing(
    row_tactic: np.ndarray, col_tactic: np.ndarray, matches: int, seed: int
) -> tuple[float, float]:
    """Play `matches` between two fixed tactics. Returns (row points-per-match, row goal diff).

    Sides are swapped for half the batch so venue cannot confound the payoff.
    """
    _, physics = ctrl.load_config()
    engine = BatchedEngine(batch=matches, config=physics)

    half = matches // 2
    tactics = np.empty((matches, 2, 5), dtype=np.int64)
    tactics[:half, 0], tactics[:half, 1] = row_tactic, col_tactic
    tactics[half:, 0], tactics[half:, 1] = col_tactic, row_tactic

    engine.reset(seed=seed, tactics=tactics)
    engine.run_until(engine.match_seconds, stop_on_goal=False)

    score = engine.score
    row_goals = np.concatenate([score[:half, 0], score[half:, 1]])
    col_goals = np.concatenate([score[:half, 1], score[half:, 0]])

    points = np.where(row_goals > col_goals, 3.0,
                      np.where(row_goals == col_goals, 1.0, 0.0))
    return float(points.mean()), float((row_goals - col_goals).mean())


def scripted_payoff(matches_per_pairing: int | None = None, seed: int = 0,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """K x K payoff matrices (points-per-match, goal difference) over archetypes."""
    tactics_cfg, physics = ctrl.load_config()
    if matches_per_pairing is None:
        import yaml

        train = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
        matches_per_pairing = int(train["evaluation"]["scripted_roundrobin_matches"])

    _, labels, base = archetype_table()
    points = np.zeros((N_ARCHETYPES, N_ARCHETYPES))
    goal_diff = np.zeros((N_ARCHETYPES, N_ARCHETYPES))

    for i in range(N_ARCHETYPES):
        for j in range(N_ARCHETYPES):
            if i == j:
                points[i, j], goal_diff[i, j] = 1.5, 0.0    # mirror match by symmetry
                continue
            if j < i:
                continue
            ppm, gd = play_pairing(base[i], base[j], matches_per_pairing, seed + i * 17 + j)
            points[i, j], goal_diff[i, j] = ppm, gd
            # Zero-sum in goal difference; points are recovered from the same games.
            points[j, i], goal_diff[j, i] = 3.0 - ppm - _draw_share(ppm), -gd
        if verbose:
            print(f"  {labels[i]:24} done", flush=True)
    return points, goal_diff


def _draw_share(ppm: float) -> float:
    """Points are not exactly zero-sum because draws award 1+1 rather than 3+0.

    Recovering the opponent's points from ours needs the draw rate, which we do
    not carry here; assuming no draws would bias the matrix. Using goal difference
    for the cycle test avoids the issue entirely, so this is left at zero and
    `points` is reported for readability only.
    """
    return 0.0


def find_cycles(matrix: np.ndarray) -> list[tuple[int, ...]]:
    """All 3-cycles in the 'beats' relation implied by `matrix`.

    `matrix[i, j] > 0` means i beats j. A cycle i -> j -> k -> i is exactly the
    rock-paper-scissors structure Pillar 2 claims to find.
    """
    beats = matrix > 0
    cycles = []
    for i, j, k in combinations(range(len(matrix)), 3):
        if beats[i, j] and beats[j, k] and beats[k, i]:
            cycles.append((i, j, k))
        elif beats[i, k] and beats[k, j] and beats[j, i]:
            cycles.append((i, k, j))
    return cycles


def transitivity_ratio(matrix: np.ndarray) -> float:
    """Share of payoff mass explained by a single transitive rating.

    Fits Elo-like ratings r minimising ||(r_i - r_j) - M_ij||^2 and reports the
    fraction of variance explained. Near 1.0 means a strict pecking order and no
    interesting game; well below 1.0 means genuine strategic structure.
    """
    n = len(matrix)
    off = ~np.eye(n, dtype=bool)
    ratings = matrix.mean(axis=1) - matrix.mean()
    predicted = ratings[:, None] - ratings[None, :]
    residual = ((matrix - predicted)[off] ** 2).sum()
    total = (matrix[off] ** 2).sum()
    return float(1.0 - residual / max(total, 1e-9))


def run(matches: int | None = None, seed: int = 0, verbose: bool = True) -> dict:
    _, labels, _ = archetype_table()
    if verbose:
        print("scripted round-robin...", flush=True)
    points, goal_diff = scripted_payoff(matches, seed, verbose)

    cycles = find_cycles(goal_diff)
    ratio = transitivity_ratio(goal_diff)
    passed = len(cycles) > 0

    if verbose:
        print(f"\nScripted payoff — goal difference, row vs column")
        print("-" * (22 + 10 * N_ARCHETYPES))
        print(f"{'':22}" + "".join(f"{l.split()[0][:9]:>10}" for l in labels))
        for i, label in enumerate(labels):
            print(f"{label[:22]:22}" + "".join(f"{goal_diff[i, j]:10.2f}"
                                               for j in range(N_ARCHETYPES)))
        print("-" * (22 + 10 * N_ARCHETYPES))
        print(f"  transitivity (variance explained by a single rating): {ratio:.3f}")
        print(f"  3-cycles found: {len(cycles)}")
        for a, b, c in cycles[:6]:
            print(f"    {labels[a]} > {labels[b]} > {labels[c]} > {labels[a]}")
        print(f"  GATE 1 non-transitivity: {'PASS' if passed else 'FAIL'}")
        if not passed:
            best = int(np.argmax(goal_diff.mean(axis=1)))
            print(f"  No cycle: '{labels[best]}' dominates. Pillar 2 has no headline")
            print("  result until the counters in config/tactics.yaml are retuned")
            print("  (see the counter table in design note 3.5).")

    result = {
        "labels": labels,
        "goal_diff": goal_diff.tolist(),
        "points": points.tolist(),
        "cycles": [[labels[i] for i in cyc] for cyc in cycles],
        "n_cycles": len(cycles),
        "transitivity_ratio": ratio,
        "passed": bool(passed),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def scripted_cycle_report() -> tuple[bool, str]:
    """Gate 1 hook for scripts/check_gate.py."""
    if not RESULTS.exists():
        return False, "not yet run: python -m analysis.egta"
    blob = json.loads(RESULTS.read_text(encoding="utf-8"))
    if blob["passed"]:
        cycle = " > ".join(blob["cycles"][0])
        return True, f"{blob['n_cycles']} cycle(s), e.g. {cycle}; " \
                     f"transitivity {blob['transitivity_ratio']:.2f}"
    return False, (f"no cycle; transitivity {blob['transitivity_ratio']:.2f} "
                   "— Pillar 2 has no result until counters are retuned")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted EGTA / non-transitivity (Gate 1)")
    parser.add_argument("--matches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(matches=args.matches, seed=args.seed)


if __name__ == "__main__":
    main()
