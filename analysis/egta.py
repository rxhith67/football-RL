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
PROGRESS = ROOT / "data" / "egta_progress.json"


def play_pairing(
    row_tactic: np.ndarray, col_tactic: np.ndarray, matches: int, seed: int,
    engine: BatchedEngine | None = None,
) -> tuple[float, float, float]:
    """Play `matches` between two fixed tactics. Returns (row points-per-match, row goal diff).

    Sides are swapped for half the batch so venue cannot confound the payoff.
    """
    if engine is None:
        _, physics = ctrl.load_config()
        engine = BatchedEngine(batch=matches, config=physics)
    elif engine.batch != matches:
        raise ValueError("reused engine batch must equal matches")

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
    diff = row_goals - col_goals
    # Standard error of the mean goal difference. Without it a payoff entry of
    # +0.5 is indistinguishable from -0.5 and any cycle built on such an edge is
    # an artefact of the sample, not a property of the game.
    stderr = float(np.std(diff, ddof=1) / np.sqrt(max(len(diff), 1))) if len(diff) > 1 else 0.0
    return float(points.mean()), float(diff.mean()), stderr


def scripted_payoff(matches_per_pairing: int | None = None, seed: int = 0,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K x K payoff matrices (points-per-match, goal difference) over archetypes."""
    tactics_cfg, physics = ctrl.load_config()
    if matches_per_pairing is None:
        import yaml

        train = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
        matches_per_pairing = int(train["evaluation"]["scripted_roundrobin_matches"])

    if matches_per_pairing < 4 or matches_per_pairing % 2:
        raise ValueError("matches_per_pairing must be an even integer >= 4")

    _, labels, base = archetype_table()
    points = np.zeros((N_ARCHETYPES, N_ARCHETYPES))
    goal_diff = np.zeros((N_ARCHETYPES, N_ARCHETYPES))
    stderr = np.zeros((N_ARCHETYPES, N_ARCHETYPES))

    np.fill_diagonal(points, 1.5)
    pairings = list(combinations(range(N_ARCHETYPES), 2))
    # The engine is fastest around 48-64 rows on the reference CPU.  Packing
    # several tactic pairings into one run shares every physics tick and turns a
    # fifteen-run round robin into three or four runs for audit-sized samples.
    pairs_per_batch = max(1, 64 // matches_per_pairing)
    half = matches_per_pairing // 2
    first_pair = 0
    if PROGRESS.exists():
        saved = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if (saved.get("config_fingerprint") == ctrl.config_fingerprint()
                and saved.get("matches_per_pairing") == matches_per_pairing
                and saved.get("seed") == seed):
            points = np.asarray(saved["points"], dtype=np.float64)
            goal_diff = np.asarray(saved["goal_diff"], dtype=np.float64)
            stderr = np.asarray(saved["stderr"], dtype=np.float64)
            first_pair = int(saved.get("completed_pairings", 0))
            if verbose and first_pair:
                print(f"  resuming after {first_pair} completed pairings", flush=True)

    for chunk_start in range(first_pair, len(pairings), pairs_per_batch):
        chunk = pairings[chunk_start:chunk_start + pairs_per_batch]
        batch = matches_per_pairing * len(chunk)
        tactics = np.empty((batch, 2, 5), dtype=np.int64)
        for p, (i, j) in enumerate(chunk):
            sl = slice(p * matches_per_pairing, (p + 1) * matches_per_pairing)
            block = tactics[sl]
            block[:half, 0], block[:half, 1] = base[i], base[j]
            block[half:, 0], block[half:, 1] = base[j], base[i]

        engine = BatchedEngine(batch=batch, config=physics)
        engine.reset(seed=seed + chunk_start * 97, tactics=tactics)
        engine.run_until(engine.match_seconds, stop_on_goal=False)

        for p, (i, j) in enumerate(chunk):
            start = p * matches_per_pairing
            score = engine.score[start:start + matches_per_pairing]
            row_goals = np.concatenate([score[:half, 0], score[half:, 1]])
            col_goals = np.concatenate([score[:half, 1], score[half:, 0]])
            row_points = np.where(row_goals > col_goals, 3.0,
                                  np.where(row_goals == col_goals, 1.0, 0.0))
            col_points = np.where(col_goals > row_goals, 3.0,
                                  np.where(col_goals == row_goals, 1.0, 0.0))
            diff = row_goals - col_goals
            points[i, j], points[j, i] = row_points.mean(), col_points.mean()
            goal_diff[i, j], goal_diff[j, i] = diff.mean(), -diff.mean()
            se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
            stderr[i, j] = stderr[j, i] = se
        if verbose:
            print(f"  pairings {chunk_start + 1:02d}-{chunk_start + len(chunk):02d} done",
                  flush=True)
        PROGRESS.write_text(json.dumps({
            "config_fingerprint": ctrl.config_fingerprint(),
            "matches_per_pairing": matches_per_pairing,
            "seed": seed,
            "completed_pairings": chunk_start + len(chunk),
            "points": points.tolist(),
            "goal_diff": goal_diff.tolist(),
            "stderr": stderr.tolist(),
        }), encoding="utf-8")
    return points, goal_diff, stderr


def find_cycles(matrix: np.ndarray, stderr: np.ndarray | None = None,
                z: float = 2.0) -> list[tuple[int, ...]]:
    """All 3-cycles in the 'beats' relation implied by `matrix`.

    A cycle i -> j -> k -> i is the rock-paper-scissors structure the tactic-space
    analysis claims to find, so every edge in it must be REAL.

    With `stderr` supplied, i only "beats" j when the margin clears `z` standard
    errors. This is not pedantry: with a bare sign test the same configuration
    reported three cycles at 30 matches per pairing and none at 40, because every
    one of them ran through a High-press vs Direct edge worth +0.47 goals — well
    inside the noise. A cycle resting on an edge that flips with the sample is an
    artefact, and reporting it would have been a false positive.
    """
    beats = matrix > 0
    if stderr is not None:
        beats &= np.abs(matrix) > z * np.maximum(stderr, 1e-9)
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


def payoff_matrix(strategies: list, play) -> np.ndarray:
    """Evaluate an antisymmetric payoff matrix with a caller-supplied match fn.

    ``play(a, b)`` returns row payoff (positive means ``a`` is better).  This
    small interface works for scripted tactics and trained checkpoints alike.
    """
    n = len(strategies)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, j in combinations(range(n), 2):
        value = float(play(strategies[i], strategies[j]))
        matrix[i, j], matrix[j, i] = value, -value
    return matrix


def nash(matrix: np.ndarray) -> dict[str, list[float]]:
    """Return one mixed Nash equilibrium of a finite two-player zero-sum game."""
    import nashpy as nashpy

    game = nashpy.Game(np.asarray(matrix, dtype=np.float64),
                       -np.asarray(matrix, dtype=np.float64))
    equilibria = list(game.support_enumeration())
    if not equilibria:
        raise RuntimeError("no Nash equilibrium found")
    row, column = equilibria[0]
    return {"row": np.asarray(row).tolist(), "column": np.asarray(column).tolist()}


def alpha_rank(matrix: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """Monomorphic α-Rank stationary distribution over symmetric strategies.

    The transition from resident ``i`` to mutant ``j`` is a logistic function of
    ``j``'s payoff against ``i``.  Returning the stationary distribution keeps
    the analysis dependency-light and deterministic for report generation.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n = len(matrix)
    if matrix.shape != (n, n) or n < 2:
        raise ValueError("matrix must be square with at least two strategies")
    transition = np.zeros_like(matrix)
    scale = 1.0 / (n - 1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            advantage = matrix[j, i]
            transition[i, j] = scale / (1.0 + np.exp(-alpha * advantage))
        transition[i, i] = 1.0 - transition[i].sum()
    values, vectors = np.linalg.eig(transition.T)
    stationary = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    stationary = np.maximum(stationary, 0.0)
    if stationary.sum() <= 1e-12:
        stationary = np.abs(np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))]))
    return stationary / stationary.sum()


def run(matches: int | None = None, seed: int = 0, verbose: bool = True) -> dict:
    _, labels, _ = archetype_table()
    if matches is None:
        import yaml
        train_cfg = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(
            encoding="utf-8"))
        used_matches = int(train_cfg["evaluation"]["scripted_roundrobin_matches"])
    else:
        used_matches = int(matches)
    if verbose:
        print("scripted round-robin...", flush=True)
    points, goal_diff, stderr = scripted_payoff(matches, seed, verbose)

    cycles = find_cycles(goal_diff, stderr)
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
        print(f"  median |goal diff| {np.median(np.abs(goal_diff[~np.eye(N_ARCHETYPES, dtype=bool)])):.2f}"
              f"  median SE {np.median(stderr[~np.eye(N_ARCHETYPES, dtype=bool)]):.2f}"
              f"  (edges need > 2 SE to count)")
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
        "config_fingerprint": ctrl.config_fingerprint(),
        "matches_per_pairing": used_matches,
        "labels": labels,
        "goal_diff": goal_diff.tolist(),
        "stderr": stderr.tolist(),
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
    if blob.get("config_fingerprint") != ctrl.config_fingerprint():
        return False, "stale evidence: rerun python -m analysis.egta"
    import yaml
    train_cfg = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(
        encoding="utf-8"))
    required = int(train_cfg["evaluation"]["scripted_roundrobin_matches"])
    actual = int(blob.get("matches_per_pairing", 0))
    if actual < required:
        return False, f"only {actual} matches/pairing; need {required}"
    if blob["passed"]:
        cycle = " > ".join(blob["cycles"][0])
        return True, f"{blob['n_cycles']} cycle(s), e.g. {cycle}; " \
                     f"transitivity {blob['transitivity_ratio']:.2f}"
    return False, (f"no cycle; transitivity {blob['transitivity_ratio']:.2f} "
                   "— Pillar 2 has no result until counters are retuned")


FULL_RESULTS = ROOT / "data" / "egta_full.json"


def run_full(seed: int = 0) -> dict:
    """Nash + alpha-Rank over the scripted archetype game (Gate 3).

    Reuses the Gate-1 goal-difference payoff matrix — the same game, asked a
    different question: not "is there a cycle" but "what does the solution
    concept say the tactic mix should look like".
    """
    if not RESULTS.exists():
        raise FileNotFoundError("run python -m analysis.egta first (Gate 1 payoff matrix)")
    blob = json.loads(RESULTS.read_text(encoding="utf-8"))
    if blob.get("config_fingerprint") != ctrl.config_fingerprint():
        raise ValueError("stale Gate 1 evidence; rerun python -m analysis.egta")
    labels = blob["labels"]
    matrix = np.asarray(blob["goal_diff"], dtype=np.float64)

    equilibrium = nash(matrix)
    stationary = alpha_rank(matrix)

    result = {
        "config_fingerprint": ctrl.config_fingerprint(),
        "labels": labels,
        "nash": equilibrium,
        "alpha_rank": stationary.tolist(),
    }
    FULL_RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def full_egta_report() -> tuple[bool, str]:
    """Gate 3 hook: EGTA payoff matrix + Nash/alpha-Rank complete."""
    if not FULL_RESULTS.exists():
        return False, "not yet run: python -m analysis.egta --full"
    blob = json.loads(FULL_RESULTS.read_text(encoding="utf-8"))
    if blob.get("config_fingerprint") != ctrl.config_fingerprint():
        return False, "stale evidence: rerun python -m analysis.egta --full"
    labels = blob["labels"]
    row = blob["nash"]["row"]
    ranked = blob["alpha_rank"]
    if abs(sum(row) - 1.0) > 1e-6 or abs(sum(ranked) - 1.0) > 1e-6:
        return False, "malformed equilibrium (does not sum to 1)"
    top_nash = labels[int(np.argmax(row))]
    top_rank = labels[int(np.argmax(ranked))]
    return True, f"Nash support peak: {top_nash} ({max(row):.2f}); alpha-Rank peak: {top_rank} ({max(ranked):.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted EGTA / non-transitivity (Gate 1)")
    parser.add_argument("--full", action="store_true",
                        help="Gate 3: Nash + alpha-Rank over the existing payoff matrix")
    parser.add_argument("--matches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.full:
        result = run_full(seed=args.seed)
        print(json.dumps(result, indent=2))
    else:
        run(matches=args.matches, seed=args.seed)


if __name__ == "__main__":
    main()
