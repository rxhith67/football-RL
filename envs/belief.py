"""Bayesian belief filter over opponent archetypes — Pillar 1.

The opposing manager's plan is hidden state. Only its statistical footprint is
observable, so the coach maintains a posterior over K archetypes and conditions
its policy on that posterior (design note 3.4):

    b_{k+1}(j)  ∝  sum_i b_k(i) T(i -> j) * N(o_k ; mu_j, Sigma_j)

The transition matrix `T` matters and is not a formality: scripted opponents
genuinely switch plan mid-match, so a filter with a near-identity `T` cannot
follow them and adaptation latency blows up.

Everything is diagonal-Gaussian and batched. That is a modelling choice, not
laziness — the same model is scored by `analysis.separability`, so the
separability number is a direct estimate of this filter's ceiling rather than a
flattering figure from some stronger classifier the filter could never match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pitchsim.controller import config_fingerprint, load_config
from pitchsim.types import WindowStats

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "belief_model.json"


class BeliefFilter:
    """Batched posterior over K opponent archetypes."""

    def __init__(
        self,
        means: np.ndarray,
        variances: np.ndarray,
        self_transition: float = 0.90,
        min_variance: float = 1e-4,
    ) -> None:
        self.means = np.asarray(means, dtype=np.float64)          # (K, F)
        self.variances = np.maximum(np.asarray(variances, dtype=np.float64), min_variance)
        self.k = len(self.means)

        # Uniform leakage to every other archetype.
        off = (1.0 - self_transition) / max(self.k - 1, 1)
        self.transition = np.full((self.k, self.k), off)
        np.fill_diagonal(self.transition, self_transition)

        self.belief = np.full((1, self.k), 1.0 / self.k)

    # ------------------------------------------------------------------

    def reset(self, batch: int) -> np.ndarray:
        """Uniform prior — the coach starts a match knowing nothing."""
        self.belief = np.full((batch, self.k), 1.0 / self.k)
        return self.belief.copy()

    def log_likelihood(self, observations: np.ndarray) -> np.ndarray:
        """(B, F) observations -> (B, K) log N(o ; mu_j, Sigma_j)."""
        diff = observations[:, None, :] - self.means[None, :, :]
        return -0.5 * np.sum(
            diff**2 / self.variances[None] + np.log(2 * np.pi * self.variances[None]),
            axis=-1,
        )

    def update(self, observations: np.ndarray) -> np.ndarray:
        """One filtering step. Returns the new (B, K) belief."""
        predicted = self.belief @ self.transition          # (B, K)
        log_post = np.log(predicted + 1e-30) + self.log_likelihood(observations)
        log_post -= log_post.max(axis=1, keepdims=True)     # stabilise before exp
        posterior = np.exp(log_post)
        self.belief = posterior / np.maximum(posterior.sum(axis=1, keepdims=True), 1e-30)
        return self.belief.copy()

    # ------------------------------------------------------------------

    def entropy(self) -> np.ndarray:
        """Nats. Drives the `belief_entropy_jump` decision trigger and the
        confidence trace in the console."""
        return -np.sum(self.belief * np.log(self.belief + 1e-30), axis=1)

    def map_estimate(self) -> np.ndarray:
        return np.argmax(self.belief, axis=1)

    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "config_fingerprint": config_fingerprint(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "n_features": int(self.means.shape[1]),
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "BeliefFilter":
        _, _ = load_config()
        import yaml

        train = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
        cfg = train["belief"]

        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m envs.belief --fit"
            )
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("config_fingerprint") != config_fingerprint():
            raise ValueError(
                f"{path} was fitted for a different simulator configuration; "
                "run: python -m analysis.separability"
            )
        return cls(
            np.array(blob["means"]),
            np.array(blob["variances"]),
            self_transition=float(cfg["transition_self_prob"]),
            min_variance=float(cfg["min_variance"]),
        )


def fit(windows_per_archetype: int = 40, seed: int = 0, verbose: bool = True) -> BeliefFilter:
    """Fit the per-archetype Gaussians offline, from simulated windows."""
    from analysis.separability import collect_windows

    if verbose:
        print("simulating archetype windows for the belief model...", flush=True)
    x, y = collect_windows(windows_per_archetype, seed=seed, verbose=verbose)

    k = int(y.max()) + 1
    means = np.stack([x[y == j].mean(axis=0) for j in range(k)])
    variances = np.stack([x[y == j].var(axis=0) for j in range(k)])

    model = BeliefFilter(means, variances)
    model.save()
    if verbose:
        print(f"fitted {k} archetypes on {len(y)} windows "
              f"({WindowStats.N_FEATURES} features) -> {MODEL_PATH}")
    return model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian archetype belief filter")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--windows", type=int, default=40)
    args = parser.parse_args()
    if args.fit:
        fit(windows_per_archetype=args.windows)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
