"""Expected-goals model, fitted on real StatsBomb shots.

This is the Gate 0 deliverable (design note 6.2): held-out AUC > 0.75. It matters
beyond the number, because xG appears in three separate places downstream —
as the sim's goal-sampling mechanism, as the potential function in the shaped
reward (design note 3.3), and as the xG-race panel in the console. The original
draft used a placeholder xG; this one is fitted and falsifiable.

Fitted coefficients serialise to JSON rather than pickle so the model is
inspectable, diffable in git, and loadable without scikit-learn at demo time.

CLI::

    python -m pitchsim.xg --fit
    python -m pitchsim.xg --report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "statsbomb" / "xg_model.json"

FEATURES = (
    "distance_m",
    "angle_rad",
    "defenders_in_cone",
    "nearest_defender_m",
    "gk_off_line_m",
    "gk_lateral_m",
    "is_header",
    "is_open_play",
    "under_pressure",
    "is_first_time",
    "is_one_on_one",
    "is_volley",
    "from_counter",
)
MIN_AUC = 0.75      # Gate 0 acceptance criterion

#: The gate is evaluated on the CROSS-VALIDATED mean, not a single split. With
#: only a few hundred goals, one held-out fold swings the AUC by several points,
#: and picking the seed that clears the bar would be fitting the gate rather than
#: the data. GroupKFold on match_id also prevents leakage: shots within a match
#: share context, so a shot-level split inflates the estimate.
N_FOLDS = 5


@dataclass
class XGModel:
    """Logistic xG. Pure NumPy at inference, so the engine has no sklearn dependency."""

    coefficients: np.ndarray
    intercept: float
    feature_names: tuple[str, ...] = FEATURES
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    metrics: dict | None = None

    def _design(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        # Non-linear terms: xG falls off sharply with distance and rises with the
        # visible goal angle, neither of which a raw-linear fit captures well.
        extra = np.stack([np.log1p(x[:, 0]), 1.0 / (x[:, 0] + 1.0)], axis=1)
        x = np.concatenate([x, extra], axis=1)
        if self.mean is not None:
            x = (x - self.mean) / self.scale
        return x

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = self._design(features) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    #: League-average values, used for any feature the caller does not supply.
    #: The engine only ever knows distance, angle and the defensive picture, so
    #: the remaining fields fall back to these rather than to zero.
    DEFAULTS = {
        "distance_m": 16.5, "angle_rad": 0.46, "defenders_in_cone": 1.7,
        "nearest_defender_m": 3.5, "gk_off_line_m": 1.5, "gk_lateral_m": 0.4,
        "is_header": 0.0, "is_open_play": 1.0, "under_pressure": 0.0,
        "is_first_time": 0.0, "is_one_on_one": 0.0, "is_volley": 0.0,
        "from_counter": 0.0, "has_freeze_frame": 1.0,
    }

    def predict_one(self, distance_m: float, angle_rad: float, **kwargs: float) -> float:
        """Convenience entry point for the engine's shot event.

        Any feature not supplied falls back to `DEFAULTS`, so the engine can pass
        only what it actually models.
        """
        values = dict(self.DEFAULTS)
        values.update(distance_m=distance_m, angle_rad=angle_rad, **kwargs)
        unknown = set(kwargs) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown xG feature(s): {sorted(unknown)}")
        row = np.array([[values[name] for name in self.feature_names]], dtype=np.float64)
        return float(self.predict(row)[0])

    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "coefficients": self.coefficients.tolist(),
                    "intercept": float(self.intercept),
                    "feature_names": list(self.feature_names),
                    "mean": None if self.mean is None else self.mean.tolist(),
                    "scale": None if self.scale is None else self.scale.tolist(),
                    "metrics": self.metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "XGModel":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m pitchsim.xg --fit"
            )
        blob = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            coefficients=np.array(blob["coefficients"], dtype=np.float64),
            intercept=blob["intercept"],
            feature_names=tuple(blob["feature_names"]),
            mean=None if blob["mean"] is None else np.array(blob["mean"]),
            scale=None if blob["scale"] is None else np.array(blob["scale"]),
            metrics=blob.get("metrics"),
        )


def fit(seed: int = 0, verbose: bool = True) -> XGModel:
    """Fit on StatsBomb shots and report Gate 0 metrics.

    Evaluation is 5-fold GroupKFold on `match_id`: out-of-fold predictions are
    pooled to give one AUC over every shot, and the per-fold spread is reported
    so the reader can see how much of any margin over the gate is noise. The
    shipped model is then refitted on all data, which is standard practice and
    stated plainly rather than quietly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from sklearn.model_selection import GroupKFold

    from .statsbomb import load_shots

    df = load_shots()
    available = [f for f in FEATURES if f in df.columns]
    if len(available) < len(FEATURES) and verbose:
        missing = set(FEATURES) - set(available)
        print(f"! shots.parquet lacks {sorted(missing)} — re-run the fetch to use them")
    df = df.dropna(subset=available + ["goal"])
    if verbose:
        print(f"fitting on {len(df)} shots ({int(df.goal.sum())} goals) "
              f"from {df.match_id.nunique()} matches, {len(available)} features")

    x_all = df[available].to_numpy(dtype=np.float64)
    y_all = df["goal"].to_numpy(dtype=np.int64)
    groups = df["match_id"].to_numpy()

    scaffold = XGModel(coefficients=np.zeros(1), intercept=0.0, feature_names=tuple(available))
    design_raw = scaffold._design(x_all)

    # --- cross-validated, leakage-free evaluation ---
    oof = np.zeros(len(df))
    fold_aucs: list[float] = []
    splitter = GroupKFold(n_splits=min(N_FOLDS, df.match_id.nunique()))
    for train_idx, test_idx in splitter.split(design_raw, y_all, groups):
        mu = design_raw[train_idx].mean(axis=0)
        sd = design_raw[train_idx].std(axis=0) + 1e-9
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit((design_raw[train_idx] - mu) / sd, y_all[train_idx])
        oof[test_idx] = clf.predict_proba((design_raw[test_idx] - mu) / sd)[:, 1]
        if len(np.unique(y_all[test_idx])) > 1:
            fold_aucs.append(float(roc_auc_score(y_all[test_idx], oof[test_idx])))

    metrics = {
        "n_shots": int(len(df)),
        "n_goals": int(y_all.sum()),
        "n_matches": int(df.match_id.nunique()),
        "n_features": len(available),
        "n_folds": len(fold_aucs),
        "auc": float(roc_auc_score(y_all, oof)),          # pooled out-of-fold
        "auc_fold_mean": float(np.mean(fold_aucs)),
        "auc_fold_std": float(np.std(fold_aucs)),
        "brier": float(brier_score_loss(y_all, oof)),
        "log_loss": float(log_loss(y_all, oof)),
        "mean_pred_xg": float(oof.mean()),
        "actual_conversion": float(y_all.mean()),
    }
    if "sb_xg" in df.columns and df["sb_xg"].notna().any():
        sb = df["sb_xg"].to_numpy()
        ok = ~np.isnan(sb)
        if ok.sum() > 10:
            metrics["statsbomb_auc"] = float(roc_auc_score(y_all[ok], sb[ok]))
            metrics["correlation_with_statsbomb"] = float(np.corrcoef(oof[ok], sb[ok])[0, 1])

    # --- ship a model refitted on everything ---
    mean, scale = design_raw.mean(axis=0), design_raw.std(axis=0) + 1e-9
    final = LogisticRegression(max_iter=3000, C=1.0)
    final.fit((design_raw - mean) / scale, y_all)

    model = XGModel(
        coefficients=final.coef_.ravel(),
        intercept=float(final.intercept_[0]),
        feature_names=tuple(available),
        mean=mean,
        scale=scale,
        metrics=metrics,
    )
    model.save()

    if verbose:
        _print_report(metrics)
    return model


def _print_report(metrics: dict) -> None:
    print(f"\nxG model — {metrics.get('n_folds', '?')}-fold GroupKFold by match "
          "(out-of-fold predictions)")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"  {key:26} {value:.4f}" if isinstance(value, float) else f"  {key:26} {value}")

    auc = metrics["auc"]
    verdict = "PASS" if auc >= MIN_AUC else "FAIL"
    print("-" * 60)
    print(f"  GATE 0: pooled out-of-fold AUC {auc:.4f} vs required >= {MIN_AUC}  ->  {verdict}")
    if "auc_fold_std" in metrics:
        print(f"  per-fold {metrics['auc_fold_mean']:.4f} +/- {metrics['auc_fold_std']:.4f} "
              "— read any margin against this spread")
    if "statsbomb_auc" in metrics:
        print(f"  StatsBomb's own xG scores {metrics['statsbomb_auc']:.4f} on the same "
              "shots: the practical ceiling,")
        print("  since they use the full freeze frame and a far larger training set.")


def main() -> None:
    parser = argparse.ArgumentParser(description="xG model")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.fit:
        fit(seed=args.seed)
    elif args.report:
        model = XGModel.load()
        _print_report(model.metrics or {})
        print("\nSanity checks (open play, league-average defensive picture):")
        for distance, angle, label in [
            (6.0, 1.00, "six-yard box, central"),
            (11.0, 0.80, "penalty spot area"),
            (18.0, 0.35, "edge of the box"),
            (30.0, 0.18, "long range"),
        ]:
            print(f"  {label:24} d={distance:5.1f}m  xG = "
                  f"{model.predict_one(distance, angle):.3f}")
        print(f"  {'header, 8m':24} d=  8.0m  xG = "
              f"{model.predict_one(8.0, 0.9, is_header=1.0):.3f}")
        print(f"  {'one-on-one, 12m':24} d= 12.0m  xG = "
              f"{model.predict_one(12.0, 0.7, is_one_on_one=1.0, defenders_in_cone=0.0):.3f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
