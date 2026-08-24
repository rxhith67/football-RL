"""Training entry point — the Phase-2 curriculum.

    python train.py --stage 0 --steps 20000            # sanity: beat a fixed opponent
    python train.py --stage 1 --steps 200000           # archetype gauntlet
    python train.py --baselines                        # ranking without any learning
    python train.py --ablation                         # blind / bayesian / oracle

Stages follow design note 3.5. Every run writes a checkpoint plus a JSON record
of its evaluation so the figures can be rebuilt without retraining.

The learner is `SMDPMaskablePPO`: MaskablePPO with the semi-Markov discount
gamma^tau (design note 3.3). Flat-gamma is available as an ablation via
`--flat-gamma`, which is the comparison the write-up needs rather than an
implementation detail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from agents import baselines as bl
from agents.league import PFSPLeague
from agents.smdp_ppo import SMDPMaskablePPO
from analysis import evaluate as ev
from envs.coach_env import VecCoachEnv
from pitchsim.controller import config_fingerprint

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "data"


def learning_fingerprint() -> str:
    """Fingerprint the simulator, MDP, agents, and learning configuration."""
    digest = hashlib.sha256(config_fingerprint().encode("ascii"))
    for relative in (
        "config/train.yaml", "envs/archetypes.py", "envs/belief.py",
        "envs/coach_env.py", "agents/baselines.py", "agents/smdp_ppo.py",
        "analysis/evaluate.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:16]


def make_env(n_envs: int, belief_mode: str, seed: int,
             stage: int = 1) -> VecCoachEnv:
    if stage not in (0, 1, 2):
        raise ValueError("stage must be 0, 1, or 2")
    opponent = {0: "fixed", 1: "archetype", 2: "policy"}[stage]
    env = VecCoachEnv(n_envs=n_envs, belief_mode=belief_mode,
                      opponent=opponent, seed=seed)
    if stage == 2:
        league = PFSPLeague.load()
        opponent_path = league.sample(np.random.default_rng(seed))
        opponent_model = SMDPMaskablePPO.load(opponent_path, device="cpu")

        def opponent_policy(obs):
            actions, _ = opponent_model.predict(obs, deterministic=True)
            return actions

        env.opponent_policy = opponent_policy
        env.league_opponent = opponent_path
    return env


def train(stage: int = 1, steps: int = 200_000, n_envs: int = 32,
          belief_mode: str = "bayesian", seed: int = 0, smdp: bool = True,
          verbose: int = 1, init_checkpoint: str | Path | None = None,
          tag_suffix: str = ""):
    env = make_env(n_envs, belief_mode, seed, stage=stage)
    cfg = env.train_cfg["maskable_ppo"]

    parent = Path(init_checkpoint) if init_checkpoint else None
    if parent is not None:
        if parent.suffix != ".zip":
            parent = parent.with_suffix(".zip")
        if not parent.is_absolute():
            parent = CHECKPOINTS / parent
        if not parent.exists():
            raise FileNotFoundError(parent)
        meta_path = parent.with_suffix(".json")
        if not meta_path.exists():
            raise ValueError(f"checkpoint has no lineage metadata: {meta_path}")
        parent_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if parent_meta.get("learning_fingerprint") != learning_fingerprint():
            raise ValueError(f"checkpoint is stale for the current learning stack: {parent}")
        if int(parent_meta.get("stage", -1)) != stage - 1:
            raise ValueError("curriculum checkpoint must come from the immediately preceding stage")
        model = SMDPMaskablePPO.load(parent, env=env, device="cpu")
        model.verbose = verbose
        # `load` restores `smdp` from the parent, so this run's choice must be
        # re-applied or the flat-gamma ablation silently trains gamma^tau.
        model.set_discounting(smdp)
    else:
        if stage > 0:
            raise ValueError(
                f"Stage {stage} requires --init from Stage {stage - 1}; "
                "curriculum stages must not restart from random weights"
            )
        model = SMDPMaskablePPO(
            cfg["policy"], env,
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            ent_coef=float(cfg["ent_coef"]),
            vf_coef=float(cfg["vf_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            policy_kwargs={"net_arch": list(cfg["net_arch"])},
            gamma_per_minute=float(env.train_cfg["smdp"]["gamma_per_minute"]),
            smdp=smdp,
            seed=seed,
            verbose=verbose,
        )

    start = time.perf_counter()
    model.learn(total_timesteps=steps)
    elapsed = time.perf_counter() - start

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    # `tag_suffix` keeps side studies (ablations, probes) from overwriting the
    # curriculum checkpoint of the same shape, whose name is cited by evidence
    # already on disk.
    tag = (f"stage{stage}_{belief_mode}_{'smdp' if smdp else 'flat'}"
           f"_seed{seed}{tag_suffix}")
    model.save(CHECKPOINTS / tag)
    (CHECKPOINTS / f"{tag}.json").write_text(json.dumps({
        "learning_fingerprint": learning_fingerprint(),
        "stage": stage,
        "belief_mode": belief_mode,
        "smdp": smdp,
        "seed": seed,
        "trained_steps": steps,
        "parent": str(parent.resolve()) if parent is not None else None,
        "elapsed_seconds": elapsed,
    }, indent=2), encoding="utf-8")
    if stage >= 1:
        PFSPLeague.load().register((CHECKPOINTS / tag).with_suffix(".zip"))
    print(f"trained {steps} steps in {elapsed / 60:.1f} min -> {tag}")
    return model, env, tag


def evaluate_agent(agent, env, n_matches: int, seed: int = 0) -> dict:
    outcomes = ev.play(agent, env, n_matches)
    return ev.summarise(outcomes, seed=seed)


def _update_baseline_index(fingerprint: str, n_matches: int) -> Path:
    runs = {}
    for path in sorted(RESULTS.glob("eval_baselines_seed*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        meta = blob.get("_meta", {})
        if (meta.get("evidence_fingerprint") == fingerprint
                and meta.get("target_matches") == n_matches):
            runs[f"seed_{meta['seed']}"] = {
                key: value for key, value in blob.items() if key != "_meta"
            }
    index = {
        "_meta": {
            "status": "current",
            "evidence_fingerprint": fingerprint,
            "target_matches_per_cell": n_matches,
            "completed_seeds": len(runs),
        },
        "runs": runs,
    }
    path = RESULTS / "eval_baselines.json"
    path.write_text(json.dumps(index, indent=2, default=float), encoding="utf-8")
    return path


def run_baselines(n_envs: int = 32, n_matches: int = 200, seed: int = 0,
                  names: tuple[str, ...] | None = None) -> dict:
    """The ranking any learned policy must beat (Gate 2's sanity check)."""
    fingerprint = learning_fingerprint()
    path = RESULTS / f"eval_baselines_seed{seed}.json"
    results = {}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        meta = saved.get("_meta", {})
        if (meta.get("evidence_fingerprint") == fingerprint
                and meta.get("target_matches") == n_matches
                and meta.get("seed") == seed):
            results = {key: value for key, value in saved.items() if key != "_meta"}

    selected = names or ("random", "static", "greedy", "tabular_q")
    for name in selected:
        if name in results and results[name].get("n_matches") == n_matches:
            print(f"{name:10} already complete for seed {seed}", flush=True)
            continue
        env = make_env(n_envs, "bayesian", seed, stage=1)
        agent = bl.make(name, seed=seed)
        results[name] = evaluate_agent(agent, env, n_matches, seed)
        r = results[name]
        print(f"{name:10} ppm {r['points_per_match']:.3f} "
              f"[{r['ppm_ci'][0]:.3f}, {r['ppm_ci'][1]:.3f}]  "
              f"W/D/L {r['win_rate']:.2f}/{r['draw_rate']:.2f}/{r['loss_rate']:.2f}",
              flush=True)
        payload = {
            "_meta": {
                "status": "current",
                "evidence_fingerprint": fingerprint,
                "target_matches": n_matches,
                "seed": seed,
            },
            **results,
        }
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        _update_baseline_index(fingerprint, n_matches)
    return results


def run_ablation(steps: int = 60_000, n_envs: int = 32, n_matches: int = 200,
                 seed: int = 0) -> dict:
    """Blind vs Bayesian vs oracle — the paper-shaped result (design note 3.4)."""
    results = {}
    for mode in ("blind", "bayesian", "oracle"):
        model, env, _ = train(stage=1, steps=steps, n_envs=n_envs,
                              belief_mode=mode, seed=seed, verbose=0)
        results[mode] = evaluate_agent(model, env, n_matches, seed)
        r = results[mode]
        print(f"{mode:10} ppm {r['points_per_match']:.3f} "
              f"[{r['ppm_ci'][0]:.3f}, {r['ppm_ci'][1]:.3f}]")

    if "blind" in results and "bayesian" in results:
        gap = results["oracle"]["points_per_match"] - results["blind"]["points_per_match"]
        recovered = (results["bayesian"]["points_per_match"]
                     - results["blind"]["points_per_match"])
        results["oracle_gap"] = gap
        results["belief_recovers"] = float(recovered / gap) if abs(gap) > 1e-9 else float("nan")
        print(f"\noracle gap {gap:.3f} ppm; belief recovers "
              f"{100 * results['belief_recovers']:.0f}% of it")
    ev.save(results, "ablation_belief")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GAFFER-RL training")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--belief", default="bayesian",
                        choices=["bayesian", "blind", "oracle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat-gamma", action="store_true",
                        help="ablation: flat gamma instead of gamma^tau")
    parser.add_argument("--init", type=Path,
                        help="checkpoint from the immediately preceding curriculum stage")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--baseline", action="append",
                        choices=["random", "static", "greedy", "tabular_q"],
                        help="evaluate only this baseline (repeatable; results resume per seed)")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.baselines:
        run_baselines(args.n_envs, args.matches, args.seed,
                      tuple(args.baseline) if args.baseline else None)
        return
    if args.ablation:
        run_ablation(args.steps, args.n_envs, args.matches, args.seed)
        return

    model, env, tag = train(args.stage, args.steps, args.n_envs,
                            args.belief, args.seed, smdp=not args.flat_gamma,
                            init_checkpoint=args.init)
    summary = evaluate_agent(model, env, args.matches, args.seed)
    print(json.dumps(summary, indent=2, default=float))
    ev.save({tag: summary}, tag)


if __name__ == "__main__":
    main()
