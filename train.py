"""Training entry point — the Phase-2 curriculum.

    python train.py --stage 0 --steps 20000            # sanity: beat a fixed opponent
    python train.py --stage 1 --steps 200000           # archetype gauntlet
    python train.py --baselines                        # ranking without any learning
    python train.py --ablation                         # blind / bayesian / oracle

Stages follow plan section 3.5. Every run writes a checkpoint plus a JSON record
of its evaluation so the figures can be rebuilt without retraining.

The learner is `SMDPMaskablePPO`: MaskablePPO with the semi-Markov discount
gamma^tau (plan section 3.3). Flat-gamma is available as an ablation via
`--flat-gamma`, which is the comparison the report needs rather than an
implementation detail.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agents import baselines as bl
from agents.smdp_ppo import SMDPMaskablePPO
from analysis import evaluate as ev
from envs.coach_env import VecCoachEnv

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "data"


def make_env(n_envs: int, belief_mode: str, seed: int) -> VecCoachEnv:
    return VecCoachEnv(n_envs=n_envs, belief_mode=belief_mode, seed=seed)


def train(stage: int = 1, steps: int = 200_000, n_envs: int = 32,
          belief_mode: str = "bayesian", seed: int = 0, smdp: bool = True,
          verbose: int = 1):
    env = make_env(n_envs, belief_mode, seed)
    cfg = env.train_cfg["maskable_ppo"]

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
    tag = f"stage{stage}_{belief_mode}_{'smdp' if smdp else 'flat'}_seed{seed}"
    model.save(CHECKPOINTS / tag)
    print(f"trained {steps} steps in {elapsed / 60:.1f} min -> {tag}")
    return model, env, tag


def evaluate_agent(agent, env, n_matches: int, seed: int = 0) -> dict:
    outcomes = ev.play(agent, env, n_matches)
    return ev.summarise(outcomes, seed=seed)


def run_baselines(n_envs: int = 32, n_matches: int = 200, seed: int = 0) -> dict:
    """The ranking any learned policy must beat (Gate 2's sanity check)."""
    results = {}
    for name in ("random", "static", "greedy", "tabular_q"):
        env = make_env(n_envs, "bayesian", seed)
        agent = bl.make(name, seed=seed)
        results[name] = evaluate_agent(agent, env, n_matches, seed)
        r = results[name]
        print(f"{name:10} ppm {r['points_per_match']:.3f} "
              f"[{r['ppm_ci'][0]:.3f}, {r['ppm_ci'][1]:.3f}]  "
              f"W/D/L {r['win_rate']:.2f}/{r['draw_rate']:.2f}/{r['loss_rate']:.2f}")
    ev.save(results, "baselines")
    return results


def run_ablation(steps: int = 60_000, n_envs: int = 32, n_matches: int = 200,
                 seed: int = 0) -> dict:
    """Blind vs Bayesian vs oracle — the paper-shaped result (plan section 3.4)."""
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
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.baselines:
        run_baselines(args.n_envs, args.matches, args.seed)
        return
    if args.ablation:
        run_ablation(args.steps, args.n_envs, args.matches, args.seed)
        return

    model, env, tag = train(args.stage, args.steps, args.n_envs,
                            args.belief, args.seed, smdp=not args.flat_gamma)
    summary = evaluate_agent(model, env, args.matches, args.seed)
    print(json.dumps(summary, indent=2, default=float))
    ev.save({tag: summary}, tag)


if __name__ == "__main__":
    main()
