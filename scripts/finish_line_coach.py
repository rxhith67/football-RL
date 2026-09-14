"""Bounded train/evaluate/export pipeline for the reduced football project.

Linux only for native simulation. The exported website needs only Python's HTTP
server on Windows/Linux; it does not need RoboCup or machine-learning libraries.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import probe_robocup as probe
sys.path.insert(0, str(probe.NATIVE))


def save_json(path, data):
    """Atomic publication so the running demo never reads a partial result."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_env(seed, horizon, output, tag, record=False):
    import torch
    from envs.line_coach_env import LineCoachEnv
    torch.set_num_threads(1)
    os.environ["ROBOCUP_ACTION_DEBUG"] = "0"
    scratch = tempfile.TemporaryDirectory(prefix="line-coach-")
    probe.OUT = Path(scratch.name)
    cfg = probe.configuration(horizon, seed, "native")
    cfg.update(text_logging=False, game_logging=False, reward_on_draw=0.0,
               reward_on_timeout=0.0)
    os.environ["RCSS_CONF_DIR"] = str(Path(scratch.name) / "config")
    env = LineCoachEnv(cfg, decision_steps=20, record=record)
    original_close = env.close

    def close():
        try:
            original_close()
        finally:
            dest = Path(output) / "native-logs" / tag
            if Path(scratch.name).exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(scratch.name, dest, dirs_exist_ok=True)
            scratch.cleanup()
    env.close = close
    return env


def training_factory(seed, horizon, output, tag):
    def factory():
        from stable_baselines3.common.monitor import Monitor
        log = Path(output) / f"{tag}.log"
        sys.stdout = log.open("a", buffering=1)
        sys.stderr = sys.stdout
        return Monitor(make_env(seed, horizon, output, tag),
                       filename=str(Path(output) / tag),
                       info_keywords=("score_left", "score_right"))
    return factory


def train(seed, options, output):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from envs.line_coach_env import GAMMA
    torch.set_num_threads(1)
    episodes = []
    started = time.perf_counter()

    class Progress(BaseCallback):
        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    episodes.append(dict(info["episode"], timesteps=self.num_timesteps))
            if self.num_timesteps % 256 == 0:
                save_json(output / f"training_{seed}.json", {
                    "seed": seed, "timesteps": self.num_timesteps, "episodes": episodes,
                    "seconds": time.perf_counter() - started, "complete": False})
                print(f"TRAIN seed={seed} decisions={self.num_timesteps}/{options.timesteps} episodes={len(episodes)}", flush=True)
            return True

    vec = SubprocVecEnv([training_factory(seed * 10000 + i * 1000, options.horizon,
                        output, f"train_{seed}_{i}") for i in range(options.workers)], start_method="spawn")
    try:
        model = PPO("MlpPolicy", vec, seed=seed, n_steps=128, batch_size=128,
                    n_epochs=4, gamma=GAMMA, learning_rate=3e-4, ent_coef=0.01,
                    policy_kwargs={"net_arch": [64, 64]}, device="cpu", verbose=0)
        before = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(total_timesteps=options.timesteps, callback=Progress())
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))
        if not finite or not changed:
            raise RuntimeError("Training did not produce a finite updated policy")
        model.save(output / f"ppo_seed_{seed}")
        result = {"seed": seed, "timesteps": model.num_timesteps, "episodes": episodes,
                  "seconds": time.perf_counter() - started, "complete": True,
                  "parameters_finite": finite, "parameters_changed": changed}
        save_json(output / f"training_{seed}.json", result)
        return result
    finally:
        vec.close()


def evaluate(job):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    policy, model_seed, episode, horizon, output_string = job
    output = Path(output_string)
    tag = f"eval_{policy}_{model_seed}_{episode}"
    sys.stdout = (output / f"{tag}.log").open("a", buffering=1)
    sys.stderr = sys.stdout
    torch.set_num_threads(1)
    # Held-out episode seeds; random-action RNG is separate from native seed.
    seed = 800000 + episode
    rng = np.random.default_rng(900000 + episode)
    env = make_env(seed, horizon, output, tag, record=episode == 0)
    started = time.perf_counter()
    try:
        model = PPO.load(output / f"ppo_seed_{model_seed}.zip", device="cpu") if policy == "ppo" else None
        obs, _ = env.reset(seed=seed)
        rewards, counts = 0.0, [0, 0, 0]
        while True:
            action = int(model.predict(obs, deterministic=True)[0]) if model is not None else (
                int(rng.integers(3)) if policy == "random" else {"normal": 0, "low": 1, "high": 2}[policy])
            obs, reward, terminated, truncated, info = env.step(action)
            counts[action] += 1
            rewards += reward
            if terminated or truncated:
                break
        difference = env.score[0] - env.score[1]
        result = {"policy": policy, "model_seed": model_seed, "episode": episode,
                  "native_seed": seed, "score_left": env.score[0], "score_right": env.score[1],
                  "goal_difference": difference, "points": 3 if difference > 0 else 1 if difference == 0 else 0,
                  "return": rewards, "action_counts": counts,
                  "seconds": time.perf_counter() - started, "success": True}
        if episode == 0:
            filename = f"replay_{policy}_{model_seed}.json"
            save_json(output / filename, env.replay(f"PPO seed {model_seed}" if model else policy))
            result["replay"] = filename
        save_json(output / f"{tag}.json", result)
        return result
    finally:
        env.close()


def summarize(results, training, options):
    import numpy as np
    rows = []
    rng = np.random.default_rng(4001)
    keys = [("ppo", seed) for seed in options.seeds] + [(p, None) for p in ("normal", "low", "high", "random")]
    for policy, seed in keys:
        group = [r for r in results if r["policy"] == policy and r["model_seed"] == seed]
        scores = np.array([r["goal_difference"] for r in group])
        boot = rng.choice(scores, (3000, len(scores)), replace=True).mean(axis=1)
        rows.append({"name": f"PPO seed {seed}" if policy == "ppo" else {"normal": "Normal HELIOS", "low": "Always low", "high": "Always high", "random": "Random"}[policy],
                     "policy": policy, "seed": seed, "episodes": len(group),
                     "mean_goal_difference": float(scores.mean()),
                     "goal_difference_ci95": np.quantile(boot, [.025, .975]).tolist(),
                     "points_per_episode": float(np.mean([r["points"] for r in group])),
                     "wins": int((scores > 0).sum()), "draws": int((scores == 0).sum()),
                     "losses": int((scores < 0).sum()),
                     "actions": np.sum([r["action_counts"] for r in group], axis=0).tolist(),
                     "replay": next((r["replay"] for r in group if "replay" in r), None)})
    return {"title": "GAFFER / Defensive Line Lab", "complete": True,
            "question": "Does learned defensive-line selection outperform fixed instructions in RoboCup episodes?",
            "horizon": options.horizon, "decision_steps": 20, "training": training, "results": rows,
            "evaluation_episodes": len(results), "selected_demo_seed": options.seeds[0],
            "selection_rule": "First predeclared training seed, not the best test result.",
            "limitations": ["Small fixed-budget experiment; no guarantee of learned advantage.",
                            "Fixed-length episodes, not regulation matches; one opponent and left-side control.",
                            "Same native seed does not give identical trajectories; comparisons use independent runs.",
                            "Intervals are exploratory episode-bootstrap intervals, not proof of superiority.",
                            "No pressing, substitutions, opponent-belief model or real-football transfer claim."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/line_coach")
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 47, 59])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if options.workers < 1 or options.horizon < 20 or options.eval_episodes < 2 or options.timesteps < 128:
        parser.error("workers >= 1, horizon >= 20, eval-episodes >= 2, timesteps >= 128 required")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=options.resume)
    save_json(output / "protocol.json", {
        "requested_timesteps_per_seed": options.timesteps, "training_seeds": options.seeds,
        "workers": options.workers, "horizon": options.horizon, "eval_episodes_per_policy": options.eval_episodes,
        "evaluation_seeds": [800000 + i for i in range(options.eval_episodes)],
        "checkpoint_selection": "Final checkpoint; all training seeds reported. Demo uses first seed.",
        "primary_metric": "Mean raw goal difference over held-out episodes",
        "reward": "Goal difference + 0.05 * (0.99 * next ball-x potential - previous potential); terminal potential zero",
        "scope": "Finite-horizon defensive-line control; no full-match or real-world realism claim",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__).resolve(), ROOT / "envs/line_coach_env.py", ROOT / "envs/robocup_coach_env.py")}})
    started = time.perf_counter()
    training = []
    for seed in options.seeds:
        existing = output / f"training_{seed}.json"
        prior = json.loads(existing.read_text()) if options.resume and existing.exists() else {}
        training.append(prior if prior.get("complete") and (output / f"ppo_seed_{seed}.zip").exists()
                        else train(seed, options, output))
    jobs = [("ppo", seed, ep, options.horizon, str(output)) for seed in options.seeds for ep in range(options.eval_episodes)]
    jobs += [(p, None, ep, options.horizon, str(output)) for p in ("normal", "low", "high", "random") for ep in range(options.eval_episodes)]
    results, pending = [], []
    for job in jobs:
        path = output / f"eval_{job[0]}_{job[1]}_{job[2]}.json"
        if options.resume and path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append(job)
    with ProcessPoolExecutor(max_workers=options.workers, mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(evaluate, job) for job in pending]):
            result = future.result()
            results.append(result)
            save_json(output / "evaluation.json", results)
            print(f"EVAL {len(results)}/{len(jobs)} {result['policy']} {result['score_left']}:{result['score_right']}", flush=True)
    summary = summarize(results, training, options)
    summary["pipeline_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    print(f"COMPLETE: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
