"""Isolated RoboCup/R2DRL integration probe. Run with the Linux test venv.

Does not import or modify the project's current football simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("ROBOCUP_BUNDLE", ROOT / "tmp/environment-comparison/R2DRL"))
NATIVE = BUNDLE / "R2DRL"
OUT = Path(os.environ.get("ROBOCUP_PROBE_OUTPUT", ROOT / "tmp/environment-comparison/robocup-results"))


def deadline(signum, frame):
    raise TimeoutError("RoboCup probe exceeded its wall-clock deadline")


def configuration(steps, seed, name):
    import yaml
    args = yaml.safe_load((NATIVE / "robocup2d/config/robocup.yaml").read_text())["env_args"]
    for key, relative in {
        "player_dir": "helios-base/src/player",
        "coach_dir": "helios-base/src/coach",
        "trainer_dir": "helios-base/src/trainer",
        "server_path": "rcssserver/build/rcssserver",
        "config_dir": "helios-base/src/formations-dt",
        "player_config": "helios-base/src/player.conf",
    }.items():
        args[key] = str(NATIVE / relative)
    args["lib_paths"] = [str(NATIVE / item) for item in (
        "librcsc/rcsc/.libs", "runtime_libs", "rcssserver/build/rcss/clang",
        "rcssserver/build/rcss/conf", "rcssserver/build/rcss/net", "rcssserver/build/rcss/gzip",
    )]
    args.update(n=11, init_n=11, episode_limit=steps, seed=seed,
                benchmark_mode="full_match", use_custom_start=False,
                terminate_on_goal=False, useMaxEpv=False, tb=False,
                text_logging=True, game_logging=True,
                logs_dir=str(OUT / name / "logs"), tb_log_dir=str(OUT / name / "tb"),
                reset_retries=0, wait_ready_timeout=20.0, playon_timeout=10.0,
                ports_wait_timeout=10.0, server_wait_seconds=20.0,
                auto_port_start=16000, auto_port_end=16100,
                epv_grid_file=str(NATIVE / "robocup2d/LaurieOnTracking/EPV_grid.csv"))
    return args


def managerial_actions(env, policy, index, steps):
    """Small controller prototype using legal player skills, never teleportation.

    Only the line and pressing interventions are represented. This is not the
    final formation/tempo/substitution controller proposed for the project.
    """
    import numpy as np
    mask = env.get_avail_actions()
    players = env.agents.coach.players()[:11, :2]
    ball = env.agents.coach.ball()[:2]

    def legal(player, preferred):
        """First action in ``preferred`` the env permits for this player; if
        none are permitted, the first legal action (never an illegal one)."""
        for candidate in preferred:
            if mask[player, candidate]:
                return int(candidate)
        available = np.flatnonzero(mask[player])
        return int(available[0]) if available.size else 18

    # Baseline every player: delegate to the HELIOS behaviour where the env
    # allows it (e.g. set plays restrict the mask), otherwise wait.
    actions = np.array([legal(p, (17, 18)) for p in range(11)], dtype=np.int64)

    def move_to(player, target):
        delta = np.asarray(target) - players[player]
        if np.linalg.norm(delta) < 1.0:
            actions[player] = legal(player, (18, 17))
            return
        candidate = (16 if delta[0] > 0 else 15) if abs(delta[0]) >= abs(delta[1]) else (13 if delta[1] > 0 else 14)
        actions[player] = legal(player, (candidate, 17, 18))

    if policy.startswith("line_"):
        high = policy == "line_high" or (policy == "line_switch" and index >= steps // 2)
        target_x = -5.0 if high else -30.0
        for defender in range(1, 5):
            move_to(defender, (target_x, players[defender, 1]))
    elif policy.startswith("press_"):
        count = 3 if policy == "press_high" else 1
        nearest = np.argsort(np.linalg.norm(players[1:] - ball, axis=1))[:count] + 1
        for player in nearest:
            if mask[player, 2]:
                actions[player] = 2
            else:
                move_to(player, ball)
    elif policy != "helios":
        raise ValueError(policy)
    assert np.all(mask[np.arange(11), actions]), "controller emitted an illegal action"
    return actions


def run_case(name, steps=300, seed=37, policy="helios"):
    import numpy as np
    from robocup2d import Robocup2dEnv

    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    args = configuration(steps, seed, name)
    os.environ["RCSS_CONF_DIR"] = str(target / "server-config")
    (target / "config.json").write_text(json.dumps(args, indent=2))
    result = {"name": name, "policy": policy, "requested_steps": steps, "config_seed": seed,
              "seed_forwarding_verified": False, "success": False}
    env = None
    states, cycles, rewards, step_times, actions_log = [], [], [], [], []
    started = time.perf_counter()
    signal.alarm(240)
    try:
        env = Robocup2dEnv(cfg=args)
        result["log_dir"] = env.runtime.log_dir
        result["run_id"] = env.runtime.run_id
        result["env_info"] = env.get_env_info()
        result["reset_info"] = env.reset()
        result["startup_seconds"] = time.perf_counter() - started
        print(f"{name}: reset completed, {result['env_info']}", flush=True)
        initial = env.agents.coach.state().copy()
        players = initial[4:].reshape(22, 6)
        result["initial_player_positions"] = players[:, :2].tolist()
        for index in range(steps):
            actions = managerial_actions(env, policy, index, steps)
            available = env.get_avail_actions()
            assert available.shape == (11, 19), available.shape
            assert np.all(available[np.arange(11), actions]), "illegal player action"
            tick = time.perf_counter()
            reward, done, info = env.step(actions)
            step_times.append(time.perf_counter() - tick)
            state = env.agents.coach.state().copy()
            assert np.isfinite(state).all() and np.isfinite(env.get_obs()).all()
            assert np.isfinite(reward), "non-finite reward"
            assert env.runtime.all_procs_live(), env.runtime.dead_proc_descriptions()
            states.append(state)
            cycles.append(env.agents.coach.cycle())
            rewards.append(reward)
            actions_log.append(actions)
            if (index + 1) % 100 == 0:
                print(f"{name}: {index + 1} steps, cycle {cycles[-1]}, score {info['score_left']}:{info['score_right']}", flush=True)
            if done:
                break
        states = np.asarray(states)
        player_states = states[:, 4:].reshape(-1, 22, 6)
        line = player_states[:, 1:5, 0].mean(axis=1)
        distances = np.linalg.norm(player_states[:, 1:11, :2] - states[:, None, :2], axis=2)
        result.update(completed_steps=len(states), terminal_info=info,
                      cycles_advanced=int(cycles[-1] - cycles[0]),
                      finite_outputs=True, all_processes_live=True,
                      step_seconds=float(sum(step_times)),
                      steps_per_second=float(len(states) / sum(step_times)),
                      p95_step_seconds=float(np.percentile(step_times, 95)),
                      trace_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
                      mean_defender_x=float(states[:, 4:].reshape(-1, 22, 6)[:, 1:5, 0].mean()),
                      defender_x_first_half=float(line[:len(line)//2].mean()),
                      defender_x_second_half=float(line[len(line)//2:].mean()),
                      defender_x_last_30_steps=float(line[-30:].mean()),
                      mean_players_within_8m_of_ball=float((distances < 8).sum(axis=1).mean()),
                      action_counts=np.bincount(np.asarray(actions_log).ravel(), minlength=19).tolist(),
                      ball_distance_travelled=float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()),
                      success=len(states) == steps and info["terminal_reason"] == "timeout")
        np.savez_compressed(target / "trace.npz", states=states, cycles=cycles,
                            rewards=rewards, actions=actions_log, step_times=step_times)
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        signal.alarm(0)
        if len(states) and not (target / "trace.npz").exists():
            np.savez_compressed(target / "trace.npz", states=states, cycles=cycles,
                                rewards=rewards, actions=actions_log, step_times=step_times)
            result["completed_steps_before_failure"] = len(states)
        if env is not None:
            env.close()
            result["cleanup_all_processes_stopped"] = not env.runtime.has_live_procs()
        result["total_seconds"] = time.perf_counter() - started
        (target / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--policy", choices=["helios", "line_low", "line_high", "line_switch", "press_low", "press_high"], default="helios")
    parser.add_argument("--suite", action="store_true")
    options = parser.parse_args()
    os.environ["ROBOCUP_ACTION_DEBUG"] = "0"
    sys.path.insert(0, str(NATIVE))
    signal.signal(signal.SIGALRM, deadline)
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "-C", str(BUNDLE), "rev-parse", "HEAD"], text=True).strip()
    (OUT / "provenance.json").write_text(json.dumps({"upstream": "https://github.com/open-starlab/R2DRL", "commit": revision, "python": sys.version, "torch": torch.__version__}, indent=2))
    if options.suite:
        results = []
        for name, policy in [("baseline_a", "helios"), ("baseline_b", "helios"),
                             ("line_low", "line_low"), ("line_high", "line_high"),
                             ("line_switch", "line_switch"), ("press_low", "press_low"),
                             ("press_high", "press_high")]:
            result = run_case(name, options.steps, options.seed, policy)
            results.append(result)
            (OUT / "suite.json").write_text(json.dumps(results, indent=2))
            if not result["success"]:
                return 1
        return 0
    result = run_case(options.name, options.steps, options.seed, options.policy)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
