"""Experimental, isolated Gymnasium coach adapter for the audited R2DRL build.

This does not replace FootballCoach-v0. Native execution requires Linux and the
separately built RoboCup/HELIOS bundle. Actions are managerial skill delegation,
not native coach protocol messages. No simulator state is changed during play.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import gymnasium as gym
import numpy as np


POLICIES = ("helios", "line_low", "line_high", "press_low", "press_high")

# Remove the launcher's fatigue-disabling overrides; native server defaults
# (8000 stamina, finite capacity, variable effort/recovery) then apply. Equal
# player types and the bundled noise settings are deliberately retained.
FATIGUE_OVERRIDES = {
    "stamina_capacity", "stamina_max", "stamina_inc_max", "recover_init",
    "recover_min", "recover_dec", "effort_init", "effort_min", "effort_dec",
    "effort_inc",
}


def restore_fatigue_args(args):
    return [arg for arg in args if not (
        str(arg).startswith("--server::")
        and str(arg).split("::", 1)[1].split("=", 1)[0] in FATIGUE_OVERRIDES
    )]


@contextmanager
def native_launch_settings():
    """Scope the launch adjustment to this adapter; do not edit vendor files.

Not thread safe: separate native environments must use separate OS processes.
"""
    from robocup2d.process import launcher
    original = launcher.popen_logged

    def launch(args, **kwargs):
        if any(str(arg).startswith("--server::stamina_max=") for arg in args):
            args = restore_fatigue_args(args)
        return original(args, **kwargs)

    with patch.object(launcher, "popen_logged", launch):
        yield


def coach_actions(players, ball, mask, policy, kickable=None):
    """Translate an instruction to supported Base skills for the left team.

Action 17 is intentionally hidden by R2DRL's *learning* mask, but its C++
executor explicitly supports it and uses it for the opponent. We enable only
that safe delegation action; all movement/intercept skills retain native masks.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    players = np.asarray(players)
    ball = np.asarray(ball)
    mask = np.asarray(mask)
    if players.shape != (11, 2) or ball.shape != (2,) or mask.shape != (11, 19):
        raise ValueError("Expected 11 player positions, one ball and a (11,19) mask")
    actions = np.full(11, 17, dtype=np.int64)
    kickable = np.zeros(11, dtype=bool) if kickable is None else np.asarray(kickable, bool)
    if policy == "helios" or kickable.any():
        return actions  # retain native passing, shooting and support in possession

    distances = np.linalg.norm(players - ball, axis=1)
    nearest = np.argsort(distances[1:], kind="stable") + 1

    def move_to(player, target):
        delta = np.asarray(target) - players[player]
        if np.linalg.norm(delta) < 1.5:
            return
        skill = ((16 if delta[0] > 0 else 15) if abs(delta[0]) >= abs(delta[1])
                 else (13 if delta[1] > 0 else 14))
        if mask[player, skill]:
            actions[player] = skill

    if policy.startswith("line_"):
        target_x = -30.0 if policy == "line_low" else -5.0
        for player in range(1, 5):
            # A defender close to the ball may defend it instead of abandoning it.
            if player == nearest[0] and distances[player] < 8:
                continue
            move_to(player, (target_x, players[player, 1]))
    else:
        primary = nearest[0]
        if mask[primary, 2]:
            actions[primary] = 2
        else:
            move_to(primary, ball)
        # One chaser, two distinct goal-side covering positions: no three-player
        # interception pile-up. Contain holds those same players further back.
        depth = 3.0 if policy == "press_high" else 12.0
        cover = sorted(nearest[1:3], key=lambda p: players[p, 1])
        for player, offset in zip(cover, (-5.0, 5.0)):
            target = np.clip(ball + (-depth, offset), (-50, -31), (50, 31))
            move_to(player, target)
    assert all(a == 17 or mask[p, a] for p, a in enumerate(actions))
    return actions


class RoboCupCoachEnv(gym.Env):
    """Five experimental coaching choices, held for ``decision_steps`` ticks.

Observation: normalized public global state (136), own measured stamina (11),
elapsed horizon fraction (1). No opponent style label is exposed. Seed forwarding
does NOT promise deterministic trajectories in the multiprocess native engine.
Native episode cutoffs are Gymnasium truncations, not regulation full time.
"""
    metadata = {"render_modes": []}

    def __init__(self, config, decision_steps=20, backend_factory=None):
        if decision_steps < 1:
            raise ValueError("decision_steps must be positive")
        self.config = dict(config)
        self.decision_steps = int(decision_steps)
        self.backend_factory = backend_factory
        self.backend = None
        self._done = True
        self.elapsed_steps = 0
        self.action_space = gym.spaces.Discrete(len(POLICIES))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (148,), np.float32)

    def _observation(self):
        stamina = self.stamina() / 8000.0
        progress = [self.elapsed_steps / self.config["episode_limit"]]
        obs = np.concatenate((self.backend.get_state(), stamina, progress)).astype(np.float32)
        if obs.shape != (148,) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid RoboCup observation")
        return obs

    def stamina(self):
        return np.array([p.obs()[4] for p in self.backend.agents.player_list[:11]],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported")
        self.close()
        if seed is not None:
            self.config["seed"] = int(seed)
        factory = self.backend_factory
        if factory is None:
            from robocup2d import Robocup2dEnv
            factory = Robocup2dEnv
        try:
            with native_launch_settings() if self.backend_factory is None else _no_launch():
                self.backend = factory(cfg=self.config)
                info = self.backend.reset()
            self.elapsed_steps = 0
            self._done = False
            return self._observation(), dict(info, deterministic_reset=False)
        except BaseException:
            self.close()
            raise

    def step(self, action):
        if self._done or self.backend is None:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Action must belong to action_space")
        reward_sum = 0.0
        start_cycle = int(self.backend.agents.coach.cycle())
        action_counts = np.zeros(19, dtype=int)
        for tick in range(self.decision_steps):
            agents = self.backend.agents
            low_level = coach_actions(
                agents.coach.players()[:11, :2], agents.coach.ball()[:2],
                self.backend.get_avail_actions(), POLICIES[int(action)],
                [p.is_kickable() for p in agents.player_list[:11]],
            )
            reward, done, info = self.backend.step(low_level)
            if not np.isfinite(reward):
                raise RuntimeError("Non-finite RoboCup reward")
            reward_sum += float(reward)
            self.elapsed_steps += 1
            action_counts += np.bincount(low_level, minlength=19)
            if done:
                break
        reason = info.get("terminal_reason", "running")
        if reason == "restart" or not self.backend.runtime.all_procs_live():
            self._done = True
            raise RuntimeError("Native simulation failed; do not train on this transition")
        self._done = bool(done)
        truncated = bool(done and reason == "timeout")
        info = dict(info, policy=POLICIES[int(action)], native_steps=tick + 1,
                    cycles_advanced=int(self.backend.agents.coach.cycle()) - start_cycle,
                    player_action_counts=action_counts.tolist(),
                    mean_stamina=float(self.stamina().mean()))
        return self._observation(), reward_sum, bool(done and not truncated), truncated, info

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        self._done = True


@contextmanager
def _no_launch():
    yield
