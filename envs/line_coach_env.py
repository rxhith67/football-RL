"""Submission-sized finite-horizon defensive-line task over RoboCup/HELIOS."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.robocup_coach_env import RoboCupCoachEnv

ACTION_NAMES = ("Normal play", "Low line", "High line")
GAMMA = 0.99
SHAPING_SCALE = 0.05


class LineCoachEnv(gym.Env):
    """Three choices; 20 native steps per decision; no regulation-match claim.

The fixed horizon is the end of this task (termination), not an interruption of
a continuing match. Remaining time is observed. Reward is goal difference plus
potential shaping, with zero terminal potential; evaluation reports raw goals.
"""
    metadata = {"render_modes": []}

    def __init__(self, config, decision_steps=20, record=False, backend_factory=None):
        self.core = RoboCupCoachEnv(config, decision_steps=1, backend_factory=backend_factory)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (149,), np.float32)
        if decision_steps < 1:
            raise ValueError("decision_steps must be positive")
        self.decision_steps = decision_steps
        self.record = record
        self.frames = []
        self.decisions = []
        self.score = [0, 0]
        self.episode_count = 0
        self.base_seed = int(config.get("seed", 0))
        self.ended = True

    def _observation(self, obs):
        return np.append(obs, np.clip((self.score[0] - self.score[1]) / 10, -1, 1)).astype(np.float32)

    def _potential(self):
        return float(np.clip(self.core.backend.agents.coach.ball()[0] / 52.5, -1, 1))

    def _frame(self, action):
        if not self.record:
            return
        coach = self.core.backend.agents.coach
        self.frames.append({"step": self.core.elapsed_steps, "cycle": int(coach.cycle()),
                            "ball": np.round(coach.ball()[:2], 3).tolist(),
                            "players": np.round(coach.players()[:, :2], 3).tolist(),
                            "score": self.score.copy(), "action": int(action),
                            "stamina": round(float(self.core.stamina().mean()) / 8000, 4)})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = int(seed)
            self.episode_count = 0
        episode_seed = self.base_seed + self.episode_count
        obs, info = self.core.reset(seed=episode_seed, options=options)
        self.episode_count += 1
        self.score = [int(info.get("score_left", 0)), int(info.get("score_right", 0))]
        self.frames, self.decisions = [], []
        self.ended = False
        self._frame(0)
        return self._observation(obs), dict(info, episode_seed=episode_seed)

    def step(self, action):
        if self.ended:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Only Normal play, Low line and High line are available")
        potential_before = self._potential()
        start_step = self.core.elapsed_steps
        initial_score = self.score.copy()
        if self.record:
            self.decisions.append({"step": start_step, "action": int(action)})
        for tick in range(self.decision_steps):
            obs, _, terminated, truncated, info = self.core.step(int(action))
            self.score = [int(info.get("score_left", 0)), int(info.get("score_right", 0))]
            self.ended = terminated or truncated
            if self.core.elapsed_steps % 2 == 0 or self.ended:
                self._frame(action)
            if self.ended:
                break
        goal_reward = (self.score[0] - initial_score[0]) - (self.score[1] - initial_score[1])
        potential_after = 0.0 if self.ended else self._potential()
        shaping = SHAPING_SCALE * (GAMMA * potential_after - potential_before)
        info = dict(info, goal_reward=goal_reward, shaping_reward=shaping,
                    task_horizon_reached=bool(truncated), native_steps=tick + 1,
                    score_left=self.score[0], score_right=self.score[1])
        return self._observation(obs), float(goal_reward + shaping), bool(self.ended), False, info

    def replay(self, controller):
        return {"schema_version": 1, "controller": controller, "actions": list(ACTION_NAMES),
                "horizon": self.core.config["episode_limit"], "decision_steps": self.decision_steps,
                "recording": "Native positions sampled every two action steps; playback interpolates positions.",
                "regulation_match": False, "score": self.score, "frames": self.frames,
                "decisions": self.decisions}

    def close(self):
        self.core.close()
        self.ended = True
