from types import SimpleNamespace
import numpy as np
import pytest

from envs.line_coach_env import LineCoachEnv, GAMMA, SHAPING_SCALE


class Backend:
    def __init__(self, cfg):
        self.cfg, self.count = cfg, 0
        self.closed = False
        self.players = np.zeros((22, 6))
        self.players[:, 0] = np.arange(22) - 30
        self.ball = np.zeros(4)
        player = SimpleNamespace(obs=lambda: np.array([0, 0, 0, 0, 8000, 0]), is_kickable=lambda: False)
        self.agents = SimpleNamespace(player_list=[player] * 22,
            coach=SimpleNamespace(ball=lambda: self.ball, players=lambda: self.players, cycle=lambda: self.count))
        self.runtime = SimpleNamespace(all_procs_live=lambda: not self.closed)

    def reset(self):
        return {"score_left": 0, "score_right": 0}

    def get_state(self):
        return np.zeros(136)

    def get_avail_actions(self):
        return np.ones((11, 19))

    def step(self, action):
        self.count += 1
        self.ball[0] = self.count
        done = self.count >= self.cfg["episode_limit"]
        return 999, done, {"score_left": int(done), "score_right": 0,
                            "terminal_reason": "timeout" if done else "running"}

    def close(self):
        self.closed = True


def test_reduced_actions_and_finite_horizon_reward():
    env = LineCoachEnv({"episode_limit": 4, "seed": 7}, decision_steps=2, record=True, backend_factory=Backend)
    with pytest.raises(RuntimeError):
        env.step(0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs) and obs.shape == (149,)
    assert info["episode_seed"] == 7
    with pytest.raises(ValueError):
        env.step(3)
    _, first, done, truncated, _ = env.step(2)
    assert not done and not truncated
    assert first == pytest.approx(SHAPING_SCALE * GAMMA * 2 / 52.5)
    obs, second, done, truncated, info = env.step(1)
    assert done and not truncated and info["task_horizon_reached"]
    assert second == pytest.approx(1 - SHAPING_SCALE * 2 / 52.5)
    assert obs[-1] == pytest.approx(.1)
    # Terminal potential cancels the prior potential in the discounted sum.
    assert first + GAMMA * second == pytest.approx(GAMMA)
    replay = env.replay("test")
    assert len(replay["frames"]) == 3 and len(replay["frames"][0]["players"]) == 22
    assert [d["action"] for d in replay["decisions"]] == [2, 1]
    assert not replay["regulation_match"]
    previous = env.core.backend
    _, info = env.reset()
    assert previous.closed and info["episode_seed"] == 8
    env.close()
