"""Adapter unit tests do not require RoboCup, WSL or native processes."""
from types import SimpleNamespace

import numpy as np
import pytest

from envs.robocup_coach_env import coach_actions, restore_fatigue_args, RoboCupCoachEnv


def fixture_positions():
    players = np.array([[-49, 0], [-20, -6], [-20, 6], [-15, -18], [-15, 18],
                        [-8, 0], [-4, -8], [-4, 8], [10, -20], [10, 20], [15, 0]], float)
    mask = np.ones((11, 19), int)
    mask[:, 17:] = 0
    return players, np.zeros(2), mask


def test_baseline_restores_native_helios_delegation():
    assert np.all(coach_actions(*fixture_positions(), "helios") == 17)


def test_line_choices_change_direction_without_moving_keeper():
    low = coach_actions(*fixture_positions(), "line_low")
    high = coach_actions(*fixture_positions(), "line_high")
    assert low[0] == high[0] == 17
    assert np.all(low[1:5] == 15)
    assert np.all(high[1:5] == 16)


def test_press_uses_one_chaser_and_separate_cover():
    high = coach_actions(*fixture_positions(), "press_high")
    low = coach_actions(*fixture_positions(), "press_low")
    assert np.sum(high == 2) == np.sum(low == 2) == 1
    assert np.sum(high != 17) == 3
    assert not np.array_equal(high, low)
    assert high[0] == 17


@pytest.mark.parametrize("policy", ["line_high", "press_high", "press_low"])
def test_in_possession_retains_native_football(policy):
    assert np.all(coach_actions(*fixture_positions(), policy, kickable=np.ones(11)) == 17)


def test_masked_movement_falls_back_to_helios_not_arbitrary_skill():
    players, ball, mask = fixture_positions()
    mask[:] = 0
    assert np.all(coach_actions(players, ball, mask, "line_low") == 17)


def test_fatigue_filter_preserves_other_native_options():
    assert restore_fatigue_args(["rcssserver", "--server::stamina_max=1e9",
                                "--server::effort_min=1", "--server::random_seed=7",
                                "--player::player_types=1"]) == [
        "rcssserver", "--server::random_seed=7", "--player::player_types=1"]


class FakeBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.count = 0
        self.closed = False
        players, ball, mask = fixture_positions()
        self.mask = mask
        player = SimpleNamespace(obs=lambda: np.array([0, 0, 0, 0, 6000, 0]),
                                 is_kickable=lambda: False)
        self.agents = SimpleNamespace(
            player_list=[player] * 22,
            coach=SimpleNamespace(players=lambda: players, ball=lambda: ball,
                                  cycle=lambda: self.count))
        self.runtime = SimpleNamespace(all_procs_live=lambda: not self.closed)

    def reset(self):
        return {}

    def get_state(self):
        return np.zeros(136)

    def get_avail_actions(self):
        return self.mask

    def step(self, actions):
        self.count += 1
        done = self.count >= self.cfg["episode_limit"]
        return 1.0, done, {"terminal_reason": "timeout" if done else "running"}

    def close(self):
        self.closed = True


def test_gym_contract_stamina_action_repeat_and_truncation():
    env = RoboCupCoachEnv({"episode_limit": 3}, decision_steps=2, backend_factory=FakeBackend)
    with pytest.raises(RuntimeError):
        env.step(0)
    obs, info = env.reset(seed=12)
    assert env.observation_space.contains(obs)
    assert np.all(obs[136:147] == .75)
    assert env.backend.cfg["seed"] == 12
    assert not info["deterministic_reset"]
    with pytest.raises(ValueError):
        env.step(5)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 2 and not terminated and not truncated
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 1 and not terminated and truncated and obs[-1] == 1
    with pytest.raises(RuntimeError):
        env.step(0)
    previous = env.backend
    env.reset()
    assert previous.closed
    env.close()
    env.close()


def test_native_failure_is_rejected_not_a_training_transition():
    env = RoboCupCoachEnv({"episode_limit": 3}, backend_factory=FakeBackend)
    env.reset()
    env.backend.runtime.all_procs_live = lambda: False
    with pytest.raises(RuntimeError, match="do not train"):
        env.step(0)
    env.close()


def test_reset_failure_closes_started_backend():
    created = []

    class FailedReset(FakeBackend):
        def reset(self):
            created.append(self)
            raise RuntimeError("reset failed")

    env = RoboCupCoachEnv({"episode_limit": 3}, backend_factory=FailedReset)
    with pytest.raises(RuntimeError, match="reset failed"):
        env.reset()
    assert created[0].closed
    assert env.backend is None
