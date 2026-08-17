"""Baseline coaches — the ranking any learned policy must beat.

Four of them, matching the algorithm table in plan section 3.5:

* `RandomCoach`  — samples a legal tactic each epoch. The floor.
* `StaticCoach`  — picks one plan at kickoff and never changes. Surprisingly
  strong, and the honest control for "does adapting actually help?".
* `GreedyCoach`  — hand-written heuristic: chase when behind, shut up shop when
  ahead, rest legs when tired. What a competent human would script.
* `TabularQCoach` — Q-learning over the *declared* abstraction in
  config/train.yaml. It will lose, and why it loses is the report paragraph:
  the abstraction throws away tempo, per-player fatigue and — critically — the
  belief *entropy*, keeping only the MAP archetype, so it cannot hedge under
  uncertainty the way the deep policy can.

All expose the same `act(obs, env) -> (n_envs, 5)` interface so the evaluation
harness can treat them and trained policies identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from pitchsim.types import HEAD_SIZES, Tactic

ROOT = Path(__file__).resolve().parent.parent
HOME, AWAY = 0, 1


def _masked_sample(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """Sample each head uniformly from its legal values, given the flat mask."""
    n = mask.shape[0]
    out = np.zeros((n, len(HEAD_SIZES)), dtype=np.int64)
    offset = 0
    for head, size in enumerate(HEAD_SIZES):
        block = mask[:, offset:offset + size]
        weights = block / np.maximum(block.sum(axis=1, keepdims=True), 1)
        draws = rng.random((n, 1))
        out[:, head] = (weights.cumsum(axis=1) < draws).sum(axis=1).clip(0, size - 1)
        offset += size
    return out


class BaseCoach:
    name = "base"

    def reset(self, n_envs: int) -> None:
        pass

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        raise NotImplementedError


class RandomCoach(BaseCoach):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return _masked_sample(self.rng, env.action_masks())


class StaticCoach(BaseCoach):
    """One plan, chosen at kickoff, held for ninety minutes."""

    name = "static"

    def __init__(self, tactic: Tactic | None = None, seed: int = 0) -> None:
        self.tactic = tactic or Tactic.from_names()
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return np.tile(self.tactic.as_array(), (env.num_envs, 1))


class GreedyCoach(BaseCoach):
    """A competent human's script. No learning, but genuinely hard to beat."""

    name = "greedy"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        n = env.num_envs
        score_diff = env.engine.score[:, HOME] - env.engine.score[:, AWAY]
        minute = env.engine.minutes()
        stamina = env.engine.stamina[:, HOME].mean(axis=1)

        actions = np.tile(Tactic.from_names().as_array(), (n, 1))

        chasing = (score_diff < 0) & (minute > 55)
        actions[chasing] = Tactic.from_names(
            "4-2-3-1", "high", "gegenpress", "direct").as_array()

        holding = (score_diff > 0) & (minute > 70)
        actions[holding] = Tactic.from_names(
            "5-4-1", "low", "contain", "patient").as_array()

        # Spend a substitution once the legs have gone and it is legal to do so.
        mask = env.action_masks()
        sub_offset = sum(HEAD_SIZES[:4])
        can_sub = mask[:, sub_offset + 1]
        tired = (stamina < 0.55) & can_sub
        actions[tired, 4] = 1
        return actions


class TabularQCoach(BaseCoach):
    """Q-learning over the abstraction declared in config/train.yaml.

    The discretisation is specified rather than improvised — "we tried tabular
    and it was bad" earns nothing without saying what the state space *was*.
    """

    name = "tabular_q"

    def __init__(self, seed: int = 0, epsilon: float = 0.05) -> None:
        cfg = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
        q = cfg["tabular_q"]
        self.score_bins = np.array(q["score_diff_bins"], dtype=np.float64)
        self.phase_bins = np.array(q["phase_bins_minutes"], dtype=np.float64)
        self.stamina_bins = np.array(q["stamina_bins"], dtype=np.float64)
        self.alpha = float(q["learning_rate"])
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.n_states = 5 * 3 * 3 * 6
        self.n_actions = HEAD_SIZES[0] * HEAD_SIZES[2]      # formation x press
        self.q = np.zeros((self.n_states, self.n_actions))
        self._prev = None

    # -- abstraction ---------------------------------------------------

    def encode(self, env) -> np.ndarray:
        score_diff = env.engine.score[:, HOME] - env.engine.score[:, AWAY]
        minute = env.engine.minutes()
        stamina = env.engine.stamina[:, HOME].mean(axis=1)
        archetype = np.argmax(env.belief, axis=1)

        s = np.clip(np.digitize(score_diff, [-1.5, -0.5, 0.5, 1.5]), 0, 4)
        p = np.clip(np.digitize(minute, self.phase_bins[:-1]), 0, 2)
        f = np.clip(np.digitize(stamina, self.stamina_bins), 0, 2)
        return ((s * 3 + p) * 3 + f) * 6 + archetype

    def decode_action(self, flat: np.ndarray, env) -> np.ndarray:
        formation = flat // HEAD_SIZES[2]
        press = flat % HEAD_SIZES[2]
        actions = np.tile(Tactic.from_names().as_array(), (env.num_envs, 1))
        actions[:, 0] = formation
        actions[:, 2] = press
        return actions

    # -- policy --------------------------------------------------------

    def act(self, obs, env):
        state = self.encode(env)
        greedy = np.argmax(self.q[state], axis=1)
        explore = self.rng.random(env.num_envs) < self.epsilon
        flat = np.where(explore,
                        self.rng.integers(0, self.n_actions, env.num_envs),
                        greedy)
        self._prev = (state, flat)
        return self.decode_action(flat, env)

    def learn(self, reward: np.ndarray, env, gamma: float = 0.97) -> None:
        if self._prev is None:
            return
        state, action = self._prev
        nxt = self.encode(env)
        target = reward + gamma * self.q[nxt].max(axis=1)
        self.q[state, action] += self.alpha * (target - self.q[state, action])


def make(name: str, seed: int = 0) -> BaseCoach:
    return {
        "random": RandomCoach,
        "static": StaticCoach,
        "greedy": GreedyCoach,
        "tabular_q": TabularQCoach,
    }[name](seed=seed)
