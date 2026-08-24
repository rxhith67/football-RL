"""MaskablePPO with a genuine semi-Markov discount.

Decision epochs have variable dwell times tau_k, so a transition should be
discounted by gamma^tau, not by a flat gamma (Sutton, Precup & Singh 1999; plan
section 3.3). Stable-Baselines3 hard-codes a single `gamma` in GAE, so the
advantage computation is overridden here.

Only `compute_returns_and_advantage` needs replacing. The dwell times come from
the environment's `tau_log`, which records one array per step; that avoids
copying SB3's eighty-line `collect_rollouts` just to thread one extra tensor
through it.

Flat-gamma is the ablation (design note 5), available via `smdp=False`.
"""

from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer


class SMDPMaskableRolloutBuffer(MaskableRolloutBuffer):
    """GAE with a per-transition discount gamma^tau."""

    def __init__(self, *args, gamma_per_minute: float = 0.995, env=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_per_minute = gamma_per_minute
        self.env = env

    def _taus(self) -> np.ndarray:
        """(n_steps, n_envs) dwell times for the transitions in this rollout."""
        log = getattr(self.env, "tau_log", None)
        if not log or len(log) < self.buffer_size:
            # Fall back to the nominal epoch length; keeps training runnable if
            # the env was not instrumented rather than silently mis-discounting.
            return np.full((self.buffer_size, self.n_envs), 5.0)
        taus = np.stack(log[-self.buffer_size:])
        return taus.reshape(self.buffer_size, self.n_envs)

    def compute_returns_and_advantage(self, last_values, dones) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        taus = self._taus()

        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            # The SMDP discount: gamma raised to the dwell time of THIS epoch.
            gamma = self.gamma_per_minute ** taus[step]

            delta = (self.rewards[step]
                     + gamma * next_values * next_non_terminal
                     - self.values[step])
            last_gae = delta + gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

        # Clear the log so the next rollout reads its own dwell times.
        if getattr(self.env, "tau_log", None) is not None:
            self.env.tau_log.clear()


class SMDPMaskablePPO(MaskablePPO):
    """MaskablePPO whose rollout buffer discounts by gamma^tau."""

    def __init__(self, *args, gamma_per_minute: float = 0.995, smdp: bool = True, **kwargs):
        self.gamma_per_minute = gamma_per_minute
        self.smdp = smdp
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if not self.smdp:
            return                       # flat-gamma ablation keeps SB3's buffer

        self.rollout_buffer = SMDPMaskableRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            gamma_per_minute=self.gamma_per_minute,
            env=self.env,
        )

    def set_discounting(self, smdp: bool) -> None:
        """Apply the SMDP/flat choice to an ALREADY-CONSTRUCTED model.

        `load()` restores `smdp` from the checkpoint's saved attributes, so a
        curriculum stage that asks for flat-gamma while initialising from a
        gamma^tau parent would otherwise inherit the parent's setting in
        silence: same discount, same gradients, a bit-identical policy saved
        under a tag claiming it was the ablation. An ablation that cannot
        differ from its control is worse than no ablation, so the switch is
        applied explicitly here rather than trusted to `__init__`.

        `_setup_model` cannot be reused for this — it rebuilds the policy and
        would discard the loaded weights.
        """
        if smdp == self.smdp:
            return
        self.smdp = smdp
        common = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        if smdp:
            self.rollout_buffer = SMDPMaskableRolloutBuffer(
                self.n_steps, gamma_per_minute=self.gamma_per_minute,
                env=self.env, **common)
        else:
            self.rollout_buffer = MaskableRolloutBuffer(self.n_steps, **common)
