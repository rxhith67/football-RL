"""The coach's SMDP environment — Gymnasium/SB3 interface over PITCHSIM.

Implemented as a **VecEnv** rather than a single-match `gym.Env`, because
PITCHSIM is batched: driving it one match at a time would discard the entire
point of the vectorisation and miss the throughput target (design note 3.2).

Semi-Markov structure (design note 3.3)
---------------------------------------
The coach acts at scheduled five-minute marks *and* on triggers (goal, half-time,
red card), so the dwell time tau_k between decisions varies. That makes this a
genuine SMDP, and transitions must be discounted by gamma^tau rather than a flat
gamma. `tau` is returned in `info` for every step; `agents.smdp_ppo` consumes it.

Reward
------
    R_k = R_terminal * 1[final] + (Phi(s_{k+1}) - Phi(s_k)) - lambda_f * f_k - lambda_c * c_k

with Phi(s) = w * (xG_for - xG_against), a potential-based shaping function whose
policy invariance follows from Ng, Harada & Russell (1999). `w` is set by the
scale argument in design note 3.3: shaping telescopes to w * final xG difference,
so w ~= 0.3 keeps typical shaping below the terminal +-1.

Partial observability
---------------------
The opponent's tactic vector is NEVER in the observation — only its statistical
footprint, via `WindowStats` and the belief posterior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from envs.archetypes import N_ARCHETYPES, ScriptedCoaches, archetype_table
from envs.belief import BeliefFilter
from pitchsim import controller as ctrl
from pitchsim.engine import BatchedEngine
from pitchsim.types import HEAD_SIZES, MASK_LENGTH, Tactic

ROOT = Path(__file__).resolve().parent.parent

HOME, AWAY = 0, 1
BELIEF_MODES = ("bayesian", "blind", "oracle")


def load_train_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))


class VecCoachEnv(VecEnv):
    """`n_envs` matches advanced in lockstep, one coaching decision per step.

    Subclasses SB3's `VecEnv` so the algorithm treats it as an already-vectorised
    environment and does not try to wrap it. The SMDP bookkeeping is kept in this
    one file so the gamma^tau story is readable end to end.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_envs: int = 32,
        belief_mode: str = "bayesian",
        opponent: str = "archetype",     # archetype | fixed | policy
        seed: int | None = None,
        config: dict | None = None,
    ) -> None:
        if belief_mode not in BELIEF_MODES:
            raise ValueError(f"belief_mode must be one of {BELIEF_MODES}")

        self.num_envs = n_envs
        self.belief_mode = belief_mode
        self.opponent_mode = opponent
        self.train_cfg = config or load_train_config()

        _, physics = ctrl.load_config()
        self.engine = BatchedEngine(batch=n_envs, config=physics)

        self.epoch_minutes = 5.0
        self.match_minutes = float(physics["time"]["match_minutes"])
        smdp = self.train_cfg["smdp"]
        self.gamma_per_minute = float(smdp["gamma_per_minute"])
        self.entropy_jump = float(smdp["belief_entropy_jump_threshold"])

        reward = self.train_cfg["reward"]
        self.w_shaping = float(reward["shaping_w"])
        self.lambda_fatigue = float(reward["lambda_fatigue"])
        self.fatigue_threshold = float(reward["fatigue_threshold"])
        self.lambda_churn = float(reward["lambda_churn"])

        self.subs_cfg = ctrl.load_config()[0]["substitutions"]

        obs_dim = int(self.train_cfg["observation"]["total"])
        super().__init__(
            num_envs=n_envs,
            observation_space=spaces.Box(-5.0, 5.0, (obs_dim,), dtype=np.float32),
            action_space=spaces.MultiDiscrete(list(HEAD_SIZES)),
        )
        self.render_mode = None

        self.filter: BeliefFilter | None = None
        if belief_mode == "bayesian":
            self.filter = BeliefFilter.load()

        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._actions: np.ndarray | None = None
        self.opponent_policy = None          # set externally for self-play

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**31 - 1))
        _, _, base = archetype_table()

        self.assignment = self.rng.integers(0, N_ARCHETYPES, size=self.num_envs)
        self.coaches = ScriptedCoaches(self.assignment)
        self.coaches.reset()

        tactics = np.empty((self.num_envs, 2, 5), dtype=np.int64)
        tactics[:, HOME] = Tactic.from_names().as_array()
        tactics[:, AWAY] = base[self.assignment]
        self.engine.reset(seed=seed, tactics=tactics)

        self.belief = (self.filter.reset(self.num_envs) if self.filter is not None
                       else np.full((self.num_envs, N_ARCHETYPES), 1.0 / N_ARCHETYPES))
        self._prev_entropy = np.zeros(self.num_envs)
        self._prev_potential = np.zeros(self.num_envs)
        self._prev_tactic = tactics[:, HOME].copy()
        self._epoch = 0
        #: Dwell times of the most recent steps, consumed by the SMDP buffer to
        #: discount each transition by gamma^tau. Cleared per rollout.
        self.tau_log: list[np.ndarray] = getattr(self, 'tau_log', [])
        self._episode_reward = np.zeros(self.num_envs)
        self._half_score = np.zeros((self.num_envs, 2), dtype=np.int64)
        return self._observe()

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs, 5)

    def step_wait(self):
        actions = self._actions
        assert actions is not None, "step_async must be called before step_wait"

        tactics = self.engine.tactics.copy()

        # --- apply the coach's decision (home) ---
        churn = np.array([
            Tactic.from_array(actions[i]).style_differs_from(
                Tactic.from_array(self._prev_tactic[i]))
            for i in range(self.num_envs)
        ], dtype=np.float64)
        tactics[:, HOME] = actions
        self._apply_substitutions(actions)

        # --- opponent decides too ---
        minute = self.engine.minutes()
        score_diff_away = self.engine.score[:, AWAY] - self.engine.score[:, HOME]
        if self.opponent_mode == "archetype":
            tactics[:, AWAY] = self.coaches.decide(minute, score_diff_away, self.rng)
        elif self.opponent_policy is not None:
            tactics[:, AWAY] = self.opponent_policy(self._observe(side=AWAY))

        self.engine.set_tactics(tactics)

        # --- advance to the next decision epoch (or an early trigger) ---
        start = self.engine.minutes().copy()
        target = np.minimum(start + self.epoch_minutes, self.match_minutes)
        self.engine.run_until(float(target.max()) * 60.0, stop_on_goal=True)
        tau = np.maximum(self.engine.minutes() - start, 1e-3)

        self._epoch += 1
        done = self.engine.done()

        # --- reward ---
        potential = self.w_shaping * (self.engine.counters.xg[:, HOME]
                                      - self.engine.counters.xg[:, AWAY])
        shaping = potential - self._prev_potential
        self._prev_potential = potential

        deficit = np.clip(self.fatigue_threshold - self.engine.stamina[:, HOME], 0.0, None)
        fatigue = np.mean(deficit**2, axis=1)

        reward = shaping - self.lambda_fatigue * fatigue - self.lambda_churn * churn

        diff = self.engine.score[:, HOME] - self.engine.score[:, AWAY]
        reward = reward + done * np.sign(diff)

        self._prev_tactic = actions.copy()
        self._episode_reward += reward

        self.tau_log.append(tau.copy())
        obs = self._observe()
        infos = [{"tau": float(tau[i]), "epoch": self._epoch} for i in range(self.num_envs)]

        if done.all():
            for i in range(self.num_envs):
                infos[i]["episode"] = {
                    "r": float(self._episode_reward[i]),
                    "l": self._epoch,
                    "score": (int(self.engine.score[i, HOME]),
                              int(self.engine.score[i, AWAY])),
                    "points": 3 if diff[i] > 0 else (1 if diff[i] == 0 else 0),
                    "belief_correct": bool(
                        np.argmax(self.belief[i]) == self.assignment[i]),
                }
            obs = self.reset()

        return obs, reward.astype(np.float32), done.copy(), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    # ------------------------------------------------------------------

    def _apply_substitutions(self, actions: np.ndarray) -> None:
        """Restore stamina at one slot, spending a sub and a window."""
        wants = actions[:, 4] > 0
        legal = (wants
                 & (self.engine.subs_used[:, HOME] < int(self.subs_cfg["max_subs"]))
                 & (self.engine.windows_used[:, HOME] < int(self.subs_cfg["max_windows"]))
                 & (self.engine.minutes() >= float(self.subs_cfg["earliest_minute"])))
        if not legal.any():
            return
        idx = np.flatnonzero(legal)
        slot = self.rng.integers(1, 11, size=idx.size)
        self.engine.stamina[idx, HOME, slot] = float(self.subs_cfg["stamina_restored"])
        self.engine.subs_used[idx, HOME] += 1
        self.engine.windows_used[idx, HOME] += 1

    # ------------------------------------------------------------------
    # Observation and mask
    # ------------------------------------------------------------------

    def _observe(self, side: int = HOME) -> np.ndarray:
        other = 1 - side
        stats = self.engine.window_stats(consume=(side == HOME))

        if side == HOME and self.filter is not None:
            self.belief = self.filter.update(stats[:, AWAY])
        elif side == HOME and self.belief_mode == "oracle":
            self.belief = np.eye(N_ARCHETYPES)[self.assignment]
        elif side == HOME and self.belief_mode == "blind":
            self.belief = np.full((self.num_envs, N_ARCHETYPES), 1.0 / N_ARCHETYPES)

        minute = self.engine.minutes()
        match_state = np.stack([
            np.clip((self.engine.score[:, side] - self.engine.score[:, other]) / 3.0, -2, 2),
            (self.match_minutes - minute) / self.match_minutes,
            (int(self.subs_cfg["max_subs"]) - self.engine.subs_used[:, side]) / 5.0,
            (int(self.subs_cfg["max_windows"]) - self.engine.windows_used[:, side]) / 3.0,
            self.engine.red_cards[:, side].astype(np.float64),
        ], axis=1)

        stamina = self.engine.stamina[:, side]
        squad = np.concatenate([
            stamina,
            (np.sum(stamina < self.fatigue_threshold, axis=1) / 11.0)[:, None],
            (self.engine.yellow_cards[:, side] / 5.0)[:, None],
        ], axis=1)

        scale = np.array([1.0, 8.0, 8.0, 1.5, 1.5, 1.0, 30.0, 1.0, 40.0])
        own_form = np.clip(stats[:, side] / scale, 0.0, 1.5)
        opp_form = np.clip(stats[:, other] / scale, 0.0, 1.5)

        tactic_onehot = np.zeros((self.num_envs, 14))
        offset = 0
        for head, size in enumerate(HEAD_SIZES[:4]):
            tactic_onehot[np.arange(self.num_envs), offset + self.engine.tactics[:, side, head]] = 1.0
            offset += size

        belief = self.belief if side == HOME else np.full(
            (self.num_envs, N_ARCHETYPES), 1.0 / N_ARCHETYPES)

        obs = np.concatenate(
            [match_state, squad, own_form, opp_form, tactic_onehot, belief], axis=1
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """FLAT boolean array of length 20 per env (design note 3.3).

        `sb3_contrib`'s MaskableMultiCategoricalDistribution expects one flat
        array of length sum(nvec), which it splits internally — the original draft
        returned a list of per-head arrays, which silently mis-masks.
        """
        mask = np.ones((self.num_envs, MASK_LENGTH), dtype=bool)
        sub_offset = sum(HEAD_SIZES[:4])

        minute = self.engine.minutes()
        can_sub = ((self.engine.subs_used[:, HOME] < int(self.subs_cfg["max_subs"]))
                   & (self.engine.windows_used[:, HOME] < int(self.subs_cfg["max_windows"]))
                   & (minute >= float(self.subs_cfg["earliest_minute"])))
        mask[:, sub_offset + 1:] = can_sub[:, None]

        # A back three needs three fit centre-backs.
        _, _, centre_backs = ctrl.formation_tables()
        fit = self.engine.stamina[:, HOME] > 0.25
        for f, backs in enumerate(centre_backs):
            if len(backs) >= 3:
                mask[:, f] = fit[:, backs].sum(axis=1) >= 3

        mask[:, sub_offset] = True          # "no substitution" is always legal
        return mask

    # -- SB3 VecEnv compatibility --------------------------------------

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """MaskablePPO discovers masks through `env_method("action_masks")`."""
        if method_name == "action_masks":
            return list(self.action_masks())
        raise NotImplementedError(method_name)

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
