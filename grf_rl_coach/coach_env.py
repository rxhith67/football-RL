"""Sequential RL coaching environment on the patched GRF engine.

The coach picks a team-level native tactical setting (defensive depth) every `decision_steps`
engine steps; all 11 left players stay on GRF's built-in AI (`builtin_ai`) the whole time.

Terminology
  engine step  : one call to GRF's env.step() (= 10 physics ticks = 100 ms of match clock)
  coach decision: one RL step of this environment (= `decision_steps` engine steps)

Run only against the patched engine (PYTHONPATH=$HOME/grf_native_poc/overlay).
"""
import random
from collections import deque

import gym
import numpy as np
from gym import spaces

import gfootball.env as football_env
from gfootball.env import football_action_set

SCENARIO = "11_vs_11_hard_stochastic"
AI = football_action_set.action_set_v2.index(football_action_set.action_builtin_ai)
MATCH_STEPS = 3001                    # GRF reports steps_left = 3001 - engine_steps; done at 0

# action -> native depth override. None means "clear the coach override" = genuine unmodified native AI.
ACTION_NAMES = ("compact", "native_default", "stretched")
ACTION_DEPTH = {0: 0.0, 1: None, 2: 1.0}
NATIVE_DEFAULT_DEPTH = 0.71          # live value of position_defense_depth_factor with no override

FEATURE_NAMES = ("ball_x", "ball_y", "own_poss", "opp_poss", "loose", "score_diff", "time_left",
                 "last_compact", "last_default", "last_stretched", "back_line_x", "team_length",
                 "own_poss_share_last", "opp_front_x")
OBS_DIM = len(FEATURE_NAMES)
N_DISCRETE = 3 * 3 * 3 * 3 * 3       # ball zone x possession x score x phase x last action


def engine(env):
    return env.unwrapped._env._env


def apply_depth(env, team, value):
    """value None/negative clears the override (native behaviour)."""
    engine(env).set_coach_defense_depth(int(team), -1.0 if value is None else float(value))


def live_depth(env, team=0):
    return engine(env).get_live_defense_depth(int(team))


def raw(obs):
    r = obs[0]
    return {"ball": np.array(r["ball"], float), "owner": int(r["ball_owned_team"]),
            "score": [int(x) for x in r["score"]], "steps_left": int(r["steps_left"]),
            "left": np.array(r["left_team"], float), "right": np.array(r["right_team"], float)}


def shape_of(s):
    xs = np.sort(s["left"][1:, 0])                      # index 0 is the goalkeeper
    return float(xs[:4].mean()), float(xs.max() - xs.min())


class GRFDepthCoachEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, decision_steps=100, shaping=0.1, gamma=0.99, scenario=SCENARIO,
                 write_video=False, logdir=None, noop_calls=False, render_video=False):
        self.decision_steps = int(decision_steps)
        self.shaping = float(shaping)
        self.gamma = float(gamma)
        other = {"action_set": "v2"}
        if render_video:
            other["render_resolution_x"] = 800
            other["render_resolution_y"] = 450
        kw = dict(env_name=scenario, representation="raw", rewards="scoring",
                  number_of_left_players_agent_controls=11, render=bool(render_video),
                  other_config_options=other)
        if write_video:
            kw.update(write_video=True, logdir=str(logdir))
            other.update(dump_full_episodes=True, display_game_stats=not render_video)
        self.env = football_env.create_environment(**kw)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.5, 1.5, (OBS_DIM,), np.float32)
        self.noop_calls = noop_calls          # True = never call the setter (pure unmodified engine)
        self.last_action = 1
        self.engine_steps = 0
        self.decisions = 0
        self._started = False

    # ----------------------------------------------------------------- helpers
    def _features(self, s, share_last):
        owner = s["owner"]
        back, length = shape_of(s)
        opp_front = float(s["right"][1:, 0].min())
        la = np.zeros(3); la[self.last_action] = 1.0
        f = np.array([s["ball"][0], s["ball"][1] / 0.42,
                      owner == 0, owner == 1, owner == -1,
                      np.clip(s["score"][0] - s["score"][1], -3, 3) / 3.0, s["steps_left"] / MATCH_STEPS,
                      la[0], la[1], la[2], back, length / 1.5, share_last, opp_front], np.float32)
        return f

    def discretize(self, f):
        """Discrete state id from a feature vector (documented thresholds, fixed a priori)."""
        zone = 0 if f[0] < -0.33 else (1 if f[0] <= 0.33 else 2)
        poss = int(np.argmax(f[2:5]))                      # 0 ours, 1 opponent, 2 loose
        sd = f[5] * 3.0
        score = 0 if sd < -0.5 else (1 if sd < 0.5 else 2)
        t = f[6]
        phase = 0 if t > 2 / 3 else (1 if t > 1 / 3 else 2)
        last = int(np.argmax(f[7:10]))
        return (((zone * 3 + poss) * 3 + score) * 3 + phase) * 3 + last

    def _apply(self, action):
        self._applied = ACTION_DEPTH[action]
        if not self.noop_calls:
            apply_depth(self.env, 1, None)                 # opponent always native
            apply_depth(self.env, 0, self._applied)

    # --------------------------------------------------------------------- api
    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(0, 2_000_000_000)
        self.seed_used = int(seed)
        cfg = self.env.unwrapped._config
        cfg["game_engine_random_seed"] = self.seed_used           # explicit, reproducible engine seed
        cfg["reverse_team_processing"] = bool(self.seed_used % 2)  # what GRF derives from the seed
        # The scenario alternates which team kicks off with episode-number parity, so pin the episode
        # number to the seed: a match is then a deterministic function of its seed (verified in tests).
        cfg._values["episode_number"] = self.seed_used - 1         # NewScenario() adds 1 on reset
        obs = self.env.reset()
        s = raw(obs)
        self.last_action = 1
        self._applied = None
        if not self.noop_calls:
            apply_depth(self.env, 0, None); apply_depth(self.env, 1, None)
        self.prev = s
        self.share_last = 0.0
        self.engine_steps = 0
        self.decisions = 0
        self.cum = {"goals_for": 0, "goals_against": 0}
        self._started = True
        self.last_features = self._features(s, 0.0)
        return self.last_features

    def phi(self, s):
        return self.shaping * float(s["ball"][0])

    def step(self, action):
        assert self._started
        action = int(action)
        assert self.action_space.contains(action)
        self.last_action = action
        self._apply(action)
        s0 = self.prev
        done = False
        poss = {0: 0, 1: 0, -1: 0}
        gains = 0
        shapes = []
        min_ball_when_opp = 1.0
        danger = 0
        prev_owner = s0["owner"]
        n = 0
        s = s0
        for _ in range(self.decision_steps):
            obs, _, done, _ = self.env.step([AI] * 11)
            n += 1
            s = raw(obs)
            poss[s["owner"]] += 1
            if s["owner"] == 0 and prev_owner == 1:
                gains += 1
            prev_owner = s["owner"]
            if s["owner"] == 1:
                min_ball_when_opp = min(min_ball_when_opp, s["ball"][0])
            shapes.append(shape_of(s))
            if s["owner"] != 0 and s["ball"][0] < -0.76 and abs(s["ball"][1]) < 0.22:
                danger += 1                      # ball in our danger zone while we do not have it
            if done:
                break
        # GRF matches last 3001 engine steps; absorb a leftover 1-step stub into this interval
        while not done and s["steps_left"] <= 1 and n < self.decision_steps + 5:
            obs, _, done, _ = self.env.step([AI] * 11)
            n += 1
            s = raw(obs)
            poss[s["owner"]] += 1
        self.engine_steps += n
        self.decisions += 1
        gf = s["score"][0] - s0["score"][0]
        ga = s["score"][1] - s0["score"][1]
        self.cum["goals_for"] += gf
        self.cum["goals_against"] += ga
        r_goal = float(gf - ga)
        # potential-based shaping on field position; terminal potential = 0 (preserves optimal policy)
        phi_next = 0.0 if done else self.phi(s)
        r_shape = self.gamma * phi_next - self.phi(s0)
        reward = r_goal + r_shape
        self.share_last = poss[0] / max(1, n)
        self.prev = s
        f = self._features(s, self.share_last)
        self.last_features = f
        info = {"engine_steps": n, "goals_for": gf, "goals_against": ga, "r_goal": r_goal, "r_shape": r_shape,
                "possession_steps": dict(poss), "possession_gains": gains,
                "live_depth": None if self.noop_calls else live_depth(self.env, 0),
                "back_line_x": float(np.mean([a for a, _ in shapes])), "team_length": float(np.mean([b for _, b in shapes])),
                "min_ball_x_opp_possession": float(min_ball_when_opp), "danger_steps": danger, "score": list(s["score"]),
                "applied_depth": self._applied, "cum": dict(self.cum), "steps_left": s["steps_left"]}
        return f, reward, bool(done), info

    # ----------------------------------------------- snapshots (tests / demos)
    def snapshot(self):
        return self.env.get_state()

    def restore(self, state, action=None):
        """set_state() does NOT restore the coach override, so it is re-applied explicitly."""
        self.env.set_state(state)
        if action is not None:
            self.last_action = action
        self._apply(self.last_action)
        self.prev = raw(self.env.unwrapped.observation())

    def close(self):
        self.env.close()
