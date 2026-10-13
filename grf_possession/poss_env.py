"""Possession-aware coaching environment. Subclasses the existing GRFDepthCoachEnv WITHOUT modifying it.

Differences from the original environment
  * the coached team can be the left (side 0) or the right (side 1) team; for side 1 all features use GRF's official
    180-degree observation rotation, so the existing 243-state discretisation and 14 features keep their meaning;
  * every engine step is classified (possession.py) and an interval's controlled-possession share is computed;
  * two reward definitions are computed for every interval and the selected one is returned:
        reward_mode="original"   : goal difference + 0.1 * gamma-discounted potential of ball position   (existing reward)
        reward_mode="possession" : poss_weight * (our share - 0.5) + goal_weight * (goals for - goals against)
  * a per-step stream (coached frame) is recorded for offline statistics.
Native player intelligence is unchanged: every left-team slot receives builtin_ai on every engine step; the right team has
no external slots. The coach only calls set_coach_defense_depth on its own team; the opponent's override is cleared.
"""
import numpy as np
from gfootball.env import observation_rotation

from coach_env import GRFDepthCoachEnv, AI, ACTION_DEPTH, MATCH_STEPS, apply_depth, live_depth, raw
import possession as P

NATIVE_DEPTH = 0.71


class PossessionCoachEnv(GRFDepthCoachEnv):
    def __init__(self, side=0, reward_mode="possession", poss_weight=1.0, goal_weight=2.0, baseline=0.5,
                 min_controlled=10, decision_steps=100, shaping=0.1, gamma=0.99, write_video=False, logdir=None):
        super().__init__(decision_steps=decision_steps, shaping=shaping, gamma=gamma, write_video=write_video, logdir=logdir)
        assert side in (0, 1) and reward_mode in ("original", "possession")
        self.side, self.reward_mode = side, reward_mode
        self.poss_weight, self.goal_weight, self.baseline, self.min_controlled = poss_weight, goal_weight, baseline, min_controlled
        self.stream = []

    # ------------------------------------------------------------------ views
    def _core_view(self):
        core = self.env.unwrapped._env.observation()
        if self.side == 1:
            core = observation_rotation.flip_observation(core, self.env.unwrapped._config)
        return raw([core])

    def _apply_side(self, action):
        self._applied = ACTION_DEPTH[action]
        if not self.noop_calls:
            apply_depth(self.env, 1 - self.side, None)            # opponent: always the native engine behaviour
            apply_depth(self.env, self.side, self._applied)

    def _own_phi(self, view):
        return self.shaping * float(view["ball"][0])

    # ------------------------------------------------------------------- api
    def reset(self, seed=None):
        super().reset(seed)                                       # pins seed/episode number, clears both overrides
        view = self._core_view()
        self.prev = view
        self.last_features = self._features(view, 0.0)
        self.stream = []
        self.interval_log = []
        return self.last_features

    def step(self, action):
        assert self._started
        action = int(action)
        assert self.action_space.contains(action)
        self.last_action = action
        self._apply_side(action)
        s0 = self.prev                                            # coached-frame view at the interval start
        side = self.side
        owners, modes = [], []
        n, done = 0, False
        own_x = []
        steps_left = None
        score_end = (0, 0)
        while True:
            obs, _, done, _ = self.env.step([AI] * 11)
            n += 1
            r = obs[0]
            go = int(r["ball_owned_team"])
            owner = go if side == 0 else (1 - go if go > -1 else -1)          # coached-frame label
            sgn = 1.0 if side == 0 else -1.0
            sl, sr = int(r["score"][0]), int(r["score"][1])
            so, sp = (sl, sr) if side == 0 else (sr, sl)
            mode = int(r["game_mode"])
            owners.append(owner); modes.append(mode)
            self.stream.append((owner, sgn * float(r["ball"][0]), sgn * float(r["ball"][1]), sgn * float(r["ball_direction"][0]),
                                sgn * float(r["ball_direction"][1]), mode, so, sp, int(r["ball_owned_player"])))
            team = np.asarray(r["left_team" if side == 0 else "right_team"], float)
            own_x.append(np.sort(sgn * team[1:, 0]))
            steps_left = int(r["steps_left"]); score_end = (so, sp)
            if done: break
            if n >= self.decision_steps and not (steps_left <= 1 and n < self.decision_steps + 5):
                break                                                           # same stub-absorption rule as the original env
        self.engine_steps += n
        self.decisions += 1
        gf, ga = score_end[0] - s0["score"][0], score_end[1] - s0["score"][1]
        self.cum["goals_for"] += gf; self.cum["goals_against"] += ga
        stats = P.interval_stats(owners, modes, self.min_controlled)
        r_poss = P.possession_reward(stats["share"], self.poss_weight, self.baseline)
        r_goalp = P.goal_reward(gf, ga, self.goal_weight)
        # original reward, identical definition to GRFDepthCoachEnv.step
        view = self._core_view()
        phi_next = 0.0 if done else self._own_phi(view)
        r_goal_o = float(gf - ga)
        r_shape_o = self.gamma * phi_next - self._own_phi(s0)
        r_original = r_goal_o + r_shape_o
        r_possession = r_poss + r_goalp
        reward = r_original if self.reward_mode == "original" else r_possession
        own_share_all = float(np.mean(np.asarray(owners) == 0))
        self.share_last = own_share_all
        self.prev = view
        f = self._features(view, own_share_all)
        self.last_features = f
        xs = np.stack(own_x)
        info = {"engine_steps": n, "goals_for": int(gf), "goals_against": int(ga), "poss": stats,
                "r_poss": r_poss, "r_goal_poss": r_goalp, "reward_possession": r_possession,
                "r_goal_original": r_goal_o, "r_shape_original": r_shape_o, "reward_original": r_original,
                "score": [int(score_end[0]), int(score_end[1])], "steps_left": steps_left, "applied_depth": self._applied,
                "live_depth": None if self.noop_calls else live_depth(self.env, side),
                "opp_live_depth": None if self.noop_calls else live_depth(self.env, 1 - side),
                "team_length": float((xs.max(1) - xs.min(1)).mean()), "back_line_x": float(xs[:, :4].mean()),
                "cum": dict(self.cum)}
        self.interval_log.append(info)
        return f, float(reward), bool(done), info
