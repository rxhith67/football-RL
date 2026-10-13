"""Environment validation (Stage 2). Run:  bash run.sh -m unittest discover -s tests -v
Each test uses one shared environment object to keep engine start-up costs down."""
import json
import unittest

import numpy as np

import coach_env as ce
from coach_env import GRFDepthCoachEnv, AI, apply_depth, live_depth

RESULTS = {}


def trajectory(env, seed, actions, decision_steps=100):
    env.reset(seed=seed)
    out = []
    for a in actions:
        _, r, d, info = env.step(a)
        out.append((np.array(env.prev["left"]).copy(), np.array(env.prev["ball"]).copy(), r))
        if d:
            break
    return out


class TestCoachEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = GRFDepthCoachEnv(decision_steps=100)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        json.dump(RESULTS, open("results/env_tests.json", "w"), indent=1, default=str)

    def test_01_multiple_decisions_full_match(self):
        e = self.env
        f = e.reset(seed=11)
        n, done, total = 0, False, 0.0
        while not done:
            f, r, done, info = e.step(1)
            n += 1
            total += info["r_goal"]
        self.assertEqual(n, 30)                       # 3001-step match, 100 per decision, 1-step stub absorbed
        self.assertEqual(e.engine_steps, 3001)
        self.assertEqual(e.decisions, 30)
        self.assertEqual(info["steps_left"], 0)
        RESULTS["full_match"] = {"decisions": n, "engine_steps": e.engine_steps, "score": info["score"]}

    def test_02_partial_final_interval(self):
        e = GRFDepthCoachEnv(decision_steps=70)
        try:
            e.reset(seed=12)
            done, n, last = False, 0, None
            while not done:
                _, _, done, info = e.step(1)
                n += 1
                last = info["engine_steps"]
            self.assertEqual(n, 43)                  # ceil(3000 / 70)
            self.assertEqual(last, 3001 - 42 * 70)   # final interval is cut short (61 steps), not padded
            self.assertEqual(e.engine_steps, 3001)
            RESULTS["partial_interval"] = {"decisions": n, "last_interval_engine_steps": last}
        finally:
            e.close()

    def test_03_actions_reach_native_controller(self):
        e = self.env
        e.reset(seed=13)
        self.assertAlmostEqual(live_depth(e.env, 0), 0.71, places=5)
        apply_depth(e.env, 0, 0.0); self.assertEqual(live_depth(e.env, 0), 0.0)   # immediate
        e.step(0);  a0 = live_depth(e.env, 0)
        e.step(2);  a2 = live_depth(e.env, 0)
        e.step(1);  a1 = live_depth(e.env, 0)             # override cleared -> native value after <= 10 engine steps
        self.assertEqual((a0, a2), (0.0, 1.0))
        self.assertAlmostEqual(a1, ce.NATIVE_DEFAULT_DEPTH, places=5)
        self.assertAlmostEqual(live_depth(e.env, 1), 0.71, places=5)          # opponent never touched
        # latency of returning to native after clearing (engine refreshes tactics every 10 engine steps)
        e.step(2)
        apply_depth(e.env, 0, None)
        lat = None
        for k in range(1, 31):
            e.env.step([AI] * 11)
            if abs(live_depth(e.env, 0) - 0.71) < 1e-4:
                lat = k; break
        self.assertIsNotNone(lat); self.assertLessEqual(lat, 11)
        RESULTS["actions"] = {"live_after_compact": a0, "live_after_stretched": a2, "live_after_default": a1,
                              "engine_steps_to_return_to_native_after_clear": lat}

    def test_04_builtin_ai_stays_active(self):
        e = self.env
        sent = []
        real_step = e.env.step
        def spy(actions):
            sent.append(list(actions)); return real_step(actions)
        e.env.step = spy
        try:
            e.reset(seed=14)
            owners, passes, last_owner = set(), 0, None
            for a in (0, 2, 1, 0, 2, 1):
                e.step(a)
        finally:
            e.env.step = real_step
        self.assertEqual(len(sent), 600)
        self.assertTrue(all(len(x) == 11 and all(v == AI for v in x) for x in sent))   # never anything but builtin_ai
        RESULTS["builtin_ai"] = {"engine_steps_checked": len(sent), "non_ai_actions_sent": 0}

    def test_05_actions_change_team_shape(self):
        res = {}
        for a in (0, 1, 2):
            e = self.env; e.reset(seed=15)
            L = []
            for _ in range(6):
                _, _, _, info = e.step(a); L.append(info["team_length"])
            res[a] = float(np.mean(L[1:]))
        RESULTS["shape_by_action_team_length"] = res
        self.assertLess(res[0], res[1]); self.assertLess(res[1], res[2])

    def test_06_finite_and_valid(self):
        e = self.env; rng = np.random.default_rng(0)
        f = e.reset(seed=16); done = False; k = 0
        while not done and k < 15:
            self.assertTrue(np.all(np.isfinite(f))); self.assertTrue(e.observation_space.contains(f))
            self.assertTrue(0 <= e.discretize(f) < ce.N_DISCRETE)
            f, r, done, info = e.step(int(rng.integers(3))); k += 1
            self.assertTrue(np.isfinite(r)); self.assertAlmostEqual(r, info["r_goal"] + info["r_shape"])
        RESULTS["finite"] = {"decisions_checked": k}

    def test_07_goal_and_terminal_accounting(self):
        e = self.env; e.reset(seed=1)       # seed 1 scored early in the smoke test
        done, total_goal, total_shape = False, 0.0, 0.0
        while not done:
            _, r, done, info = e.step(1); total_goal += info["r_goal"]; total_shape += info["r_shape"]
        gf, ga = info["score"]
        self.assertEqual(total_goal, gf - ga)
        self.assertEqual(info["cum"], {"goals_for": gf, "goals_against": ga})
        # potential-based shaping telescopes: sum = gamma-discounted potential difference, terminal potential 0
        RESULTS["goal_accounting"] = {"final_score": info["score"], "sum_r_goal": total_goal, "sum_r_shape": total_shape}

    def test_08_set_state_requires_explicit_override(self):
        e = self.env; e.reset(seed=17)
        e.step(2); e.step(2)
        snap = e.snapshot(); self.assertEqual(live_depth(e.env, 0), 1.0)
        e.step(1); e.step(1)                                          # clear -> back to native
        self.assertAlmostEqual(live_depth(e.env, 0), 0.71, places=5)
        e.restore(snap, action=2)                                     # restore + explicit re-apply
        self.assertEqual(live_depth(e.env, 0), 1.0)
        for _ in range(30): e.env.step([AI] * 11)
        self.assertEqual(live_depth(e.env, 0), 1.0)                   # persists over 3 native update cycles
        # control: raw set_state without re-applying -> override member was cleared, so live reverts
        apply_depth(e.env, 0, None)                                   # override cleared, as after any other decision
        e.env.set_state(snap)
        raw_live = live_depth(e.env, 0)
        for _ in range(20): e.env.step([AI] * 11)
        RESULTS["set_state"] = {"live_after_restore_with_reapply_30_steps": 1.0,
                                "live_after_raw_set_state_immediately": raw_live,
                                "live_after_raw_set_state_20_steps": live_depth(e.env, 0)}

    def test_09_default_action_equals_unmodified_engine(self):
        seed, acts = 21, [1] * 5
        a = trajectory(self.env, seed, acts)                        # action 1 -> setter(-1) every decision
        b_env = GRFDepthCoachEnv(noop_calls=True)                    # never calls the setter at all
        try:
            b = trajectory(b_env, seed, acts)
        finally:
            b_env.close()
        diff_clear = max(float(np.abs(x[0] - y[0]).max()) for x, y in zip(a, b))
        # numeric neutral override (0.71) is NOT assumed equivalent: measure it
        c = []
        self.env.reset(seed=seed)
        for _ in acts:
            apply_depth(self.env.env, 0, 0.71)
            for _ in range(100): self.env.env.step([AI] * 11)
            c.append(np.array(raw_left(self.env.env)))
        diff_neutral = max(float(np.abs(x[0] - y).max()) for x, y in zip(a, c))
        RESULTS["default_equivalence"] = {"max_pos_diff_clear_vs_unmodified": diff_clear,
                                          "max_pos_diff_numeric_0.71_vs_clear": diff_neutral}
        self.assertEqual(diff_clear, 0.0)

    def test_10_same_seed_same_process_repeatable(self):
        a = trajectory(self.env, 31, [0, 2, 1])
        b = trajectory(self.env, 31, [0, 2, 1])
        d = max(float(np.abs(x[0] - y[0]).max()) for x, y in zip(a, b))
        RESULTS["same_seed_repeat"] = {"max_diff_in_process": d}
        self.assertEqual(d, 0.0)


def raw_left(core_env):
    return ce.raw(core_env.unwrapped.observation())["left"]


if __name__ == "__main__":
    unittest.main()
