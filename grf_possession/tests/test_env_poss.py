"""Engine-level tests for the possession environment. Run: bash run.sh -W ignore -m unittest tests.test_env_poss -v  (about 3 min)"""
import json
import unittest

import numpy as np

import possession as P
from coach_env import GRFDepthCoachEnv, AI, apply_depth, live_depth
from poss_env import PossessionCoachEnv

RESULTS = {}


def play(env, seed, actions, side=0):
    env.side = side
    env.reset(seed=seed)
    rows = []
    for a in actions:
        f, r, d, info = env.step(a)
        rows.append((f.copy(), r, d, info))
        if d: break
    return rows


class TestPossessionEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PossessionCoachEnv(side=0, reward_mode="possession")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        json.dump(RESULTS, open("results/env_poss_tests.json", "w"), indent=1, default=str)

    def test_01_original_reward_mode_matches_existing_environment(self):
        acts = [0, 1, 2, 2, 1, 0]
        old = GRFDepthCoachEnv()
        try:
            old.reset(seed=777001); orows = []
            for a in acts:
                f, r, d, info = old.step(a); orows.append((f.copy(), r, d))
        finally:
            old.close()
        self.env.reward_mode = "original"
        new = play(self.env, 777001, acts)
        self.env.reward_mode = "possession"
        for (fo, ro, do), (fn, rn, dn, info) in zip(orows, new):
            self.assertTrue(np.allclose(fo, fn, atol=1e-6)); self.assertAlmostEqual(ro, rn, places=9); self.assertEqual(do, dn)
        RESULTS["original_reward_parity"] = {"decisions": len(acts), "max_reward_diff": max(abs(o[1] - n[1]) for o, n in zip(orows, new))}

    def test_02_full_match_possession_and_goal_accounting(self):
        rows = play(self.env, 777002, [1] * 30)
        self.assertEqual(len(rows), 30); self.assertTrue(rows[-1][2]); self.assertFalse(any(r[2] for r in rows[:-1]))
        st = np.array(self.env.stream)
        self.assertEqual(len(st), 3001); self.assertEqual(sum(r[3]["engine_steps"] for r in rows), 3001)
        # independent recomputation of every interval's statistics from the raw stream
        pos = 0
        for f, r, d, info in rows:
            n = info["engine_steps"]; seg = st[pos:pos + n]; pos += n
            ours = int(((seg[:, 0] == 0) & (seg[:, 5] == 0)).sum()); opp = int(((seg[:, 0] == 1) & (seg[:, 5] == 0)).sum())
            self.assertEqual((info["poss"]["ours"], info["poss"]["opp"]), (ours, opp))
            self.assertEqual(info["poss"]["contested"], int(((seg[:, 0] == -1) & (seg[:, 5] == 0)).sum()))
            self.assertEqual(info["poss"]["dead"], int((seg[:, 5] != 0).sum()))
            share = ours / (ours + opp) if ours + opp >= 10 else None
            exp = (0.0 if share is None else share - 0.5) + 2.0 * (info["goals_for"] - info["goals_against"])
            self.assertAlmostEqual(r, exp, places=9)                    # reward = (share - 0.5) + 2 * goal difference
            self.assertAlmostEqual(info["reward_possession"], exp, places=9)
        # every goal counted exactly once
        final_gd = int(st[-1, 6] - st[-1, 7]); n_events = int(np.sum(np.diff(st[:, 6]) > 0) + np.sum(np.diff(st[:, 7]) > 0))
        self.assertEqual(sum(r[3]["goals_for"] - r[3]["goals_against"] for r in rows), final_gd)
        self.assertEqual(sum(r[3]["goals_for"] + r[3]["goals_against"] for r in rows), n_events)
        m = P.match_possession([r[3]["poss"] for r in rows])
        self.assertEqual(m["steps"], 3001)
        RESULTS["full_match_accounting"] = {"steps": 3001, "decisions": 30, "possession_share": m["share"], "contested_frac": m["contested_frac"],
                                            "dead_frac": m["dead_frac"], "goal_events": n_events, "final_goal_diff": final_gd}

    def test_03_dead_ball_and_loose_steps_never_counted_as_possession(self):
        play(self.env, 777003, [1] * 8)
        st = np.array(self.env.stream)
        self.assertGreater((st[:, 5] != 0).sum(), 0)                    # the kick-off alone gives dead-ball steps
        info = self.env.interval_log
        controlled = sum(i["poss"]["ours"] + i["poss"]["opp"] for i in info)
        self.assertEqual(controlled, int(((st[:, 0] >= 0) & (st[:, 5] == 0)).sum()))
        self.assertGreater(sum(i["poss"]["contested"] for i in info), 0)  # loose-ball steps exist and are reported separately

    def test_04_actions_reach_native_controller_on_both_sides(self):
        res = {}
        for side in (0, 1):
            rows = play(self.env, 777004, [0, 2, 1], side=side)
            lives = [(r[3]["live_depth"], r[3]["opp_live_depth"]) for r in rows]
            self.assertAlmostEqual(lives[0][0], 0.0, places=4); self.assertAlmostEqual(lives[1][0], 1.0, places=4)
            self.assertAlmostEqual(lives[2][0], 0.71, places=4)
            self.assertTrue(all(abs(o - 0.71) < 1e-4 for _, o in lives))     # opponent never touched
            res[f"side{side}"] = lives
        RESULTS["live_depth_by_side"] = res

    def test_05_actions_change_team_shape_on_both_sides(self):
        out = {}
        for side in (0, 1):
            ml = {}
            for a in (0, 2):
                rows = play(self.env, 777005, [a] * 5, side=side)
                ml[a] = float(np.mean([r[3]["team_length"] for r in rows[1:]]))
            self.assertLess(ml[0], ml[2]); out[f"side{side}"] = ml
        RESULTS["shape_by_side"] = out

    def test_06_right_side_view_is_the_mirror_of_the_left_side_view(self):
        play(self.env, 777006, [1, 1], side=1)
        core = self.env.env.unwrapped._env.observation(); view = self.env._core_view()
        self.assertTrue(np.allclose(view["ball"][:2], -np.asarray(core["ball"][:2])))
        go = int(core["ball_owned_team"]); self.assertEqual(view["owner"], (1 - go) if go > -1 else -1)
        self.assertEqual(list(view["score"]), [core["score"][1], core["score"][0]])
        self.assertTrue(np.allclose(view["left"], -np.asarray(core["right_team"]))); self.assertTrue(np.allclose(view["right"], -np.asarray(core["left_team"])))

    def test_07_native_vs_native_is_side_symmetric_and_stream_mirrors(self):
        self.env.noop_calls = True
        try:
            play(self.env, 777007, [1] * 4, side=0); s0 = np.array(self.env.stream)
            play(self.env, 777007, [1] * 4, side=1); s1 = np.array(self.env.stream)
        finally:
            self.env.noop_calls = False
        self.assertEqual(s0.shape, s1.shape)
        swap = np.where(s0[:, 0] >= 0, 1 - s0[:, 0], -1)
        self.assertTrue(np.array_equal(s1[:, 0], swap))                                    # ownership labels swapped
        self.assertTrue(np.allclose(s1[:, 1:5], -s0[:, 1:5], atol=1e-6))                  # ball position and velocity mirrored
        self.assertTrue(np.array_equal(s1[:, 6], s0[:, 7]) and np.array_equal(s1[:, 7], s0[:, 6]))   # scores swapped
        RESULTS["native_vs_native_mirror"] = {"steps": int(len(s0)), "max_abs_diff_ball": float(np.abs(s1[:, 1:5] + s0[:, 1:5]).max())}

    def test_08_native_default_action_equals_unmodified_engine_on_both_sides(self):
        out = {}
        for side in (0, 1):
            play(self.env, 777008, [1] * 4, side=side); a = np.array(self.env.stream)
            self.env.noop_calls = True
            try:
                play(self.env, 777008, [1] * 4, side=side); b = np.array(self.env.stream)
            finally:
                self.env.noop_calls = False
            self.assertTrue(np.array_equal(a, b)); out[f"side{side}"] = float(np.abs(a - b).max())
        RESULTS["native_default_vs_unmodified_max_diff"] = out

    def test_09_only_builtin_ai_actions_are_ever_sent(self):
        sent = []
        real = self.env.env.step
        def spy(actions):
            sent.append(list(actions)); return real(actions)
        self.env.env.step = spy
        try:
            play(self.env, 777009, [0, 2, 1], side=1)
        finally:
            self.env.env.step = real
        self.assertEqual(len(sent), 300); self.assertTrue(all(len(x) == 11 and all(v == AI for v in x) for x in sent))

    def test_10_deterministic_per_seed_and_side(self):
        for side in (0, 1):
            play(self.env, 777010, [0, 2, 1], side=side); a = np.array(self.env.stream)
            play(self.env, 777010, [0, 2, 1], side=side); b = np.array(self.env.stream)
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
