"""Simulator-free tests: possession detector, reward formula, Q-learning and Double Q-learning updates.
Run:  bash run.sh -W ignore -m unittest tests.test_logic -v"""
import unittest

import numpy as np

import possession as P
from learners import q_update, double_q_update


class FixedRng:
    """Stub RNG: random() returns scripted values; choice picks the first candidate."""
    def __init__(self, coins): self.coins = list(coins)
    def random(self): return self.coins.pop(0)
    def choice(self, x): return x[0]


class TestPossession(unittest.TestCase):
    def test_share_counts_controlled_steps_only(self):
        owner = [0] * 65 + [1] * 35
        s = P.interval_stats(owner, [0] * 100)
        self.assertAlmostEqual(s["share"], 0.65); self.assertEqual((s["ours"], s["opp"], s["contested"], s["dead"]), (65, 35, 0, 0))

    def test_contested_balls_are_not_possession(self):
        owner = [0] * 65 + [1] * 35 + [-1] * 100
        s = P.interval_stats(owner, [0] * 200)
        self.assertAlmostEqual(s["share"], 0.65)                      # loose steps do not dilute or inflate the share
        self.assertEqual(s["contested"], 100); self.assertEqual(s["controlled"], 100)

    def test_dead_ball_steps_are_excluded_even_if_an_owner_is_reported(self):
        owner = [0] * 50 + [0] * 30 + [1] * 20
        mode = [0] * 50 + [1] * 30 + [0] * 20                       # 30 steps of kick-off/throw-in with us "owning" the ball
        s = P.interval_stats(owner, mode)
        self.assertEqual((s["ours"], s["opp"], s["dead"]), (50, 20, 30))
        self.assertAlmostEqual(s["share"], 50 / 70)

    def test_no_clear_possession_gives_none_and_zero_reward(self):
        s = P.interval_stats([-1] * 95 + [0] * 5, [0] * 100, min_controlled=10)
        self.assertIsNone(s["share"]); self.assertEqual(P.possession_reward(s["share"]), 0.0)

    def test_every_step_classified_exactly_once(self):
        rng = np.random.default_rng(0); owner = rng.integers(-1, 2, 500); mode = rng.integers(0, 3, 500)
        s = P.interval_stats(owner, mode)
        self.assertEqual(s["ours"] + s["opp"] + s["contested"] + s["dead"], 500)

    def test_match_totals_are_time_weighted(self):
        a = P.interval_stats([0] * 90 + [1] * 10, [0] * 100); b = P.interval_stats([0] * 10 + [1] * 10, [0] * 20)
        m = P.match_possession([a, b])
        self.assertAlmostEqual(m["share"], 100 / 120)                 # not the mean of 0.9 and 0.5


class TestReward(unittest.TestCase):
    def rows(self):
        return [(0.65, 0, 0, 0.15, 0.0, 0.15), (0.35, 0, 0, -0.15, 0.0, -0.15), (0.65, 1, 0, 0.15, 2.0, 2.15), (0.65, 0, 1, 0.15, -2.0, -1.85)]

    def test_specification_table(self):
        for share, gf, ga, p, g, tot in self.rows():
            stats = {"share": share}
            t, pp, gg = P.total_reward(stats, gf, ga)
            self.assertAlmostEqual(pp, p); self.assertAlmostEqual(gg, g); self.assertAlmostEqual(t, tot)

    def test_weights_are_configurable(self):
        t, p, g = P.total_reward({"share": 0.8}, 1, 0, poss_weight=2.0, goal_weight=5.0, baseline=0.4)
        self.assertAlmostEqual(p, 2.0 * 0.4); self.assertAlmostEqual(g, 5.0); self.assertAlmostEqual(t, 5.8)

    def test_goal_without_clear_possession_still_rewarded(self):
        t, p, g = P.total_reward({"share": None}, 1, 0)
        self.assertEqual((p, g, t), (0.0, 2.0, 2.0))


class TestQLearning(unittest.TestCase):
    def test_update_matches_formula(self):
        q = np.zeros((4, 3)); q[2] = [0.0, 1.0, 0.5]; v = np.zeros((4, 3), int)
        q_update(q, v, [(0, None, 1, 0.5, 2, None, False)], alpha=0.1, gamma=0.9)
        self.assertAlmostEqual(q[0, 1], 0.1 * (0.5 + 0.9 * 1.0)); self.assertEqual(v[0, 1], 1)

    def test_terminal_has_no_bootstrap(self):
        q = np.zeros((4, 3)); q[2] = 9.0; v = np.zeros((4, 3), int)
        q_update(q, v, [(0, None, 0, 1.0, 2, None, True)], alpha=0.5, gamma=0.99)
        self.assertAlmostEqual(q[0, 0], 0.5)

    def test_same_as_existing_project_implementation(self):
        import sys; sys.argv = ["x"]
        from train_q import td_update                                  # grf_rl_coach, imported read-only
        rng = np.random.default_rng(3); trans = [(int(rng.integers(5)), None, int(rng.integers(3)), float(rng.normal()), int(rng.integers(5)), None, bool(i % 7 == 6)) for i in range(60)]
        a = np.zeros((5, 3)); b = np.zeros((5, 3)); va = np.zeros((5, 3), int); vb = np.zeros((5, 3), int)
        td_update(a, va, trans, 0.1, 0.99); q_update(b, vb, trans, 0.1, 0.99)
        self.assertTrue(np.array_equal(a, b))


class TestDoubleQ(unittest.TestCase):
    def tables(self):
        q1 = np.zeros((4, 3)); q2 = np.zeros((4, 3)); q1[2] = [0.0, 2.0, 1.0]; q2[2] = [5.0, 3.0, 9.0]
        return q1, q2, np.zeros((4, 3), int)

    def test_update_q1_selects_with_q1_and_evaluates_with_q2(self):
        q1, q2, v = self.tables()
        chosen = double_q_update(q1, q2, v, [(0, None, 1, 1.0, 2, None, False)], alpha=0.5, gamma=0.9, rng=FixedRng([0.1]))
        self.assertEqual(chosen, [0])
        a_star = 1                                                     # argmax of q1[2]
        self.assertAlmostEqual(q1[0, 1], 0.5 * (1.0 + 0.9 * q2[2, a_star]))   # q2[2,1] = 3, NOT q1's max (2) and NOT q2's max (9)
        self.assertTrue(np.all(q2[0] == 0))                            # the other table is untouched

    def test_update_q2_selects_with_q2_and_evaluates_with_q1(self):
        q1, q2, v = self.tables()
        chosen = double_q_update(q1, q2, v, [(0, None, 0, 1.0, 2, None, False)], alpha=0.5, gamma=0.9, rng=FixedRng([0.9]))
        self.assertEqual(chosen, [1])
        a_star = 2                                                     # argmax of q2[2]
        self.assertAlmostEqual(q2[0, 0], 0.5 * (1.0 + 0.9 * q1[2, a_star]))   # q1[2,2] = 1
        self.assertTrue(np.all(q1[0] == 0))

    def test_terminal_transition(self):
        q1, q2, v = self.tables()
        double_q_update(q1, q2, v, [(0, None, 2, 1.5, 2, None, True)], alpha=0.4, gamma=0.99, rng=FixedRng([0.2]))
        self.assertAlmostEqual(q1[0, 2], 0.4 * 1.5)

    def test_tables_chosen_with_probability_half_and_both_learn(self):
        q1 = np.zeros((3, 3)); q2 = np.zeros((3, 3)); v = np.zeros((3, 3), int)
        trans = [(0, None, 0, 1.0, 1, None, False)] * 10000
        chosen = double_q_update(q1, q2, v, trans, 0.01, 0.9, np.random.default_rng(0))
        self.assertAlmostEqual(np.mean(chosen), 0.5, delta=0.02)
        self.assertGreater(q1[0, 0], 0); self.assertGreater(q2[0, 0], 0)

    def test_reproducible_with_same_seed(self):
        trans = [(i % 3, None, i % 3, float(i % 5) - 2, (i + 1) % 3, None, i % 11 == 10) for i in range(300)]
        out = []
        for _ in range(2):
            q1 = np.zeros((3, 3)); q2 = np.zeros((3, 3)); v = np.zeros((3, 3), int)
            double_q_update(q1, q2, v, trans, 0.1, 0.9, np.random.default_rng(42)); out.append((q1.copy(), q2.copy()))
        self.assertTrue(np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1]))

    def test_greedy_policy_uses_the_mean_of_both_tables(self):
        from learners import behaviour_table
        q1 = np.zeros((1, 3)); q2 = np.zeros((1, 3)); q1[0] = [1.0, 0.0, 0.0]; q2[0] = [0.0, 3.0, 0.0]
        tb = behaviour_table("double_q", q1=q1, q2=q2)
        self.assertTrue(np.allclose(tb[0], [0.5, 1.5, 0.0])); self.assertEqual(int(np.argmax(tb[0])), 1)   # not Q1's choice (0), not Q1+Q2 ties
        self.assertTrue(np.array_equal(behaviour_table("q", q=q1), q1))                                    # plain Q-learning uses its own table

    def test_reduces_maximisation_bias_relative_to_q_learning(self):
        # state 0 with 3 actions, zero-mean noisy reward, self-transition: true value of every action is 0
        rng_env = np.random.default_rng(7)
        def run(double):
            q = np.zeros((1, 3)); q1 = np.zeros((1, 3)); q2 = np.zeros((1, 3)); v = np.zeros((1, 3), int); lr = np.random.default_rng(1)
            for _ in range(4000):
                a = int(rng_env.integers(3)); t = [(0, None, a, float(rng_env.normal()), 0, None, False)]
                if double: double_q_update(q1, q2, v, t, 0.05, 0.9, lr)
                else: q_update(q, v, t, 0.05, 0.9)
            return (q1 + q2).mean() / 2 if double else q.mean()
        self.assertLess(run(True), run(False))                           # classical result: Q-learning over-estimates


if __name__ == "__main__":
    unittest.main()
