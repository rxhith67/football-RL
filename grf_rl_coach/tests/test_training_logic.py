"""Learning-logic tests: TD update arithmetic, discount per coach decision, transition bookkeeping.
Run:  bash run.sh -W ignore -m unittest tests/test_training_logic.py -v"""
import unittest

import numpy as np

import rollout
from train_q import td_update


class TestTDUpdate(unittest.TestCase):
    def test_non_terminal_update_uses_one_gamma_per_decision(self):
        q = np.zeros((4, 3)); q[2] = [0.0, 1.0, 0.5]; v = np.zeros((4, 3), int)
        td_update(q, v, [(0, None, 1, 0.5, 2, None, False)], alpha=0.1, gamma=0.9)
        self.assertAlmostEqual(q[0, 1], 0.1 * (0.5 + 0.9 * 1.0 - 0.0))      # r + gamma * max Q(s')
        self.assertEqual(v[0, 1], 1)

    def test_terminal_update_has_no_bootstrap(self):
        q = np.zeros((4, 3)); q[2] = [5.0, 5.0, 5.0]; v = np.zeros((4, 3), int)
        td_update(q, v, [(0, None, 0, 1.0, 2, None, True)], alpha=0.5, gamma=0.99)
        self.assertAlmostEqual(q[0, 0], 0.5 * 1.0)

    def test_updates_are_applied_in_order(self):
        q = np.zeros((3, 3)); v = np.zeros((3, 3), int)
        trans = [(0, None, 0, 1.0, 1, None, False), (1, None, 0, 2.0, 2, None, True)]
        td_update(q, v, trans, alpha=1.0, gamma=1.0)
        self.assertAlmostEqual(q[1, 0], 2.0)
        self.assertAlmostEqual(q[0, 0], 1.0)                                  # forward order: bootstraps from Q(s1) BEFORE s1 is updated


class TestTransitionBookkeeping(unittest.TestCase):
    def test_episode_transitions_match_environment(self):
        rollout._init(500, 0.1, 0.99)                                         # 500-step decisions
        ep, trans = rollout.run_episode(dict(policy="fixed", action=1, seed=424242, collect=True))
        self.assertEqual(ep["decisions"], len(trans))
        self.assertEqual(ep["decisions"], 6)                                  # 3001 steps: 6 x 500, the 1-step stub joins the last interval
        self.assertEqual(ep["engine_steps"], 3001)
        self.assertEqual([t[6] for t in trans], [False] * 5 + [True])         # only the last transition is terminal
        for a, b in zip(trans, trans[1:]):
            self.assertTrue(np.allclose(a[5], b[1]))                          # next-state of i == state of i+1
            self.assertEqual(a[4], b[0])
        self.assertAlmostEqual(sum(t[3] for t in trans), ep["return"], places=9)
        self.assertAlmostEqual(ep["return_goal"], ep["goal_diff"], places=9)  # goal reward sums to final score diff
        self.assertEqual(sum(g for _, g, _ in ep["goal_events"]) - sum(a for _, _, a in ep["goal_events"]), ep["goal_diff"])


if __name__ == "__main__":
    unittest.main()
