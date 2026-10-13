"""Tabular learners: ordinary Q-learning and tabular Double Q-learning (van Hasselt, 2010). NOT Double DQN.

Both consume one episode's coach-decision transitions (s, f, a, r, s2, f2, done) in time order.
Q-learning:        Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
Double Q-learning: with probability 1/2 update Q1 else Q2, using the updated table to CHOOSE the next action and the OTHER
                   table to EVALUATE it:
                   Q1(s,a) += alpha * (r + gamma * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a))      (and symmetrically for Q2)
Terminal transitions have no bootstrap term. Behaviour policy during learning: epsilon-greedy on (Q1 + Q2) / 2.
"""
import numpy as np


def q_update(q, visits, trans, alpha, gamma):
    for (s, f, a, r, s2, f2, done) in trans:
        target = r if done else r + gamma * q[s2].max()
        q[s, a] += alpha * (target - q[s, a])
        visits[s, a] += 1


def _argmax_random_ties(v, rng):
    best = np.flatnonzero(np.isclose(v, v.max()))
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def double_q_update(q1, q2, visits, trans, alpha, gamma, rng):
    """Returns the list of tables updated (0 for Q1, 1 for Q2) so the coin flips can be audited."""
    chosen = []
    for (s, f, a, r, s2, f2, done) in trans:
        which = 0 if rng.random() < 0.5 else 1
        upd, other = (q1, q2) if which == 0 else (q2, q1)
        if done:
            target = r
        else:
            a_star = _argmax_random_ties(upd[s2], rng)          # selection with the table being updated
            target = r + gamma * other[s2, a_star]              # evaluation with the other table
        upd[s, a] += alpha * (target - upd[s, a])
        visits[s, a] += 1
        chosen.append(which)
    return chosen


def behaviour_table(algo, q=None, q1=None, q2=None):
    return q if algo == "q" else (q1 + q2) / 2.0
