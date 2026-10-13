"""Coach policies (action selection from features / discrete state). Pure numpy: usable inside workers."""
import numpy as np

DEFAULT_ACTION = 1


def hand_coach(f):
    """Simple situation-based coach, declared BEFORE any results were seen.
    compact when the opponent has the ball in our third, or when protecting a lead late;
    stretched when trailing late; otherwise leave the native default."""
    ball_x, opp_poss = f[0], f[3] > 0.5
    score_diff, time_left = f[5] * 3.0, f[6]
    if opp_poss and ball_x < -0.33:
        return 0
    if score_diff <= -0.5 and time_left < 1 / 3:
        return 2
    if score_diff >= 0.5 and time_left < 1 / 2:
        return 0
    return 1


def q_action(q, s, rng=None, epsilon=0.0, tie_default=True):
    if rng is not None and rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    v = q[s]
    best = np.flatnonzero(np.isclose(v, v.max()))
    if len(best) == 1:
        return int(best[0])
    if tie_default and DEFAULT_ACTION in best:          # evaluation: ties resolve to the native default
        return DEFAULT_ACTION
    return int(rng.choice(best)) if rng is not None else int(best[0])


def mlp_forward(weights, x):
    """weights = [W1,b1,W2,b2,W3,b3] numpy arrays; ReLU MLP."""
    h = np.maximum(0, x @ weights[0] + weights[1])
    h = np.maximum(0, h @ weights[2] + weights[3])
    return h @ weights[4] + weights[5]


def dqn_action(weights, f, rng=None, epsilon=0.0):
    if rng is not None and rng.random() < epsilon:
        return int(rng.integers(3))
    q = mlp_forward(weights, np.asarray(f, np.float32))
    best = np.flatnonzero(np.isclose(q, q.max(), atol=1e-6))
    return DEFAULT_ACTION if DEFAULT_ACTION in best else int(np.argmax(q))
