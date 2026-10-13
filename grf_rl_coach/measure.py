"""Evaluation-only measurement layer. It records the raw per-engine-step ball/possession stream of a match WITHOUT
changing the coach, the policy, the rewards or the engine: a read-only spy on the observation returned by env.step.
Used by measured_eval.py. Metric definitions live in metrics.py and are computed offline from these arrays.

Per-step columns: owner(-1 loose / 0 ours / 1 opponent), ball x, ball y, ball vx, ball vy, game_mode, our score, opp score,
owner player index."""
import numpy as np

import rollout

COLS = ("owner", "bx", "by", "bvx", "bvy", "game_mode", "score_us", "score_opp", "owner_player")


def run_measured(spec):
    e = rollout._ENV
    real_step = e.env.step
    buf = []

    def spy(actions):
        out = real_step(actions)
        r = out[0][0]
        buf.append((int(r["ball_owned_team"]), float(r["ball"][0]), float(r["ball"][1]), float(r["ball_direction"][0]),
                    float(r["ball_direction"][1]), int(r["game_mode"]), int(r["score"][0]), int(r["score"][1]),
                    int(r["ball_owned_player"])))
        return out

    e.env.step = spy
    try:
        ep, trans = rollout.run_episode(dict(spec, collect=True))
    finally:
        e.env.step = real_step
    steps = np.array(buf, np.float32)
    dec = np.array([np.concatenate([t[1], [t[2], t[3]]]) for t in trans], np.float32)   # features (14), action, reward
    return ep, steps, dec
