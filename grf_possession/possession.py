"""Possession accounting and the possession-based reward. Pure functions (no simulator), fully unit-tested.

Detector (from GRF raw observations, per engine step):
    owner = obs["ball_owned_team"]  -> -1 nobody owns the ball, 0 left team, 1 right team   (re-labelled to the coached frame:
                                       0 = our team, 1 = opponent, -1 = nobody)
    mode  = obs["game_mode"]        -> 0 normal play; anything else is a dead-ball restart (kick-off, goal kick, free kick,
                                       corner, throw-in, penalty)
Every engine step falls into exactly one class:
    OURS       normal play and we own the ball
    OPP        normal play and the opponent owns the ball
    CONTESTED  normal play and nobody owns the ball (passes in flight, loose and contested balls). NOT possession.
    DEAD       any dead-ball mode. NOT possession, so kick-offs after goals, throw-ins etc. earn nothing.
Controlled-possession share of an interval = OURS / (OURS + OPP), the denominator being controlled steps only.
If an interval has fewer than `min_controlled` controlled steps there is no clear possession: the share is None and the
possession reward is 0.
"""
import numpy as np

OURS, OPP, CONTESTED, DEAD = 0, 1, 2, 3


def classify_steps(owner, game_mode):
    owner = np.asarray(owner); game_mode = np.asarray(game_mode)
    cls = np.full(len(owner), CONTESTED, dtype=np.int8)
    cls[owner == 0] = OURS
    cls[owner == 1] = OPP
    cls[game_mode != 0] = DEAD                       # dead ball overrides whatever owner is reported
    return cls


def interval_stats(owner, game_mode, min_controlled=10):
    """owner/game_mode: per-engine-step arrays for ONE decision interval, in the coached team's frame."""
    cls = classify_steps(owner, game_mode)
    ours, opp = int((cls == OURS).sum()), int((cls == OPP).sum())
    controlled = ours + opp
    share = ours / controlled if controlled >= min_controlled else None
    return {"steps": int(len(cls)), "ours": ours, "opp": opp, "contested": int((cls == CONTESTED).sum()),
            "dead": int((cls == DEAD).sum()), "controlled": controlled, "share": share}


def possession_reward(share, weight=1.0, baseline=0.5):
    """weight * (our controlled-possession share - baseline); 0 when the interval has no clear possession."""
    return 0.0 if share is None else float(weight * (share - baseline))


def goal_reward(goals_for, goals_against, weight=2.0):
    return float(weight * (goals_for - goals_against))


def total_reward(stats, goals_for, goals_against, poss_weight=1.0, goal_weight=2.0, baseline=0.5):
    p = possession_reward(stats["share"], poss_weight, baseline)
    g = goal_reward(goals_for, goals_against, goal_weight)
    return p + g, p, g


def match_possession(per_interval_stats):
    """Whole-match totals from interval stats (sums of steps, so shares are time-weighted, not an average of averages)."""
    ours = sum(s["ours"] for s in per_interval_stats); opp = sum(s["opp"] for s in per_interval_stats)
    contested = sum(s["contested"] for s in per_interval_stats); dead = sum(s["dead"] for s in per_interval_stats)
    controlled = ours + opp
    steps = ours + opp + contested + dead
    return {"ours": ours, "opp": opp, "contested": contested, "dead": dead, "steps": steps,
            "share": ours / controlled if controlled else None,
            "contested_frac": contested / steps if steps else None, "dead_frac": dead / steps if steps else None}
