"""Unit tests for stats_metrics on hand-built traces (no engine).  usage: python -m pytest tests/test_stats_metrics.py  (from grf_rl_coach/)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stats_metrics as sm


def trace(owner, mode=None, x=None, y=None, vx=None, vy=None, score=None, near_team=None):
    T = len(owner)
    ball = np.zeros((T, 3), np.float32)
    if x is not None: ball[:, 0] = x
    if y is not None: ball[:, 1] = y
    vel = np.zeros((T, 3), np.float32)
    if vx is not None: vel[:, 0] = vx
    if vy is not None: vel[:, 1] = vy
    return {"near_team": np.zeros(T, np.int8) if near_team is None else np.array(near_team, np.int8), "near_d": np.full(T, 0.01, np.float32),
            "owner": np.array(owner, np.int8), "mode": np.zeros(T, np.int8) if mode is None else np.array(mode, np.int8),
            "ball": ball, "dir": vel, "score": np.zeros((T, 2), np.int8) if score is None else np.array(score, np.int8)}


def test_direct_and_loose_recovery_vs_same_team_pass():
    o = [1] * 5 + [0] * 5 + [-1] * 3 + [0] * 5 + [-1] * 4 + [1] * 5 + [-1] * 3 + [0] * 5   # direct 1->0 ; 0 loose 0 (pass, no change); 0->loose->1 ; 1->loose->0
    m, ev = sm.compute_match(trace(o))
    assert m["recoveries_all"] == 2 and m["recoveries_direct"] == 1 and m["recoveries_open"] == 2
    assert m["losses_all"] == 1
    assert [(r["from"], r["to"], r["gap"]) for r in ev["transitions"]] == [(1, 0, 0), (0, 1, 4), (1, 0, 3)]


def test_restart_between_spells_is_not_an_open_play_recovery():
    o = [1] * 5 + [-1] * 4 + [0] * 5
    mode = [0] * 5 + [5] * 4 + [0] * 5                                                  # throw-in awarded to us
    m, ev = sm.compute_match(trace(o, mode=mode))
    assert m["recoveries_all"] == 1 and m["recoveries_open"] == 0


def test_touch_shorter_than_min_spell_is_ignored():
    o = [1] * 6 + [0] + [1] * 6                                                         # one-step bounce off our player
    m, _ = sm.compute_match(trace(o))
    assert m["recoveries_all"] == 0 and m["losses_all"] == 0
    assert sm.P["min_spell"] == 2


def test_time_to_recover_and_censoring():
    o = [0] * 4 + [1] * 6 + [-1] * 2 + [0] * 4 + [1] * 5 + [-1] * 3                       # lose at 4 -> regain at 12 (8 steps); lose at 16, never regained
    m, ev = sm.compute_match(trace(o))
    assert [(t, d, c) for t, d, c, _ in ev["oop"]] == [(4, 8, False), (16, len(o) - 16, True)]
    assert m["t_recover_mean"] == 8 and m["oop_censored"] == 1


def test_possession_shares():
    o = [0] * 2 + [-1] * 2 + [0] * 1 + [1] * 3 + [-1] * 2                              # 10 steps: own 3, opp 3, loose 4
    m, _ = sm.compute_match(trace(o))
    assert abs(m["frac_loose"] - 0.4) < 1e-9 and abs(m["poss_owned_share"] - 0.5) < 1e-9
    assert abs(m["poss_carry_share"] - 0.5) < 1e-9                                       # carry-forward owners: [0,0,0,0,0,1,1,1,1,1]


def test_opponent_territory():
    o = [1] * 6 + [-1] * 2
    x = [-0.1, -0.2, -0.4, -0.8, -0.9, -0.5, 0, 0]
    y = [0, 0, 0, 0.0, 0.1, 0.5, 0, 0]
    m, _ = sm.compute_match(trace(o, x=x, y=y))
    assert m["opp_def_third_steps"] == 4          # x < -1/3 while opponent owns: -0.4, -0.8, -0.9, -0.5
    assert m["opp_box_steps"] == 2                # x < -0.686 and |y| < 0.249: steps 3 and 4
    assert m["opp_box_entries"] == 1 and m["opp_def_third_entries"] == 1 and abs(m["opp_max_progress"] - 0.9) < 1e-6


def _shot_trace(team, y0, vy, owner_after):
    """Team releases at x=-0.6 (opp attack toward -x) or +0.6, speed 0.06 along the attack direction."""
    sgn = -1 if team == 1 else 1
    o = [team] * 4 + [-1] * 4 + owner_after
    T = len(o)
    x = np.full(T, 0.6 * sgn); y = np.full(T, y0)
    vx = np.where(np.arange(T) >= 4, 0.06 * sgn, 0.0); vyv = np.where(np.arange(T) >= 4, vy, 0.0)
    return trace(o, x=x, y=y, vx=vx, vy=vyv, near_team=[team] * T)


def test_shot_on_target_wide_and_pass():
    # straight at goal centre
    ev = sm.shot_events(_shot_trace(1, 0.0, 0.0, [1 - 1] * 0 + [0] * 4), 1)                # keeper (our team) owns -> saved
    assert len(ev) == 1 and ev[0]["on_target"] and ev[0]["outcome"] == "saved_or_blocked"
    # drifting 0.5 wide of the post over the remaining 0.4 distance -> not even an attempt
    assert sm.shot_events(_shot_trace(1, 0.0, 0.06, [-1] * 4), 1) == []
    # same speed but re-owned by the releasing team = completed pass, excluded
    assert sm.shot_events(_shot_trace(1, 0.0, 0.0, [1] * 4), 1) == []
    # our own shot goes the other way
    assert len(sm.shot_events(_shot_trace(0, 0.02, 0.0, [-1] * 4), 0)) == 1
    # slow dribble is not a shot
    t = _shot_trace(1, 0.0, 0.0, [0] * 4); t["dir"][:] *= 0.1
    assert sm.shot_events(t, 1) == []


def test_goal_recall():
    t = _shot_trace(1, 0.0, 0.0, [-1] * 6)
    t["score"][10:, 1] = 1
    _, ev = sm.compute_match(t)
    assert sm.goal_recall(ev) == (1, 1)


def test_volley_from_loose_ball_is_attributed_by_nearest_player():
    """Ball is never 'owned' (owner==-1 throughout): the shooter is identified by the player at the ball when the velocity jumps."""
    T = 12
    x = np.full(T, -0.7); vx = np.where(np.arange(T) >= 5, -0.06, -0.012)
    t = trace([-1] * T, x=x, vx=vx, near_team=[1] * T)
    ev = sm.shot_events(t, 1)
    assert len(ev) == 1 and ev[0]["t"] == 5 and ev[0]["on_target"]
    t2 = trace([-1] * T, x=x, vx=vx, near_team=[0] * T)            # same kick by OUR player toward our own goal is not an opponent shot
    assert sm.shot_events(t2, 1) == []
