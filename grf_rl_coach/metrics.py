"""Football-statistic definitions, computed offline from the per-step ball/possession stream (see measure.py).
All thresholds are fixed here BEFORE the secondary evaluation (shot thresholds calibrated on separate seeds 810000+).

Coordinates: our goal is at x = -1, the opponent's at x = +1; "opponent advancement" = -ball_x while they have the ball.
Steps are GRF engine steps (3001 per match).

DEFINITIONS
possession_share_owned : our owned steps / (our + opponent owned steps); loose-ball steps excluded.
loose_share            : share of steps with nobody owning the ball.
debounced possessor    : the team that last owned the ball for >= K consecutive steps. Loose steps and shorter contested
                         touches do NOT change it. This is what removes rapid possession flicker.
recovery (live)        : debounced possessor switches opponent -> us with no dead-ball (game_mode != 0) step and no goal
                         between the opponent's last owned step and our first step of the new possession.
recovery (dead ball)   : same switch but a dead ball intervened (throw-in, goal kick, kick-off ... ): reported separately.
opponent possession spell : from the opponent's debounced start to the next switch (or match end).
time to recover (RMST) : mean over opponent spells of min(duration, CAP); spells that end by dead ball, goal or match end
                         are censored at their length. Also reported: mean duration of completed live recoveries only.
opp_adv_mean           : mean of (-ball_x) over steps the opponent owns the ball. opp_adv_max: max of the same.
opp_third_share        : share of opponent-owned steps with ball_x < -0.33 (our defensive third).
danger_steps           : steps with ball in our danger zone (x < -0.76, |y| < 0.22) and not owned by us.
shot-like event        : a release by team T (T owned the ball, next step it does not, normal play) after which the ball
                         moves toward the goal with speed >= SHOT_SPEED two steps later, from within SHOT_DIST of that goal
                         line; events <= 8 steps apart are merged. On target: projected y at the goal line within +/-0.055.
                         HEURISTIC (GRF exposes no shot events); recall on calibration goals about 80%.
"""
import numpy as np

K = 5
CAP = 150
SHOT_SPEED = 0.040
SHOT_DIST = 0.65       # release must be within this distance of the attacked goal line (release |x| >= 0.35)
ON_TARGET_Y = 0.055
MERGE = 8
OPP, US, LOOSE = 1, 0, -1


def possessor_events(owner, gmode, su, so):
    """Debounced possession switches. Returns list of (start_step, from, to, dead_ball_flag)."""
    n = len(owner)
    poss, cand, cnt, cand_start = None, None, 0, 0
    last_owned = {0: -1, 1: -1}
    events = []
    for t in range(n):
        o = int(owner[t])
        if o in (0, 1):
            last_owned[o] = t
            if o == poss:
                cand, cnt = None, 0
            else:
                if o == cand: cnt += 1
                else: cand, cnt, cand_start = o, 1, t
                if cnt >= K:
                    if poss is not None:
                        a, b = last_owned[poss], cand_start
                        lo, hi = max(a, 0), b
                        dead = bool(np.any(gmode[lo:hi + 1] != 0)) or (su[hi] != su[lo]) or (so[hi] != so[lo])
                        events.append((cand_start, poss, cand, dead))
                    else:
                        events.append((cand_start, None, cand, False))
                    poss, cand, cnt = o, None, 0
        else:
            cand, cnt = None, 0
    return events


def shots(st, team):
    """Shot-like events by `team` (0 = us attacking x=+1, 1 = opponent attacking x=-1). Returns (n_shots, n_on_target)."""
    owner, bx, by, vx, vy, gm = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 5]
    direction = 1.0 if team == 0 else -1.0
    goal_x = direction
    ev = []
    for t in range(1, len(st) - 2):
        if owner[t - 1] == team and owner[t] != team and gm[t] == 0:
            j = t + 2
            speed = float(np.hypot(vx[j], vy[j]))
            if speed < SHOT_SPEED or vx[j] * direction <= 0: continue
            if (goal_x - bx[t - 1]) * direction > SHOT_DIST: continue       # too far from the goal line
            yproj = by[j] + (vy[j] / vx[j]) * (goal_x - bx[j])
            ev.append((t, abs(yproj) <= ON_TARGET_Y))
    merged = []
    for t, ot in ev:
        if merged and t - merged[-1][0] <= MERGE:
            merged[-1] = (merged[-1][0], merged[-1][1] or ot)
        else:
            merged.append((t, ot))
    return len(merged), int(sum(1 for _, ot in merged if ot))


def match_metrics(st):
    owner, bx, by, gm, su, so = st[:, 0], st[:, 1], st[:, 2], st[:, 5], st[:, 6], st[:, 7]
    n = len(st)
    m = {}
    own_us, own_op = int((owner == US).sum()), int((owner == OPP).sum())
    m["possession_share_owned"] = own_us / max(1, own_us + own_op)
    m["loose_share"] = float((owner == LOOSE).mean())
    ev = possessor_events(owner, gm, su, so)
    # debounced possessor timeline
    poss_t = np.full(n, -1)
    cur, idx = -1, 0
    for t in range(n):
        while idx < len(ev) and ev[idx][0] <= t:
            cur = ev[idx][2]; idx += 1
        poss_t[t] = cur
    m["possession_share_debounced"] = float((poss_t == US).sum() / max(1, ((poss_t == US) | (poss_t == OPP)).sum()))
    live = [e for e in ev if e[1] == OPP and e[2] == US and not e[3]]
    dead = [e for e in ev if e[1] == OPP and e[2] == US and e[3]]
    losses = [e for e in ev if e[1] == US and e[2] == OPP]
    m["recoveries_live"] = len(live); m["recoveries_dead_ball"] = len(dead); m["possession_losses"] = len(losses)
    opp_steps = int((poss_t == OPP).sum())
    m["recoveries_live_per_1000_opp_steps"] = 1000.0 * len(live) / max(1, opp_steps)
    m["opp_possession_steps_debounced"] = opp_steps
    # opponent spells: start at switch to OPP, end at the next switch or the match end
    starts = [e[0] for e in ev if e[2] == OPP]
    spells = []
    for s in starts:
        nxt = [e for e in ev if e[0] > s]
        if nxt:
            e2 = nxt[0]
            spells.append((e2[0] - s, e2[2] == US and not e2[3]))
        else:
            spells.append((n - s, False))
    if spells:
        m["time_to_recover_rmst"] = float(np.mean([min(d, CAP) for d, _ in spells]))
        comp = [d for d, ok in spells if ok]
        m["time_to_recover_completed"] = float(np.mean(comp)) if comp else np.nan
        m["opp_spells"] = len(spells); m["opp_spells_censored_frac"] = 1.0 - len(comp) / len(spells)
    else:
        m["time_to_recover_rmst"] = np.nan; m["time_to_recover_completed"] = np.nan; m["opp_spells"] = 0; m["opp_spells_censored_frac"] = np.nan
    opp = owner == OPP
    m["opp_adv_mean"] = float((-bx[opp]).mean()) if opp.any() else np.nan
    m["opp_adv_max"] = float((-bx[opp]).max()) if opp.any() else np.nan
    m["opp_third_share"] = float((bx[opp] < -0.33).mean()) if opp.any() else np.nan
    m["danger_steps"] = int(((bx < -0.76) & (np.abs(by) < 0.22) & (owner != US)).sum())
    m["shots_conceded"], m["shots_on_target_conceded"] = shots(st, OPP)
    m["shots_for"], m["shots_on_target_for"] = shots(st, US)
    m["goals_conceded"] = int(so[-1]); m["goals_scored"] = int(su[-1]); m["goal_diff"] = int(su[-1] - so[-1])
    return m
