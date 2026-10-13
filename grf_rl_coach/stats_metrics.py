"""Football statistics computed from per-engine-step GRF raw observations (pure numpy; no engine needed).

Conventions (GRF raw observation, our coached team = LEFT = team 0, attacking toward +x; opponent = RIGHT = team 1,
attacking toward -x, i.e. at OUR goal x = -1). One 'step' = one engine step; a match has 3001.
  owner[t]   = ball_owned_team in {0 ours, 1 opponent, -1 nobody (loose / in flight / dead)}
  mode[t]    = game_mode (0 normal play; 1 kick-off, 2 goal kick, 3 free kick, 4 corner, 5 throw-in, 6 penalty)
  ball[t]    = (x, y, z), x in [-1, 1] (goal lines), |y| <= 0.42
  dir[t]     = ball_direction, displacement per engine step

Definitions (details and rationale in EVAL_SPEC.md)
  OWNERSHIP vs LOOSE   owner==-1 is a *state*, not a possession change.
  POSSESSION SPELL     maximal run of consecutive steps with owner==k (k in {0,1}) lasting >= min_spell steps.
                       Shorter runs are 'touches' (deflections/tackle bounces) and are ignored.
  POSSESSION TRANSITION consecutive spells with different teams. NOT a transition: same team owns again after a loose phase
                       (a pass, a dribble touch) -- the legacy `possession_gains` counter only caught direct 1->0 flips.
  RECOVERY (open play) opp->us transition with no non-normal game_mode step between the spells (no restart in between).
  RECOVERY (all)       also counts those won back through a restart (goal kick, throw-in, free kick, corner, kick-off).
  OUT-OF-POSSESSION    starts the step after our last owned step of a spell that is followed by an opponent spell; ends at the start
  PERIOD / TIME TO     of our next spell. Right-censored at the end of the match if never regained. Durations in engine steps,
  RECOVER              dead-ball time included.
  SHOT-LIKE EVENT      PROXY (GRF exposes no shot flag): a kick (xy ball-velocity jump >= kick_dv with a player within touch_d) by a
                       player of team k, leaving at speed >= shot_vmin from the attacking part of the pitch, whose straight-line
                       xy extrapolation crosses the attacked goal line within `post` (on target) or `post + y_tol` (attempt).
                       If team k re-owns the ball within k_out steps (before anyone else) it was a pass, not a shot.
"""
import numpy as np

P = dict(min_spell=2,        # steps a run of ownership must last to count as a spell
         third=-1.0 / 3,     # boundary of our defensive third (same thresholds as coach_env.discretize)
         box_x=-0.686, box_y=0.249,   # penalty area: 16.5 m / 52.5 m and 20.16 m / 34 m * 0.42
         post=0.07,          # effective on-target half-width: goal post 0.044 + 0.026 allowance for ball curvature (straight-line extrapolation)
         y_tol=0.08, shot_vmin=0.04, shot_x=0.30,   # attempt tolerance; min speed per step; launch must be beyond |x| >= shot_x toward the goal attacked
         kick_dv=0.025, touch_d=0.05,   # xy-velocity jump that marks a kick; a player must be this close to the ball (xy)
         k_out=15, dedup=5)  # steps to classify a kick's outcome; merge kicks of one team closer than this


def _runs(a):
    """[(value, start, end_inclusive)] for maximal constant runs."""
    out, i, n = [], 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        out.append((int(a[i]), i, j)); i = j + 1
    return out


def spells(owner, min_spell):
    return [(k, s, e) for k, s, e in _runs(owner) if k in (0, 1) and e - s + 1 >= min_spell]


def transitions(sp, mode):
    """One row per change of team between consecutive spells."""
    rows = []
    for (k0, s0, e0), (k1, s1, e1) in zip(sp, sp[1:]):
        if k0 != k1:
            gap = s1 - e0 - 1
            rows.append({"from": k0, "to": k1, "t_last_owned": e0, "t_start": s1, "gap": gap,
                         "stoppage": bool(np.any(mode[e0 + 1:s1] != 0)) if gap > 0 else False})
    return rows


def out_of_possession_periods(sp, mode, T):
    """Our periods without the ball: [(t_loss, duration, censored, open_play)]."""
    res = []
    for i, ((k0, s0, e0), (k1, s1, e1)) in enumerate(zip(sp, sp[1:])):
        if k0 == 0 and k1 == 1:
            t_loss = e0 + 1
            nxt = next((s for k, s, e in sp[i + 2:] if k == 0), None)
            if nxt is None:
                res.append((t_loss, T - t_loss, True, bool(np.all(mode[t_loss:] == 0))))
            else:
                res.append((t_loss, nxt - t_loss, False, bool(np.all(mode[t_loss:nxt] == 0))))
    return res


def shot_events(tr, team, p=P):
    """Shot-like events by `team` (1 = opponent shooting at our goal x=-1; 0 = us at x=+1). A PROXY, see module docstring.
    A KICK is a step with an abrupt change of the ball's xy velocity (>= kick_dv) while a player is within touch_d of the ball;
    it is attributed to that player's team (needed because volleys/first-time shots are struck from a loose ball, where
    ball_owned_team never names the shooter)."""
    mode, ball, vel = tr["mode"], tr["ball"], tr["dir"]
    owner, nt, nd = tr["owner"], tr["near_team"], tr["near_d"]
    T = len(owner); sgn = -1.0 if team == 1 else 1.0          # +1 = attacking toward +x
    ev, last = [], -10 ** 9
    for t in range(1, T - 1):
        if mode[t] != 0 or mode[t - 1] != 0:
            continue
        dv = float(np.linalg.norm(vel[t, :2] - vel[t - 1, :2]))
        if dv < p["kick_dv"]:
            continue
        j = t if nd[t] <= nd[t - 1] else t - 1                # nearest-player evidence from the step before or at the kick
        if nt[j] != team or nd[j] > p["touch_d"]:
            continue
        speed = float(np.linalg.norm(vel[t, :2]))
        vx, vy = float(vel[t, 0]), float(vel[t, 1])
        x, y = float(ball[t, 0]), float(ball[t, 1])
        if speed < p["shot_vmin"] or sgn * x < p["shot_x"] or sgn * vx <= 1e-6:
            continue
        y_cross = y + vy * (sgn * 1.0 - x) / vx
        if abs(y_cross) > p["post"] + p["y_tol"] or t - last < p["dedup"]:
            continue
        outcome = "out"                                     # nobody owned the ball for k_out steps: goal, out of play, loose
        for u in range(t + 1, min(T, t + 1 + p["k_out"])):
            if owner[u] == team:
                outcome = "pass"; break
            if owner[u] == 1 - team:
                outcome = "saved_or_blocked"; break
        if outcome == "pass":
            continue
        last = t
        ev.append({"t": t, "team": team, "x": x, "y": y, "speed": speed, "dv": dv, "near_d": float(nd[j]), "y_cross": float(y_cross),
                   "on_target": bool(abs(y_cross) <= p["post"]), "outcome": outcome})
    return ev


def goal_steps(score):
    """[(step, scoring_team)] where the score changes."""
    out = []
    for t in np.flatnonzero((np.diff(score.astype(int), axis=0) != 0).any(1)) + 1:
        d = score[t].astype(int) - score[t - 1].astype(int)
        for k in (0, 1):
            if d[k] > 0:
                out.append((int(t), k))
    return out


def compute_match(tr, p=P):
    owner, mode, ball, score = tr["owner"], tr["mode"], tr["ball"], tr["score"]
    T = len(owner)
    normal = mode == 0
    n0, n1, nl = int((owner == 0).sum()), int((owner == 1).sum()), int((owner == -1).sum())
    ff = owner.copy()                                         # carry-forward: loose/dead steps stay with the last owner
    last = -1
    for t in range(T):
        if ff[t] == -1:
            ff[t] = last
        else:
            last = ff[t]
    valid = ff >= 0
    m = {"goals_for": int(score[-1, 0]), "goals_against": int(score[-1, 1]), "steps": T,
         "frac_own": n0 / T, "frac_opp": n1 / T, "frac_loose": nl / T, "frac_dead_ball": float((~normal).mean()),
         "poss_owned_share": n0 / max(1, n0 + n1),            # legacy convention: loose steps ignored
         "poss_carry_share": float((ff[valid] == 0).mean()) if valid.any() else float("nan")}
    m["goal_diff"] = m["goals_for"] - m["goals_against"]
    sp = spells(owner, p["min_spell"])
    tr_rows = transitions(sp, mode)
    rec = [r for r in tr_rows if r["from"] == 1 and r["to"] == 0]
    loss = [r for r in tr_rows if r["from"] == 0 and r["to"] == 1]
    m.update(recoveries_open=sum(not r["stoppage"] for r in rec), recoveries_all=len(rec),
             recoveries_direct=sum(r["gap"] == 0 for r in rec), losses_open=sum(not r["stoppage"] for r in loss), losses_all=len(loss),
             n_spells_own=sum(k == 0 for k, _, _ in sp), n_spells_opp=sum(k == 1 for k, _, _ in sp))
    oop = out_of_possession_periods(sp, mode, T)
    done = [d for _, d, c, _ in oop if not c]
    done_open = [d for _, d, c, o in oop if not c and o]
    m.update(oop_n=len(oop), oop_censored=sum(c for _, _, c, _ in oop),
             t_recover_mean=float(np.mean(done)) if done else float("nan"),
             t_recover_open_mean=float(np.mean(done_open)) if done_open else float("nan"))
    # territory (normal play only, so kick-off / goal-kick teleports do not pollute positions)
    x, y = ball[:, 0], ball[:, 1]
    opp_own = (owner == 1) & normal
    m.update(opp_def_third_steps=int((opp_own & (x < p["third"])).sum()),
             any_def_third_steps=int((normal & (x < p["third"])).sum()),
             opp_box_steps=int((opp_own & (x < p["box_x"]) & (np.abs(y) < p["box_y"])).sum()),
             ball_x_mean=float(x[normal].mean()), opp_ball_x_mean=float(x[opp_own].mean()) if opp_own.any() else float("nan"))
    mins, box_entries, def_entries = [], 0, 0
    for k, s, e in sp:
        if k != 1:
            continue
        sl = slice(s, e + 1); nm = normal[sl]
        if not nm.any():
            continue
        mins.append(float(x[sl][nm].min()))
        box_entries += int(((x[sl] < p["box_x"]) & (np.abs(y[sl]) < p["box_y"]) & nm).any())
        def_entries += int(((x[sl] < p["third"]) & nm).any())
    m.update(opp_spell_min_x_mean=float(np.mean(mins)) if mins else float("nan"), opp_box_entries=box_entries, opp_def_third_entries=def_entries,
             opp_max_progress=float(-min(mins)) if mins else float("nan"))
    # shot-like proxy, both directions (ours is descriptive / for validation)
    se1, se0 = shot_events(tr, 1, p), shot_events(tr, 0, p)
    m.update(opp_shot_attempts=len(se1), opp_shots_on_target=sum(e["on_target"] for e in se1),
             own_shot_attempts=len(se0), own_shots_on_target=sum(e["on_target"] for e in se0))
    events = {"spells": sp, "transitions": tr_rows, "oop": oop, "shots_opp": se1, "shots_own": se0, "goals": goal_steps(score)}
    return m, events


def goal_recall(events, window=25):
    """Validation of the shot proxy: of all goals, the fraction preceded (within `window` steps) by an on-target event of the scoring team."""
    hit = tot = 0
    for t, k in events["goals"]:
        tot += 1
        ev = events["shots_opp"] if k == 1 else events["shots_own"]
        hit += any(e["on_target"] and 0 < t - e["t"] <= window for e in ev)
    return hit, tot
