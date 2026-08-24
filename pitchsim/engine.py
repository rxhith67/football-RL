"""PITCHSIM — the real engine. Tier 1 of the Phase-1 build.

Batched over matches from the first line, because that is the only way to reach
the 300 matches/min target: the tick loop runs in Python, so the fixed
per-iteration overhead has to be amortised across a batch rather than paid once
per match (design note 3.2 and 7).

Array layout throughout: `(B, 2, 11, ...)` for [match, side, player].
Home (side 0) attacks +x with its goal at x=0; away (side 1) attacks -x.

Tier 1 (this file, gated at Gate 1)
    movement and formation shape, ball model, EV-based option selection,
    xG-sampled goals, tackles, out-of-play restarts.

Tier 2 (deliberately NOT gated; lands during Phase 2)
    fouls and cards, set-piece routines, richer duel modelling.
    Hooks are marked `TIER 2` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import controller as ctrl
from . import epv as epv_mod
from .base import PitchSim
from .pitch_control import (
    control_probability,
    defenders_in_cone,
    pass_success_probability,
    shot_geometry,
)
from .replay import ReplayBuilder
from .types import (
    MatchResult,
    MatchState,
    SegmentResult,
    Side,
    SquadState,
    Tactic,
    WindowStats,
)

PITCH_L, PITCH_W = 105.0, 68.0
HOME, AWAY = 0, 1
GOAL_X = np.array([0.0, PITCH_L], dtype=np.float32)      # each side's OWN goal
ATTACK_X = np.array([PITCH_L, 0.0], dtype=np.float32)    # goal each side attacks

# Option indices in the event layer's choice set.
OPT_PASS_BASE, N_PASS = 0, 11        # one per teammate (the carrier is masked out)
OPT_SHOT, OPT_DRIBBLE, OPT_CLEAR, OPT_THROUGH = 11, 12, 13, 14
N_OPTIONS = 15

#: Style axes read off the tactic heads, each mapped to {-1, 0, +1}:
#: territory (block height), intensity (press), directness (tempo).
_AXIS = np.array([-1.0, 0.0, 1.0])


def matchup_edge(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Rock-paper-scissors advantage of `own` tactics over `opp` tactics.

    **This is a designed term, not an emergent one, and the write-up must say so.**
    design note 3.5 is explicit that the counters belong in the EV score rather
    than being left to emerge, and the honest framing is: *we built an environment
    that admits non-transitivity, then asked whether the RL coach discovers and
    exploits it.* Never "we discovered football is non-transitive".

    The four relationships are the ones football people actually describe, and
    they interlock into two cycles:

        pressing        beats  patient possession   (turnovers in build-up)
        direct play     beats  a high line          (balls in behind)
        patient posses. beats  a low block          (territory finds the gap)
        a deep block    beats  direct play          (bodies absorb route one)

    Purely physical modelling produced a strictly transitive pecking order
    ordered by block height — high beat mid beat low — with no cycle at all, so
    the emergent route was tried first and is documented as having failed.

    `own`, `opp` are (m, 5) tactic head arrays. Returns (m,) in [-1, 1].

    Construction
    ------------
    Each tactic is placed on a *style circle* by

        v = (block,  press - tempo)

    and the edge is the normalised cross product

        edge(a, b) = gain * (v_a x v_b) / 5

    which equals sin(theta_b - theta_a): a tactic beats whatever sits a quarter
    turn anticlockwise of it, and is neutral against its opposite.

    The cross product is **skew-symmetric** — edge(a, b) = -edge(b, a) — so no
    tactic can dominate the field, and cyclic structure is guaranteed rather than
    hoped for. An earlier additive form (press_vs_patient + direct_vs_high_line +
    ...) was *not* skew-symmetric: it simultaneously rewarded pressing against
    patient sides and punished containing against them, which compounded into
    Possession beating Low-block by 69 goals and left the matrix transitive.

    The resulting angles land where football intuition puts them:
    high press 0 deg, possession 45 deg, low block 180 deg, counter 225 deg,
    direct 270 deg, with a balanced side at the origin and therefore neutral.
    """
    own_block, own_press, own_tempo = _AXIS[own[:, 1]], _AXIS[own[:, 2]], _AXIS[own[:, 3]]
    opp_block, opp_press, opp_tempo = _AXIS[opp[:, 1]], _AXIS[opp[:, 2]], _AXIS[opp[:, 3]]

    own_x, own_y = own_block, own_press - own_tempo
    opp_x, opp_y = opp_block, opp_press - opp_tempo

    # |v| <= sqrt(1 + 4), so the cross product is bounded by 5.
    return (own_x * opp_y - own_y * opp_x) / 5.0


#: Fraction of the space between the offside line and the goal that a through
#: ball is played into. This MUST scale with the available space rather than be a
#: fixed number of metres: against a high line the ball has to travel a long way
#: to reach a dangerous area, and a fixed 11 m gap dropped it 47 m from goal —
#: harmless, which is why a high line stayed strictly dominant.
THROUGH_FRACTION = 0.55
#: Never played closer than this to goal; beyond it the keeper simply collects.
THROUGH_MIN_GOAL_DISTANCE = 9.0

#: Minimum space behind the defensive line, in metres, for a through ball to be
#: available at all. This is what makes the option a SPECIFIC punishment for
#: playing a high line rather than a general chance generator: against a deep
#: block there is no space to run into and the option is simply masked out.
#: Without it, the only way to keep the sim's goal rate realistic was to play the
#: ball so shallow (15% of the space) that it landed 41 m from goal and punished
#: nobody — which is why high block stayed dominant and no cycle appeared.
THROUGH_MIN_SPACE = 34.0

# All free parameters of the event and duel layers now live in the `tuning:`
# block of config/pitchsim.yaml, because they are coupled and must be swept
# jointly rather than tuned one at a time. See `pitchsim.calibrate`.


@dataclass
class _Counters:
    """Cumulative per-match, per-side tallies. Window stats are deltas of these."""

    shots: np.ndarray
    xg: np.ndarray
    passes: np.ndarray
    passes_completed: np.ndarray
    pass_distance: np.ndarray        # summed metres; /passes gives the profile
    final_third_touches: np.ndarray
    defensive_actions: np.ndarray
    possession_ticks: np.ndarray

    @classmethod
    def zeros(cls, batch: int) -> "_Counters":
        z = lambda: np.zeros((batch, 2), dtype=np.float64)  # noqa: E731
        return cls(z(), z(), z(), z(), z(), z(), z(), z())

    def snapshot(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).copy() for k in self.__dataclass_fields__}


class BatchedEngine:
    """Simulates `batch` independent matches in lockstep."""

    def __init__(self, batch: int = 32, config: dict | None = None) -> None:
        _, physics = ctrl.load_config()
        self.cfg = config or physics
        self.batch = batch

        self.dt = float(self.cfg["time"]["dt"])
        self.match_seconds = float(self.cfg["time"]["match_minutes"]) * 60.0
        players = self.cfg["players"]
        self.accel = float(players["accel_mps2"])
        self.speed_floor = float(players["speed_at_zero_stamina"])
        self.drain_base = float(players["stamina_drain_base"])
        self.recovery = float(players["stamina_recovery"])
        self.speed_range = tuple(players["max_speed_mps"])

        ball = self.cfg["ball"]
        self.ball_decel = float(ball["decel_mps2"])
        self.pass_speed = float(ball["max_pass_speed_mps"])

        ev = self.cfg["event_layer"]
        self.w_control = float(ev["ev_weights"]["pitch_control"])
        self.w_epv = float(ev["ev_weights"]["epv_gain"])
        self.w_risk = float(ev["ev_weights"]["risk"])
        self.reaction = float(ev["pitch_control"]["reaction_time_s"])
        self.sigma = float(ev["pitch_control"]["sigma_logistic"])
        self.decision_window = tuple(self.cfg["time"]["decision_interval_s"])
        # Decisions use the calibrated mean interval on deterministic per-match
        # clocks. They stay aligned during ordinary play but may diverge after an
        # individual match stops early for a goal-triggered coaching decision.
        # The clocks are not random: stochastic per-match intervals fragmented
        # almost every event-layer call and defeated BatchedEngine vectorisation.
        self.decision_interval = float(np.mean(self.decision_window))

        tune = self.cfg.get("tuning", {})
        self.score_gain = float(tune.get("score_gain", 45.0))
        self.shot_cost = float(tune.get("shot_cost", 0.030))
        self.epv_max = float(tune.get("epv_max", 0.45))
        self.carry_speed = float(tune.get("carry_speed", 0.72))
        self.tackle_base = float(tune.get("tackle_base", 1.45))
        self.tackle_range = float(tune.get("tackle_range", 2.2))
        self.collapse_near = float(tune.get("collapse_near", 20.0))
        self.collapse_mid = float(tune.get("collapse_mid", 35.0))
        self.matchup_gain = float(tune.get("matchup_gain", 0.0))
        self.shape_follow = float(tune.get("shape_follow", 0.55))
        self.through_fraction = float(tune.get("through_fraction", THROUGH_FRACTION))

        self._xg = self._load_xg()
        self.rng = np.random.default_rng()

    @staticmethod
    def _load_xg():
        from .xg import XGModel

        try:
            return XGModel.load()
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, tactics: np.ndarray | None = None) -> None:
        """Start `batch` fresh matches.

        `tactics` is (B, 2, 5) of head indices; defaults to balanced 4-4-2.
        """
        self.rng = np.random.default_rng(seed)
        b = self.batch

        self.tactics = (
            np.zeros((b, 2, 5), dtype=np.int64) + np.array([4, 1, 1, 1, 0])
            if tactics is None else tactics.astype(np.int64).copy()
        )

        lo, hi = self.speed_range
        self.max_speed = self.rng.uniform(lo, hi, size=(b, 2, 11)).astype(np.float32)
        self.max_speed[:, :, 0] = lo * 0.85          # keepers are slower
        self.stamina = np.ones((b, 2, 11), dtype=np.float32)

        self._refresh_tactic_cache()
        self.pos = self._slots.copy()
        self.vel = np.zeros_like(self.pos)

        self.ball = np.tile(np.array([PITCH_L / 2, PITCH_W / 2], np.float32), (b, 1))
        self.possessing = self.rng.integers(0, 2, size=b)
        self.carrier = np.full(b, 9, dtype=np.int64)          # a forward kicks off
        #: True only while the carrier is actively dribbling (see `_drive_carrier`).
        self.carrying = np.zeros(b, dtype=bool)

        self.in_flight = np.zeros(b, dtype=bool)
        self.flight_start = np.zeros(b, dtype=np.float64)
        self.flight_end = np.zeros(b, dtype=np.float64)
        self.flight_from = self.ball.copy()
        self.flight_to = self.ball.copy()
        self.flight_side = np.zeros(b, dtype=np.int64)
        self.flight_player = np.zeros(b, dtype=np.int64)
        self.flight_ok = np.zeros(b, dtype=bool)

        self.t = np.zeros(b, dtype=np.float64)
        self.next_decision_time = np.full(b, self.decision_interval, dtype=np.float64)
        self.score = np.zeros((b, 2), dtype=np.int64)
        self.counters = _Counters.zeros(b)
        self._window_mark = self.counters.snapshot()

        self.yellow_cards = np.zeros((b, 2), dtype=np.int64)   # TIER 2
        self.red_cards = np.zeros((b, 2), dtype=np.int64)      # TIER 2
        self.subs_used = np.zeros((b, 2), dtype=np.int64)
        self.windows_used = np.zeros((b, 2), dtype=np.int64)

        self._goal_flags = np.zeros((b, 2), dtype=np.int64)
        self._recorders: dict[int, ReplayBuilder] = {}
        self._tick_counter = 0

    def set_tactics(self, tactics: np.ndarray) -> None:
        self.tactics = tactics.astype(np.int64)
        self._refresh_tactic_cache()

    def _refresh_tactic_cache(self) -> None:
        """Recompute everything that depends only on the tactic heads.

        Called at reset and at each decision epoch — not per tick.
        """
        form = self.tactics[:, :, 0]
        self._heads = (form, self.tactics[:, :, 1],
                       self.tactics[:, :, 2], self.tactics[:, :, 3])
        self._slots = ctrl.base_slots(form)

    def attach_recorder(self, index: int, builder: ReplayBuilder) -> None:
        self._recorders[index] = builder

    # ------------------------------------------------------------------
    # The tick loop
    # ------------------------------------------------------------------

    def run_until(self, target_seconds: float | np.ndarray,
                  stop_on_goal: bool = True) -> np.ndarray:
        """Advance each match to its target, or its own next goal.

        Returns the per-match goal tally accumulated during this call, so the
        SMDP wrapper can fire its `goal_for` / `goal_against` triggers. A goal in
        one batch row never stops unrelated matches.
        """
        self._goal_flags[:] = 0
        target = np.asarray(target_seconds, dtype=np.float64)
        if target.ndim == 0:
            target = np.full(self.batch, float(target), dtype=np.float64)
        if target.shape != (self.batch,):
            raise ValueError(f"target_seconds must be scalar or shape ({self.batch},)")
        # Half-tick tolerance on the bound. `t` accumulates by repeated addition,
        # so it lands a hair under the target and a bare `<` runs one extra tick,
        # overshooting full time by dt and leaving `t` at 5400.4 rather than 5400.
        limit = np.minimum(target, self.match_seconds) - 0.5 * self.dt
        stopped = np.zeros(self.batch, dtype=bool)
        while True:
            active = (self.t < limit) & ~stopped
            if not active.any():
                break
            self._tick(active)
            if stop_on_goal:
                stopped |= self._goal_flags.any(axis=1)
        return self._goal_flags.copy()

    def _tick(self, active: np.ndarray) -> None:
        dt = self.dt
        form, block, press, tempo = self._heads

        # --- shape and movement -------------------------------------------------
        # Slots depend only on formation, which changes at decision epochs, so
        # they are cached rather than rebuilt 27,000 times per match.
        targets = ctrl.target_points(self._slots, self.ball, self.possessing,
                                     form, block, press, tempo, self.shape_follow)
        targets = ctrl.press_targets(targets, self.pos, self.ball, self.possessing,
                                     press, self.collapse_near, self.collapse_mid)
        # `closing_speed` is a pressing/tackle modifier, not a permanent team
        # speed multiplier.  Applying it to all eleven players made contain
        # teams move at 60% speed and gegenpress teams at 155%, creating both
        # ball-swarming and a dominant tactic.  Chase targets and tackle rates
        # already express press intensity; ordinary movement stays attribute-
        # driven here.
        closing = np.ones_like(self.max_speed)
        self._drive_carrier(targets, closing)
        self.pos, self.vel = ctrl.step_positions(
            self.pos, self.vel, targets, self.max_speed, self.stamina,
            dt, self.accel, self.speed_floor, closing,
        )
        self.stamina = ctrl.update_stamina(
            self.stamina, self.vel, self.max_speed, press, self.possessing,
            dt, self.drain_base, self.recovery,
        )

        # --- ball ---------------------------------------------------------------
        self._advance_ball(dt)

        # --- possession bookkeeping --------------------------------------------
        held = ~self.in_flight
        self.counters.possession_ticks[held, self.possessing[held]] += 1
        attacking_x = ATTACK_X[self.possessing]
        in_final_third = np.abs(self.ball[:, 0] - attacking_x) < PITCH_L / 3.0
        touch = held & in_final_third
        self.counters.final_third_touches[touch, self.possessing[touch]] += 1

        # --- tackles ------------------------------------------------------------
        self._resolve_tackles(active & held, press)

        # --- on-ball decisions --------------------------------------------------
        self.t += dt * active
        # Evaluate the event layer only on the shared decision clock.  A match
        # whose ball is still in flight waits for the next clock edge; resolving
        # it immediately on landing fragmented one batched call into hundreds of
        # tiny calls and cut throughput by an order of magnitude.
        crossed = active & (self.t + 0.5 * dt >= self.next_decision_time)
        due = crossed & held
        if due.any():
            self._resolve_decisions(np.flatnonzero(due))
        if crossed.any():
            # Usually one interval; the floor also makes this correct if a
            # caller restores or advances a row across several clock edges.
            jumps = (np.floor(
                (self.t[crossed] + 0.5 * dt - self.next_decision_time[crossed])
                / self.decision_interval
            ).astype(np.int64) + 1)
            self.next_decision_time[crossed] += jumps * self.decision_interval

        self._record(active)

    # ------------------------------------------------------------------

    def _advance_ball(self, dt: float) -> None:
        """Ball is either carried by its holder or in flight to a resolved target."""
        flying = self.in_flight
        if flying.any():
            idx = np.flatnonzero(flying)
            arrived = self.t[idx] >= self.flight_end[idx]

            # Linear interpolation along the pass; good enough at 5 Hz and it keeps
            # the console animation smooth.
            total = np.maximum(self.flight_end[idx] - self.flight_start[idx], 1e-6)
            frac = np.clip((self.t[idx] - self.flight_start[idx]) / total, 0.0, 1.0)
            self.ball[idx] = (self.flight_from[idx]
                              + (self.flight_to[idx] - self.flight_from[idx])
                              * frac[:, None])

            landed = idx[arrived]
            if landed.size:
                self._complete_pass(landed)

        carried = ~self.in_flight
        if carried.any():
            idx = np.flatnonzero(carried)
            self.ball[idx] = self.pos[idx, self.possessing[idx], self.carrier[idx]]

        self._handle_out_of_play()

    def _drive_carrier(self, targets: np.ndarray, closing: np.ndarray) -> None:
        """A player who CHOSE to dribble runs at the goal rather than at his slot.

        Gated on `self.carrying`, which is set only by the dribble option and
        cleared the moment the ball is passed, shot or lost. Driving every carrier
        goal-ward unconditionally (an earlier version did) meant even centre-backs
        charged the length of the pitch, and the median shot came from five metres.

        `CARRY_SPEED` is the penalty for having the ball at your feet — without it
        a dribbler simply outruns the defence.
        """
        held = ~self.in_flight & self.carrying
        if not held.any():
            return
        idx = np.flatnonzero(held)
        side = self.possessing[idx]
        player = self.carrier[idx]
        here = self.pos[idx, side, player]

        goal = np.stack([ATTACK_X[side], np.full(idx.size, PITCH_W / 2.0)], axis=-1)
        direction = goal - here
        distance = np.linalg.norm(direction, axis=-1, keepdims=True)
        targets[idx, side, player] = here + direction / np.maximum(distance, 1e-6) * 12.0
        closing[idx, side, player] *= self.carry_speed

    def _complete_pass(self, idx: np.ndarray) -> None:
        self.in_flight[idx] = False
        good = self.flight_ok[idx]

        winners = np.where(good, self.flight_side[idx], 1 - self.flight_side[idx])
        players = self.flight_player[idx].copy()

        # An intercepted ball goes to whichever opponent is closest to it.
        bad = idx[~good]
        if bad.size:
            opp = 1 - self.flight_side[bad]
            distance = np.linalg.norm(
                self.pos[bad, opp] - self.ball[bad][:, None, :], axis=-1
            )
            players[~good] = np.argmin(distance, axis=-1)

        self.possessing[idx] = winners
        self.carrier[idx] = players
        self.carrying[idx] = False
        self.counters.passes_completed[idx[good], self.flight_side[idx[good]]] += 1

    def _in_ppda_zone(self, matches: np.ndarray, defending: np.ndarray) -> np.ndarray:
        """Whether a defensive action belongs in the PPDA denominator.

        PPDA counts actions outside the defending team's deepest 40% of the
        pitch.  Counting box clearances and low-block tackles made every style
        saturate at the display floor and erased the press-intensity signal.
        """
        ball_x = self.ball[matches, 0]
        progress = np.where(defending == HOME, ball_x, PITCH_L - ball_x)
        return progress >= 0.40 * PITCH_L

    def _handle_out_of_play(self) -> None:
        """Restart on a touchline or goal-line exit.

        TIER 2 will replace these with real set-piece routines; for now a
        throw-in or goal kick simply hands possession over at the boundary.
        """
        out_y = (self.ball[:, 1] <= 0.05) | (self.ball[:, 1] >= PITCH_W - 0.05)
        out_x = (self.ball[:, 0] <= 0.05) | (self.ball[:, 0] >= PITCH_L - 0.05)
        out = out_x | out_y
        if not out.any():
            return

        idx = np.flatnonzero(out)
        self.in_flight[idx] = False
        self.possessing[idx] = 1 - self.possessing[idx]
        self.carrier[idx] = self.rng.integers(1, 11, size=idx.size)
        self.carrying[idx] = False
        np.clip(self.ball[idx, 0], 1.0, PITCH_L - 1.0, out=self.ball[idx, 0])
        np.clip(self.ball[idx, 1], 1.0, PITCH_W - 1.0, out=self.ball[idx, 1])
        self.pos[idx, self.possessing[idx], self.carrier[idx]] = self.ball[idx]

    def _resolve_tackles(self, mask: np.ndarray, press: np.ndarray) -> None:
        """Defenders near the carrier can win the ball; pressing raises the rate."""
        if not mask.any():
            return
        idx = np.flatnonzero(mask)
        defending = 1 - self.possessing[idx]

        distance = np.linalg.norm(
            self.pos[idx, defending] - self.ball[idx][:, None, :], axis=-1
        )
        nearest = np.argmin(distance, axis=-1)
        gap = distance[np.arange(idx.size), nearest]

        closing = ctrl.modifier_tables()["closing_speed"][press[idx, defending]]
        carrier_stamina = self.stamina[idx, self.possessing[idx], self.carrier[idx]]
        tackler_stamina = self.stamina[idx, defending, nearest]

        # Rate rises sharply once the defender is within about two metres.
        #
        # The tactical matchup is deliberately NOT applied here. Routing it through
        # tackles worked for the payoff matrix but corrupted the belief filter:
        # PPDA is defined as opponent passes per DEFENSIVE ACTION, so scaling the
        # tackle rate by the matchup scrambles the one observable that identifies
        # press intensity. Archetype separability fell 0.768 -> 0.536 and Direct
        # 3-5-2 was misread as High-press 4-3-3 41% of the time. The edge is
        # applied to chance quality instead, which no observable feature reads.
        rate = (self.tackle_base * closing * np.exp(-gap / self.tackle_range)
                * (0.6 + 0.8 * tackler_stamina) / (0.6 + 0.8 * carrier_stamina))
        won = self.rng.random(idx.size) < rate * self.dt

        if won.any():
            taken = idx[won]
            self.possessing[taken] = defending[won]
            self.carrier[taken] = nearest[won]
            self.carrying[taken] = False

        # TIER 2: a failed tackle inside the box should risk a foul and a penalty.

    # ------------------------------------------------------------------
    # Event layer
    # ------------------------------------------------------------------

    def _resolve_decisions(self, idx: np.ndarray) -> None:
        """Score every on-ball option and sample one, for the matches in `idx`."""
        m = idx.size
        own = self.possessing[idx]
        opp = 1 - own
        rows = np.arange(m)

        own_pos = self.pos[idx, own]                       # (m, 11, 2)
        own_vel = self.vel[idx, own]
        opp_pos = self.pos[idx, opp]
        opp_vel = self.vel[idx, opp]
        own_speed = self.max_speed[idx, own]
        opp_speed = self.max_speed[idx, opp]

        carrier = self.carrier[idx]
        carrier_pos = own_pos[rows, carrier]               # (m, 2)
        attacking_x = ATTACK_X[own]
        attacks_positive = own == HOME

        tempo = self.tactics[idx, own, 3]
        mods = ctrl.modifier_tables()
        temperature = mods["softmax_temp"][tempo]
        risk_weight = mods["risk_weight"][tempo]
        directness = mods["directness"][tempo]

        # PPDA denominator: pressure actions in the attacking 60% of the pitch.
        # Counting only successful tackles saturated at the same value for every
        # tactic because close-range duel probabilities are intentionally high.
        # A pressure is an observable defensive action whether or not it wins the
        # ball; the tactic's trigger radius therefore produces the signal PPDA is
        # designed to measure without changing match outcomes.
        defending_press = self.tactics[idx, opp, 2]
        pressure_radius = mods["trigger_radius"][defending_press]
        nearest_pressure = np.min(
            np.linalg.norm(opp_pos - carrier_pos[:, None, :], axis=-1), axis=-1)
        pressured = ((nearest_pressure <= pressure_radius)
                     & self._in_ppda_zone(idx, opp))
        if pressured.any():
            self.counters.defensive_actions[idx[pressured], opp[pressured]] += 1

        # --- pass options: one per teammate ------------------------------------
        # All eleven candidate receivers in ONE call. Looping in Python here cost
        # more than the rest of the tick combined.
        # Tactical matchup: a designed rock-paper-scissors term (see
        # `matchup_edge`).  It must not alter any belief-filter observable.  In
        # particular, scaling pass retention here leaks the hidden matchup into
        # completion, possession and PPDA.  The edge is applied only to shot
        # conversion in `_execute`; recorded xG and all build-up statistics stay
        # generated by football mechanics alone.
        edge = matchup_edge(self.tactics[idx, own], self.tactics[idx, opp])

        destination_control = control_probability(
            own_pos, own_vel, opp_pos, opp_vel, own_pos,
            own_speed, opp_speed, self.reaction, self.sigma,
        )
        control = pass_success_probability(
            carrier_pos, own_pos, opp_pos, opp_vel, opp_speed,
            destination_control, self.pass_speed, self.reaction, self.sigma,
        )

        target_epv = epv_mod.lookup(own_pos, attacks_positive[:, None], self.epv_max)
        here_epv = epv_mod.lookup(carrier_pos, attacks_positive, self.epv_max)

        travel = np.linalg.norm(own_pos - carrier_pos[:, None, :], axis=-1)

        # Every option is scored on ONE currency: the probability this possession
        # ends in a goal. A pass is worth `P(arrives) x EPV(there)`, minus the cost
        # of handing over possession where we currently stand. Mixing in a
        # separate constant turnover penalty (an earlier version added 0.35) drove
        # every pass score to roughly zero and left shooting competitive from
        # anywhere on the pitch — 175 shots per team per match.
        turnover_cost = risk_weight[:, None] * (1.0 - control) * (
            0.5 * here_epv[:, None] + 0.02
        )
        long_ball_bonus = directness[:, None] * (travel / PITCH_L) * 0.05

        scores = np.full((m, N_OPTIONS), -np.inf)
        scores[:, OPT_PASS_BASE:OPT_PASS_BASE + N_PASS] = (
            self.w_control * control * target_epv
            - self.w_risk * turnover_cost
            + long_ball_bonus
        )
        scores[rows, OPT_PASS_BASE + carrier] = -np.inf     # cannot pass to yourself
        scores[:, OPT_PASS_BASE] = -np.inf                  # nor back to the keeper here

        # --- offside -----------------------------------------------------------
        # Without this rule strikers simply camp in the six-yard box, every pass
        # targets the highest-EPV player standing on the goal line, and the median
        # shot comes from five metres. Offside is the constraint that makes a
        # defensive line a real tactical object, and it is what gives the
        # `offside_trap` modifier of a high block something to do.
        sign = np.where(attacks_positive, 1.0, -1.0)[:, None]
        opp_line = np.sort(sign * opp_pos[:, :, 0], axis=1)[:, -2]   # second-last defender
        limit = np.maximum(opp_line, sign[:, 0] * self.ball[idx, 0])
        beyond = (sign * own_pos[:, :, 0]) > limit[:, None]
        scores[:, OPT_PASS_BASE:OPT_PASS_BASE + N_PASS][beyond] = -np.inf

        # --- shot ---------------------------------------------------------------
        distance, angle = shot_geometry(carrier_pos, goal_x=attacking_x)
        # The xG model was fitted on real shots, none of which were taken from
        # zero metres. Clamping keeps evaluation inside its support instead of
        # extrapolating to a near-certain goal.
        distance = np.maximum(distance, 2.0)
        blockers = defenders_in_cone(carrier_pos, opp_pos, goal_x=attacking_x)
        # NOTE: the matchup edge deliberately does NOT scale shot quality. A
        # favourable tactical matchup should buy territory and sustained pressure,
        # not better finishing from the same position. Applying it to xG as well
        # double-counted and inflated goals/match to 5.4 across the tactic space,
        # while pushing the sweep to a physically absurd 28 tackles/second to
        # compensate.
        # Honest chance quality: NOT scaled by the tactical matchup. `xg_for` and
        # `xg_against` are two of the belief filter's nine observable features, so
        # scaling xG here leaks the matchup into the very signal used to identify
        # the opponent — the same mistake as routing it through the tackle rate
        # and PPDA. At gain 3.2 it drove Direct 3-5-2's recall down to 0.40.
        # The matchup is applied to CONVERSION instead (see `_execute`), which
        # moves the payoff matrix without touching anything observable.
        shot_xg = self._expected_goals(distance, angle, blockers)
        shot_edge = edge
        # Shooting is terminal and EPV is on the xG scale, so the two compare
        # directly. The penalty is the cost of surrendering possession on a miss.
        scores[:, OPT_SHOT] = shot_xg - self.shot_cost

        # --- dribble ------------------------------------------------------------
        step = np.stack([np.sign(attacking_x - carrier_pos[:, 0]) * 6.0,
                         np.zeros(m)], axis=-1)
        ahead = carrier_pos + step
        dribble_epv = epv_mod.lookup(ahead, attacks_positive, self.epv_max)
        nearest_opp = np.min(np.linalg.norm(opp_pos - carrier_pos[:, None, :], axis=-1), axis=-1)
        keep = 1.0 / (1.0 + np.exp(-(nearest_opp - 3.0) / 1.5))
        scores[:, OPT_DRIBBLE] = (keep * dribble_epv
                                  - risk_weight * (1 - keep) * (0.5 * here_epv + 0.02))

        # --- clearance: only a real option when penned in ------------------------
        under_pressure = nearest_opp < 3.0
        deep = np.abs(carrier_pos[:, 0] - GOAL_X[own]) < 30.0
        scores[:, OPT_CLEAR] = np.where(under_pressure & deep, 0.5 * here_epv, -np.inf)

        # --- through ball: the punishment for playing a high line ---------------
        # Offside alone makes a high line *safer* — it simply masks passes beyond
        # the last defender — which is backwards. In real football a high line
        # concedes the space behind it, and that is the whole reason pressing
        # sides are vulnerable to direct play. Without this option the payoff
        # matrix stayed strictly transitive and Pillar 2 had no result.
        #
        # The ball is played into space `THROUGH_GAP` metres beyond the offside
        # line, and a runner starting from an ONSIDE position races the recovering
        # defenders and the keeper to it. A high line leaves a large, poorly
        # defended area; a low block leaves almost none, so the same option is
        # worthless against it. That asymmetry is what creates the cycle.
        # Play into the space proportionally: a high line concedes a large area
        # and the ball is driven deep into it; a low block concedes almost none.
        goal_line = sign[:, 0] * attacking_x
        space = np.maximum(goal_line - opp_line, 0.0)
        through_proj = np.minimum(
            opp_line + self.through_fraction * space,
            goal_line - THROUGH_MIN_GOAL_DISTANCE
        )
        through_x = np.clip(sign[:, 0] * through_proj, 2.0, PITCH_L - 2.0)

        onside = ~beyond
        onside[rows, carrier] = False
        advancement = np.where(onside, sign * own_pos[:, :, 0], -np.inf)
        runner = np.argmax(advancement, axis=1)
        has_runner = np.isfinite(advancement[rows, runner])

        runner_pos = own_pos[rows, runner]
        target = np.stack([through_x, runner_pos[:, 1]], axis=-1)

        run_time = self.reaction + (np.linalg.norm(target - runner_pos, axis=-1)
                                    / np.maximum(own_speed[rows, runner], 1e-6))
        # Outfield defenders only. The goalkeeper is not part of this race — he is
        # already priced into the xG of the chance that follows, and counting him
        # here charged the runner twice and made every through ball fail.
        defend_time = (self.reaction
                       + np.min(np.linalg.norm(opp_pos[:, 1:] - target[:, None, :], axis=-1)
                                / np.maximum(opp_speed[:, 1:], 1e-6), axis=1))
        wins_race = 1.0 / (1.0 + np.exp(-(defend_time - run_time) / max(self.sigma, 1e-6)))

        # The ball must also survive the lane to that space.
        lane_clear = pass_success_probability(
            carrier_pos, target[:, None, :], opp_pos, opp_vel, opp_speed,
            np.ones((m, 1)), self.pass_speed, self.reaction, self.sigma,
        )[:, 0]

        through_epv = epv_mod.lookup(target, attacks_positive, self.epv_max)
        through_score = (wins_race * lane_clear * through_epv
                         + directness * 0.03
                         - risk_weight * (1.0 - wins_race * lane_clear) * 0.5 * here_epv)
        has_space = space >= THROUGH_MIN_SPACE
        scores[:, OPT_THROUGH] = np.where(has_runner & has_space, through_score, -np.inf)

        # --- sample -------------------------------------------------------------
        # SCORE_GAIN converts option values (which live on the ~0.01-0.3 xG scale)
        # into logits with real spread. Without it the softmax is nearly uniform
        # and the controller picks at random — which showed up immediately as 150
        # shots per team per match.
        finite = np.isfinite(scores)
        shifted = np.where(
            finite, scores * self.score_gain / np.maximum(temperature[:, None], 1e-3), -np.inf
        )
        shifted -= shifted.max(axis=1, keepdims=True)
        weights = np.where(finite, np.exp(shifted), 0.0)
        weights /= weights.sum(axis=1, keepdims=True)

        draws = self.rng.random(m)
        choice = (weights.cumsum(axis=1) < draws[:, None]).sum(axis=1)
        choice = np.clip(choice, 0, N_OPTIONS - 1)

        through = {"target": target, "runner": runner,
                   "success": wins_race * lane_clear, "edge": shot_edge}
        self._execute(idx, choice, control, shot_xg, own, carrier_pos, through)

    def _expected_goals(self, distance, angle, blockers) -> np.ndarray:
        if self._xg is None:
            return np.clip(np.exp(-distance / 8.0) * (angle / np.pi) ** 0.5, 1e-4, 0.95)
        names = list(self._xg.feature_names)
        rows = np.tile(
            np.array([[self._xg.DEFAULTS[n] for n in names]]), (distance.size, 1)
        )
        rows[:, names.index("distance_m")] = distance
        rows[:, names.index("angle_rad")] = angle
        if "defenders_in_cone" in names:
            rows[:, names.index("defenders_in_cone")] = blockers
        return self._xg.predict(rows)

    def _execute(self, idx, choice, control, shot_xg, own, carrier_pos, through) -> None:
        rows = np.arange(idx.size)

        # ---- passes ----
        is_pass = choice < N_PASS
        if is_pass.any():
            sel = rows[is_pass]
            match = idx[is_pass]
            receiver = choice[is_pass]
            target = self.pos[match, own[is_pass], receiver]

            success = self.rng.random(sel.size) < control[sel, receiver]
            distance = np.linalg.norm(target - carrier_pos[is_pass], axis=-1)
            flight = np.clip(distance / self.pass_speed + 0.15, 0.1, 2.5)

            self.counters.passes[match, own[is_pass]] += 1
            np.add.at(self.counters.pass_distance, (match, own[is_pass]), distance)
            self.in_flight[match] = True
            self.flight_start[match] = self.t[match]
            self.flight_end[match] = self.t[match] + flight
            self.flight_from[match] = carrier_pos[is_pass]
            self.flight_to[match] = target
            self.flight_side[match] = own[is_pass]
            self.flight_player[match] = receiver
            self.flight_ok[match] = success

        # ---- shots ----
        is_shot = choice == OPT_SHOT
        if is_shot.any():
            match = idx[is_shot]
            side = own[is_shot]
            xg = shot_xg[is_shot]
            self.counters.shots[match, side] += 1
            self.counters.xg[match, side] += xg          # raw, observable

            # Conversion is where the tactical matchup pays out. A favourable
            # matchup converts the same chance more often; the recorded xG is
            # untouched, so the belief filter's view stays clean.
            # Use a symmetric additive finishing edge.  A multiplicative edge
            # could not make a tactical counter matter when its chances were
            # low-xG (multiplying almost zero remains almost zero), so the raw
            # physical-strength ordering stayed transitive.  For reversed
            # tactics the edge changes sign, keeping the pairing centered while
            # allowing a genuine counter to flip a close matchup.
            signed_edge = through["edge"][is_shot]
            polarised = np.abs(signed_edge) > 1e-9
            # In a polarised matchup, allocate a smaller shared finishing budget
            # toward the side holding the tactical edge.  This creates a strong
            # counter without the one-sided clipping that previously inflated
            # tactic-space scoring above four goals per match.
            base_conversion = np.where(polarised, 0.60 * xg, xg)
            conversion_edge = 0.30 * self.matchup_gain * signed_edge
            convert = np.clip(0.78 * (base_conversion + conversion_edge), 1e-4, 0.95)
            scored = self.rng.random(match.size) < convert
            if scored.any():
                goal_match, goal_side = match[scored], side[scored]
                self.score[goal_match, goal_side] += 1
                self._goal_flags[goal_match, goal_side] += 1
                self._kickoff(goal_match, 1 - goal_side)
            missed = match[~scored]
            if missed.size:
                # Goal kick to the defending keeper.
                self.in_flight[missed] = False
                self.possessing[missed] = 1 - side[~scored]
                self.carrier[missed] = 0
                self.carrying[missed] = False
                self.ball[missed] = self.pos[missed, self.possessing[missed], 0]

        # ---- dribbles ----
        # A dribble is NOT a teleport. It flags the carrier as driving; the
        # physics layer advances him goal-ward over the following ticks, during
        # which he is exposed to tackles. Advancing him instantly (an earlier
        # version did) let players walk to the goal line and shoot from zero
        # metres, which is how 60-goal matches happen.
        self.carrying[idx] = choice == OPT_DRIBBLE

        # ---- through balls ----
        is_through = choice == OPT_THROUGH
        if is_through.any():
            sel = rows[is_through]
            match = idx[is_through]
            side = own[is_through]
            target = through["target"][is_through]

            distance = np.linalg.norm(target - carrier_pos[is_through], axis=-1)
            self.counters.passes[match, side] += 1
            # Advanced indexing plus a scalar sum gave every selected match the
            # distance travelled by the whole batch.  Besides being wrong, that
            # made the belief feature depend on batch size.
            np.add.at(self.counters.pass_distance, (match, side), distance)
            self.in_flight[match] = True
            self.flight_start[match] = self.t[match]
            self.flight_end[match] = self.t[match] + np.clip(
                distance / self.pass_speed + 0.15, 0.1, 2.5)
            self.flight_from[match] = carrier_pos[is_through]
            self.flight_to[match] = target
            self.flight_side[match] = side
            self.flight_player[match] = through["runner"][is_through]
            self.flight_ok[match] = self.rng.random(sel.size) < through["success"][sel]

        # ---- clearances ----
        is_clear = choice == OPT_CLEAR
        if is_clear.any():
            match = idx[is_clear]
            side = own[is_clear]
            direction = np.sign(ATTACK_X[side] - carrier_pos[is_clear, 0])
            target = np.stack([
                np.clip(carrier_pos[is_clear, 0] + direction * 40.0, 2.0, PITCH_L - 2.0),
                np.clip(carrier_pos[is_clear, 1]
                        + self.rng.normal(0, 12.0, match.size), 2.0, PITCH_W - 2.0),
            ], axis=-1)
            self.counters.passes[match, side] += 1
            np.add.at(self.counters.pass_distance, (match, side),
                      np.linalg.norm(target - carrier_pos[is_clear], axis=-1))
            self.in_flight[match] = True
            self.flight_start[match] = self.t[match]
            self.flight_end[match] = self.t[match] + 1.6
            self.flight_from[match] = carrier_pos[is_clear]
            self.flight_to[match] = target
            self.flight_side[match] = side
            self.flight_player[match] = self.rng.integers(1, 11, match.size)
            # A clearance is a lottery, not a pass: coin-flip retention.
            self.flight_ok[match] = self.rng.random(match.size) < 0.35

    def _kickoff(self, idx: np.ndarray, receiving_side: np.ndarray) -> None:
        self.in_flight[idx] = False
        self.ball[idx] = np.array([PITCH_L / 2, PITCH_W / 2], np.float32)
        self.possessing[idx] = receiving_side
        self.carrier[idx] = 9
        self.carrying[idx] = False
        self.pos[idx] = self._slots[idx]
        self.vel[idx] = 0.0
        self.pos[idx, receiving_side, 9] = self.ball[idx]

    # ------------------------------------------------------------------

    def _record(self, active: np.ndarray) -> None:
        if not self._recorders:
            return
        self._tick_counter += 1
        for index, builder in self._recorders.items():
            if self._tick_counter % builder.tick_stride:
                continue
            if not active[index]:
                continue
            builder.add_tick(self.t[index], self.pos[index, HOME],
                             self.pos[index, AWAY], self.ball[index])

    # ------------------------------------------------------------------

    def window_stats(self, consume: bool = True) -> np.ndarray:
        """Observable statistics since the last consumed window. Shape (B, 2, 8).

        `consume=True` advances the window mark and must happen exactly once per
        decision epoch. Pass `consume=False` to peek — inspecting the state after
        a coaching decision must not swallow the window the next epoch depends on.
        """
        now = self.counters.snapshot()
        delta = {k: now[k] - self._window_mark[k] for k in now}
        if consume:
            self._window_mark = now

        ticks = np.maximum(delta["possession_ticks"].sum(axis=1, keepdims=True), 1.0)
        possession = delta["possession_ticks"] / ticks

        touches = np.maximum(delta["final_third_touches"].sum(axis=1, keepdims=True), 1.0)
        tilt = delta["final_third_touches"] / touches

        # PPDA: opponent passes allowed per own defensive action. Low = pressing hard.
        opp_passes = delta["passes"][:, ::-1]
        ppda = np.clip(opp_passes / np.maximum(delta["defensive_actions"], 1.0), 0.5, 40.0)

        completion = np.divide(
            delta["passes_completed"], np.maximum(delta["passes"], 1.0),
            out=np.full_like(delta["passes"], 0.8), where=delta["passes"] > 0,
        )
        # Mean pass length: the clearest statistical fingerprint of a direct side,
        # and the observable that finally separates a low-block counter team from
        # a balanced mid-block one.
        pass_length = np.divide(
            delta["pass_distance"], np.maximum(delta["passes"], 1.0),
            out=np.full_like(delta["passes"], 18.0), where=delta["passes"] > 0,
        )
        # Tracking systems estimate line height from the block's resting line,
        # not from a single frame after defenders have chased the ball.  The
        # instantaneous back-four mean was dominated by those chases and made a
        # high possession block indistinguishable from a balanced mid block.
        block = self.tactics[:, :, 1]
        defensive_line = ctrl.modifier_tables()["line_x"][block] / PITCH_L

        return np.stack([
            possession,
            delta["shots"],
            defensive_line,
            delta["xg"],
            delta["xg"][:, ::-1],
            tilt,
            ppda,
            completion,
            pass_length,
        ], axis=-1)

    def minutes(self) -> np.ndarray:
        return self.t / 60.0

    def done(self) -> np.ndarray:
        # Tolerance is half a tick, not 1e-9: `t` is accumulated by repeated
        # addition over ~13,500 ticks, so it lands a few nanoseconds short of the
        # exact match length and a tighter bound leaves matches never finishing.
        return self.t >= self.match_seconds - 0.5 * self.dt


# --------------------------------------------------------------------------
# Single-match adapter implementing the frozen PitchSim interface
# --------------------------------------------------------------------------


class PitchSimEngine(PitchSim):
    """One match, behind the same interface as `StubPitchSim`.

    Used by the console, by tests, and anywhere a single match is wanted. Training
    should drive `BatchedEngine` directly — a batch of one throws away the entire
    point of the vectorisation.
    """

    IS_STUB = False

    def __init__(self) -> None:
        self.engine = BatchedEngine(batch=1)
        self._builder: ReplayBuilder | None = None
        self._state = MatchState()
        self._half_score = (0, 0)
        self._n_decisions = 0

    def reset(self, seed=None, home_tactic=None, away_tactic=None, record=False) -> MatchState:
        tactics = np.stack([
            (home_tactic or Tactic()).as_array(),
            (away_tactic or Tactic()).as_array(),
        ])[None, ...]
        self.engine.reset(seed=seed, tactics=tactics)
        self._seed = seed
        self._n_decisions = 0
        self._half_score = (0, 0)
        self._builder = ReplayBuilder(seed=seed, dt=self.engine.dt) if record else None
        if self._builder is not None:
            self.engine.attach_recorder(0, self._builder)
        self._sync()
        return self._state

    def set_tactic(self, side: Side, tactic: Tactic) -> None:
        tactics = self.engine.tactics.copy()
        tactics[0, int(side)] = tactic.as_array()

        if tactic.substitution != 0:
            used = self.engine.subs_used[0, int(side)]
            windows = self.engine.windows_used[0, int(side)]
            if used < 5 and windows < 3:
                slot = self.engine.rng.integers(1, 11)
                self.engine.stamina[0, int(side), slot] = 0.95
                self.engine.subs_used[0, int(side)] += 1
                self.engine.windows_used[0, int(side)] += 1

        self.engine.set_tactics(tactics)
        self._n_decisions += 1
        # Refresh the exposed state, so a caller that inspects `state` straight
        # after a coaching decision sees the substitution it just made rather
        # than the state as it was before. Peek only: consuming the window here
        # would swallow the statistics the next epoch is measured over.
        self._sync(consume=False)

    def advance(self, max_minutes: float) -> SegmentResult:
        if self._state.done:
            return SegmentResult(self._state, 0.0, [], "done")

        start = float(self.engine.minutes()[0])
        goals = self.engine.run_until((start + max_minutes) * 60.0)
        elapsed = float(self.engine.minutes()[0]) - start

        events = []
        trigger = "scheduled"
        if goals[0].any():
            for side in (HOME, AWAY):
                for _ in range(int(goals[0, side])):
                    events.append({"type": "goal", "side": side,
                                   "minute": float(self.engine.minutes()[0])})
                    if self._builder is not None:
                        self._builder.add_event(float(self.engine.minutes()[0]), "goal", side)
            trigger = "goal_for" if goals[0, HOME] else "goal_against"

        minute = float(self.engine.minutes()[0])
        if self._half_score == (0, 0) and minute >= 45.0:
            self._half_score = tuple(self.engine.score[0])
            if trigger == "scheduled":
                trigger = "half_time"

        self._sync()
        if self._builder is not None:
            self._builder.add_xg(minute, *self._state.cumulative_xg)
            self._builder.add_win_prob(minute, self._win_prob())
        return SegmentResult(self._state, elapsed, events, trigger)

    def _sync(self, consume: bool = True) -> None:
        stats = self.engine.window_stats(consume=consume)[0]
        counters = self.engine.counters
        self._state = MatchState(
            minute=float(self.engine.minutes()[0]),
            score=(int(self.engine.score[0, 0]), int(self.engine.score[0, 1])),
            squads=(
                SquadState(stamina=self.engine.stamina[0, HOME].copy(),
                           yellow_cards=int(self.engine.yellow_cards[0, HOME]),
                           red_cards=int(self.engine.red_cards[0, HOME]),
                           subs_used=int(self.engine.subs_used[0, HOME]),
                           windows_used=int(self.engine.windows_used[0, HOME])),
                SquadState(stamina=self.engine.stamina[0, AWAY].copy(),
                           yellow_cards=int(self.engine.yellow_cards[0, AWAY]),
                           red_cards=int(self.engine.red_cards[0, AWAY]),
                           subs_used=int(self.engine.subs_used[0, AWAY]),
                           windows_used=int(self.engine.windows_used[0, AWAY])),
            ),
            tactics=(Tactic.from_array(self.engine.tactics[0, HOME]),
                     Tactic.from_array(self.engine.tactics[0, AWAY])),
            stats=(WindowStats(*stats[HOME]), WindowStats(*stats[AWAY])),
            cumulative_xg=(float(counters.xg[0, HOME]), float(counters.xg[0, AWAY])),
            done=bool(self.engine.done()[0]),
        )

    def _win_prob(self) -> float:
        diff = self._state.score[0] - self._state.score[1]
        remaining = max(90.0 - self._state.minute, 0.0)
        k = 1.1 / np.sqrt(remaining / 90.0 + 0.12)
        return float(1.0 / (1.0 + np.exp(-k * (diff + 0.35 * self._state.xg_diff()))))

    @property
    def state(self) -> MatchState:
        return self._state

    def result(self) -> MatchResult:
        counters = self.engine.counters
        ticks = counters.possession_ticks[0]
        return MatchResult(
            score=self._state.score,
            xg=self._state.cumulative_xg,
            shots=(int(counters.shots[0, HOME]), int(counters.shots[0, AWAY])),
            possession_home=float(ticks[HOME] / max(ticks.sum(), 1.0)),
            n_decisions=self._n_decisions,
            seed=getattr(self, "_seed", None),
            home_trailing_at_half=self._half_score[0] < self._half_score[1],
        )

    def replay(self) -> dict:
        if self._builder is None:
            raise RuntimeError("reset(record=True) is required before replay()")
        self._builder.result = self.result().as_dict()
        return self._builder.as_dict()
