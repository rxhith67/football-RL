"""StubPitchSim — a placeholder engine with the real interface.

WHAT THIS IS: a cheap generative model that emits statistically plausible match
data through the exact `PitchSim` API. It exists so that owners B and C can build
the Gymnasium wrapper, the belief filter, the console and the evaluation harness
during Phase 0, instead of idling until the real engine lands. design note 6.1.

WHAT THIS IS NOT: football. There are no players, no ball and no physics — the
`ticks` it records are decorative circles. Nothing produced by this module may
appear in the write-up as a result. `StubPitchSim.IS_STUB` is True precisely so
that downstream code and figures can assert against it.

It is nonetheless deliberately *directionally* correct: possession responds to
tempo, stamina drains with press, and the archetype counter-cycle of design note
3.5 is baked into the goal-rate model. That is enough for B to confirm the RL
plumbing learns *something* and for C to confirm the EGTA harness finds a cycle,
so that when the real engine arrives the only open question is whether the
physics behaves — not whether the surrounding code works.
"""

from __future__ import annotations

import numpy as np

from .base import PitchSim
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

# Style axes derived from the tactic heads: (territory, intensity, directness).
# Territory ~ how high the team plays; intensity ~ pressing energy; directness ~
# vertical risk. These three drive every observable the stub emits.
_BLOCK_TERRITORY = np.array([-1.0, 0.0, 1.0])      # low / mid / high
_PRESS_INTENSITY = np.array([-1.0, 0.0, 1.0])      # contain / balanced / gegenpress
_TEMPO_DIRECTNESS = np.array([-1.0, 0.0, 1.0])     # patient / balanced / direct
_FORMATION_ATTACK = np.array([0.55, 0.35, 0.20, -0.45, 0.0])   # 433 4231 352 541 442
_FORMATION_SOLIDITY = np.array([-0.30, -0.05, 0.05, 0.55, 0.15])


class StubPitchSim(PitchSim):
    """Placeholder implementation of the PITCHSIM contract."""

    IS_STUB = True

    def __init__(self, dt: float = 0.2, match_minutes: float = 90.0) -> None:
        self.dt = dt
        self.match_minutes = match_minutes
        self._rng = np.random.default_rng()
        self._state = MatchState()
        self._record = False
        self._builder: ReplayBuilder | None = None
        self._shots = [0, 0]
        self._possession_accum = [0.0, 0.0]
        self._n_decisions = 0
        self._half_time_fired = False
        self._score_at_half = (0, 0)

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _style(tactic: Tactic) -> np.ndarray:
        """(territory, intensity, directness, attack, solidity) in roughly [-1, 1]."""
        return np.array(
            [
                _BLOCK_TERRITORY[tactic.block_height],
                _PRESS_INTENSITY[tactic.press],
                _TEMPO_DIRECTNESS[tactic.tempo],
                _FORMATION_ATTACK[tactic.formation],
                _FORMATION_SOLIDITY[tactic.formation],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _counter_edge(own: np.ndarray, opp: np.ndarray) -> float:
        """Rock-paper-scissors advantage, matching design note 3.5's cycle.

        Positive means `own` has the tactical upper hand this window:
          * pressing high beats a patient, short-passing side
          * playing direct beats a high line
          * patient possession beats a deep block
          * sitting deep absorbs a direct side
        """
        press_vs_patient = 0.30 * own[1] * (-opp[2])
        direct_vs_highline = 0.30 * own[2] * opp[0]
        patient_vs_lowblock = 0.22 * (-own[2]) * (-opp[0])
        deep_vs_direct = 0.22 * (-own[0]) * opp[2]
        return press_vs_patient + direct_vs_highline + patient_vs_lowblock + deep_vs_direct

    # ------------------------------------------------------------------
    # PitchSim interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        home_tactic: Tactic | None = None,
        away_tactic: Tactic | None = None,
        record: bool = False,
    ) -> MatchState:
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._state = MatchState(
            minute=0.0,
            score=(0, 0),
            squads=(SquadState(stamina=np.ones(11, dtype=np.float32)),
                    SquadState(stamina=np.ones(11, dtype=np.float32))),
            tactics=(home_tactic or Tactic(), away_tactic or Tactic()),
            stats=(WindowStats(), WindowStats()),
            cumulative_xg=(0.0, 0.0),
            done=False,
        )
        self._shots = [0, 0]
        self._possession_accum = [0.0, 0.0]
        self._n_decisions = 0
        self._half_time_fired = False
        self._score_at_half = (0, 0)
        self._record = record
        self._builder = ReplayBuilder(seed=seed, dt=self.dt) if record else None
        return self._state

    def set_tactic(self, side: Side, tactic: Tactic) -> None:
        tactics = list(self._state.tactics)
        squads = list(self._state.squads)

        if tactic.substitution != 0:
            squad = squads[side]
            if squad.subs_used < 5 and squad.windows_used < 3:
                slot = self._rng.integers(1, 11)   # never the keeper
                squad.stamina[slot] = 0.95
                squad.subs_used += 1
                squad.windows_used += 1
                if self._builder is not None:
                    self._builder.add_event(
                        self._state.minute, "substitution", int(side), slot=int(slot)
                    )

        tactics[side] = tactic
        self._state.tactics = (tactics[0], tactics[1])
        self._state.squads = (squads[0], squads[1])
        self._n_decisions += 1

    def advance(self, max_minutes: float) -> SegmentResult:
        if self._state.done:
            return SegmentResult(self._state, 0.0, [], "done")

        start = self._state.minute
        remaining = self.match_minutes - start
        span = float(min(max_minutes, remaining))
        events: list[dict] = []
        trigger = "scheduled"

        home_style = self._style(self._state.tactics[Side.HOME])
        away_style = self._style(self._state.tactics[Side.AWAY])
        edge_home = self._counter_edge(home_style, away_style)
        edge_away = self._counter_edge(away_style, home_style)

        # --- possession: patient + high-territory sides hold the ball ---
        poss_pressure = (
            0.55 * (-home_style[2] + away_style[2])
            + 0.25 * (home_style[0] - away_style[0])
            + 0.20 * (home_style[1] - away_style[1])
        )
        poss_home = float(np.clip(0.5 + 0.11 * poss_pressure, 0.22, 0.78))

        # --- shot volume and quality ---
        stamina = [float(np.mean(s.stamina)) for s in self._state.squads]
        rate_scale = span / 90.0
        shots = [0, 0]
        xg_added = [0.0, 0.0]
        for side, (style, opp_style, edge, poss) in enumerate(
            [(home_style, away_style, edge_home, poss_home),
             (away_style, home_style, edge_away, 1.0 - poss_home)]
        ):
            attack = 0.35 * style[3] + 0.25 * style[0] + 0.20 * style[2]
            defend = 0.40 * opp_style[4] - 0.20 * opp_style[0]
            lam = 12.0 * rate_scale * np.exp(
                0.30 * attack - 0.25 * defend + 0.55 * edge
                + 0.9 * (poss - 0.5) + 0.25 * (stamina[side] - 0.75)
            )
            n = int(self._rng.poisson(max(lam, 0.02)))
            shots[side] = n
            if n:
                quality = 0.10 * np.exp(0.30 * edge + 0.20 * style[3] - 0.20 * opp_style[4])
                xg_added[side] = float(
                    np.sum(self._rng.beta(1.6, 1.6 / max(quality, 1e-3) - 1.6, size=n))
                )

        # Each shot converts independently with probability equal to its mean xG.
        goals = [int(np.sum(self._rng.random(s) < (xg_added[i] / max(s, 1))))
                 if s else 0 for i, s in enumerate(shots)]

        # --- stamina drain, scaled by press ---
        for side in (Side.HOME, Side.AWAY):
            style = home_style if side == Side.HOME else away_style
            drain = (0.055 + 0.030 * style[1]) * (span / 15.0)
            squad = self._state.squads[side]
            noise = self._rng.normal(1.0, 0.12, size=11).clip(0.5, 1.5)
            squad.stamina = np.clip(squad.stamina - drain * noise, 0.05, 1.0).astype(np.float32)

        # --- commit ---
        self._state.minute = start + span
        self._state.score = (self._state.score[0] + goals[0], self._state.score[1] + goals[1])
        self._state.cumulative_xg = (
            self._state.cumulative_xg[0] + xg_added[0],
            self._state.cumulative_xg[1] + xg_added[1],
        )
        self._shots[0] += shots[0]
        self._shots[1] += shots[1]
        self._possession_accum[0] += poss_home * span
        self._possession_accum[1] += (1.0 - poss_home) * span

        self._state.stats = (
            self._window_stats(Side.HOME, poss_home, shots, xg_added, home_style, away_style),
            self._window_stats(Side.AWAY, 1 - poss_home, shots[::-1], xg_added[::-1],
                               away_style, home_style),
        )

        for side in (0, 1):
            for _ in range(goals[side]):
                events.append({"type": "goal", "side": side, "minute": self._state.minute})
                if self._builder is not None:
                    self._builder.add_event(self._state.minute, "goal", side)

        if goals[0] or goals[1]:
            trigger = "goal_for" if goals[0] else "goal_against"

        if not self._half_time_fired and self._state.minute >= 45.0:
            self._half_time_fired = True
            self._score_at_half = self._state.score
            if trigger == "scheduled":
                trigger = "half_time"

        if self._state.minute >= self.match_minutes - 1e-9:
            self._state.done = True

        if self._builder is not None:
            self._builder.add_xg(self._state.minute, *self._state.cumulative_xg)
            self._builder.add_win_prob(self._state.minute, self._win_prob_home())
            self._record_decorative_ticks(start, span)

        return SegmentResult(self._state, span, events, trigger)

    # ------------------------------------------------------------------

    def _window_stats(
        self, side: Side, poss: float, shots, xg, style, opp_style
    ) -> WindowStats:
        """Observable footprint. Signal is intentionally strong enough that the
        archetypes are separable, with noise so the filter is not trivial."""
        n = self._rng.normal
        return WindowStats(
            possession=float(np.clip(poss + n(0, 0.03), 0.05, 0.95)),
            shots_for=float(shots[0]),
            shots_against=float(shots[1]),
            xg_for=float(xg[0]),
            xg_against=float(xg[1]),
            field_tilt=float(np.clip(0.5 + 0.16 * (style[0] + style[3]) + n(0, 0.05), 0.02, 0.98)),
            # PPDA falls as pressing rises — the classic press signature.
            ppda=float(np.clip(12.0 * np.exp(-0.45 * style[1] - 0.15 * style[0]) + n(0, 1.1), 2.0, 40.0)),
            pass_completion=float(np.clip(0.80 - 0.06 * style[2] - 0.03 * opp_style[1] + n(0, 0.02), 0.4, 0.98)),
        )

    def _win_prob_home(self) -> float:
        diff = self._state.score[0] - self._state.score[1]
        remaining = max(self.match_minutes - self._state.minute, 0.0)
        k = 1.1 / np.sqrt(remaining / 90.0 + 0.12)
        return float(1.0 / (1.0 + np.exp(-k * (diff + 0.35 * self._state.xg_diff()))))

    def _record_decorative_ticks(self, start: float, span: float) -> None:
        """Placeholder positions so the console renderer can be developed.

        These are Lissajous wobbles around formation slots, NOT simulated motion.
        """
        assert self._builder is not None
        n_ticks = max(int(span * 60.0 / (self.dt * self._builder.tick_stride)), 1)
        for i in range(min(n_ticks, 60)):
            t = (start + span * i / max(n_ticks, 1)) * 60.0
            phase = t / 20.0
            home = np.stack([
                26.0 + 18.0 * np.sin(phase + np.arange(11) * 0.5),
                34.0 + 22.0 * np.cos(phase * 0.7 + np.arange(11) * 0.9),
            ], axis=1)
            away = np.stack([
                79.0 - 18.0 * np.sin(phase * 0.9 + np.arange(11) * 0.6),
                34.0 + 22.0 * np.sin(phase * 0.6 + np.arange(11) * 0.7),
            ], axis=1)
            ball = np.array([52.5 + 25.0 * np.sin(phase * 1.3), 34.0 + 15.0 * np.cos(phase)])
            self._builder.add_tick(t, home, away, ball)

    # ------------------------------------------------------------------

    @property
    def state(self) -> MatchState:
        return self._state

    def result(self) -> MatchResult:
        total = max(self._possession_accum[0] + self._possession_accum[1], 1e-9)
        return MatchResult(
            score=self._state.score,
            xg=self._state.cumulative_xg,
            shots=(self._shots[0], self._shots[1]),
            possession_home=self._possession_accum[0] / total,
            n_decisions=self._n_decisions,
            seed=getattr(self, "_seed", None),
            home_trailing_at_half=self._score_at_half[0] < self._score_at_half[1],
        )

    def replay(self) -> dict:
        if self._builder is None:
            raise RuntimeError("reset(record=True) is required before replay()")
        self._builder.result = self.result().as_dict()
        return self._builder.as_dict()
