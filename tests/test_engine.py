"""Engine invariants — physics and football rules that must hold at any calibration.

These are deliberately separate from the calibration bands. Calibration is a
tuning question and its numbers move; the properties below are structural, and if
one breaks the engine is wrong rather than merely mistuned.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchsim import Side, Tactic, validate_replay
from pitchsim.engine import PITCH_L, PITCH_W, BatchedEngine, PitchSimEngine
from pitchsim.pitch_control import defenders_in_cone, shot_geometry


@pytest.fixture(scope="module")
def played():
    engine = BatchedEngine(batch=6)
    engine.reset(seed=0)
    engine.run_until(engine.match_seconds, stop_on_goal=False)
    return engine


# --------------------------------------------------------------------------
# Physics invariants
# --------------------------------------------------------------------------


def test_players_stay_on_the_pitch(played):
    assert played.pos[..., 0].min() >= -0.01
    assert played.pos[..., 0].max() <= PITCH_L + 0.01
    assert played.pos[..., 1].min() >= -0.01
    assert played.pos[..., 1].max() <= PITCH_W + 0.01


def test_ball_stays_on_the_pitch(played):
    assert 0.0 <= played.ball[:, 0].min() and played.ball[:, 0].max() <= PITCH_L
    assert 0.0 <= played.ball[:, 1].min() and played.ball[:, 1].max() <= PITCH_W


def test_stamina_stays_in_range_and_decreases(played):
    assert played.stamina.min() >= 0.0
    assert played.stamina.max() <= 1.0
    # Everyone starts fresh, so a full match must have cost something.
    assert played.stamina.mean() < 0.95


def test_match_runs_to_full_time(played):
    assert np.allclose(played.t, played.match_seconds)
    assert played.done().all()


def test_goal_trigger_stops_only_the_match_that_scored(monkeypatch):
    engine = BatchedEngine(batch=2)
    engine.reset(seed=0)

    def fake_tick(active):
        engine.t += engine.dt * active
        if active[0] and engine.t[0] >= 2.0:
            engine._goal_flags[0, 0] = 1

    monkeypatch.setattr(engine, "_tick", fake_tick)
    engine.run_until(np.array([10.0, 10.0]), stop_on_goal=True)

    assert engine.t[0] == pytest.approx(2.0)
    assert engine.t[1] == pytest.approx(10.0)


def test_exactly_one_side_holds_the_ball(played):
    assert set(np.unique(played.possessing)).issubset({0, 1})
    assert played.carrier.min() >= 0 and played.carrier.max() <= 10


def test_seeding_is_reproducible():
    def run(seed):
        e = BatchedEngine(batch=4)
        e.reset(seed=seed)
        e.run_until(30 * 60, stop_on_goal=False)
        return e.score.copy(), e.counters.xg.copy()

    a_score, a_xg = run(11)
    b_score, b_xg = run(11)
    np.testing.assert_array_equal(a_score, b_score)
    np.testing.assert_allclose(a_xg, b_xg)


# --------------------------------------------------------------------------
# Football rules
# --------------------------------------------------------------------------


def test_shot_geometry_is_sane():
    """Closer and more central must mean shorter distance and a wider angle."""
    close = np.array([[99.0, 34.0]])
    far = np.array([[60.0, 34.0]])
    wide = np.array([[99.0, 5.0]])

    d_close, a_close = shot_geometry(close, goal_x=PITCH_L)
    d_far, a_far = shot_geometry(far, goal_x=PITCH_L)
    d_wide, a_wide = shot_geometry(wide, goal_x=PITCH_L)

    assert d_close < d_far
    assert a_close > a_far, "a closer central shot must subtend a wider goal angle"
    assert a_wide < a_close, "a tight angle must subtend less of the goal"


def test_defenders_in_cone_counts_only_blockers():
    shooter = np.array([[90.0, 34.0]])
    # One directly in the way, one behind the shooter, one out wide.
    opponents = np.array([[[97.0, 34.0], [80.0, 34.0], [95.0, 5.0]]])
    assert defenders_in_cone(shooter, opponents, goal_x=PITCH_L)[0] == 1


def test_offside_is_enforced():
    """No pass may be completed to a player beyond the second-last defender."""
    engine = BatchedEngine(batch=8)
    engine.reset(seed=5)
    engine.run_until(20 * 60, stop_on_goal=False)

    # Reconstruct the offside line for the side in possession and check that the
    # intended receiver of any in-flight pass is legal.
    flying = np.flatnonzero(engine.in_flight)
    for i in flying:
        side = engine.flight_side[i]
        sign = 1.0 if side == 0 else -1.0
        opp = engine.pos[i, 1 - side, :, 0]
        line = np.sort(sign * opp)[-2]
        receiver_x = sign * engine.pos[i, side, engine.flight_player[i], 0]
        limit = max(line, sign * engine.flight_from[i][0])
        assert receiver_x <= limit + 2.0, "pass sent to an offside receiver"


# --------------------------------------------------------------------------
# Tactical directionality — the coach's levers must actually do something
# --------------------------------------------------------------------------


def _run_with(home: Tactic, away: Tactic, seed: int = 0, batch: int = 8):
    engine = BatchedEngine(batch=batch)
    tactics = np.tile(np.stack([home.as_array(), away.as_array()]), (batch, 1, 1))
    engine.reset(seed=seed, tactics=tactics)
    engine.run_until(engine.match_seconds, stop_on_goal=False)
    return engine


def test_pressing_lowers_ppda_and_costs_stamina():
    high = _run_with(Tactic.from_names(press="gegenpress"), Tactic.from_names())
    low = _run_with(Tactic.from_names(press="contain"), Tactic.from_names())

    assert (high.counters.defensive_actions[:, 0].mean()
            > low.counters.defensive_actions[:, 0].mean()), "pressing must win the ball more"
    assert high.stamina[:, 0].mean() < low.stamina[:, 0].mean(), "pressing must cost stamina"


def test_patient_tempo_increases_possession():
    patient = _run_with(Tactic.from_names(tempo="patient"), Tactic.from_names(tempo="direct"))
    ticks = patient.counters.possession_ticks
    share = ticks[:, 0] / np.maximum(ticks.sum(axis=1), 1.0)
    assert share.mean() > 0.5, "a patient side should hold the ball more than a direct one"


def test_high_block_pushes_the_defensive_line_up():
    high = _run_with(Tactic.from_names(block_height="high"), Tactic.from_names())
    low = _run_with(Tactic.from_names(block_height="low"), Tactic.from_names())
    # Home attacks +x, so a higher line means a larger mean x for the back four.
    assert high.pos[:, 0, 1:5, 0].mean() > low.pos[:, 0, 1:5, 0].mean()


def test_engine_is_not_flagged_as_a_stub():
    assert PitchSimEngine.IS_STUB is False


def test_single_match_adapter_matches_the_interface():
    sim = PitchSimEngine()
    sim.reset(seed=2, record=True)
    steps = 0
    while not sim.done and steps < 60:
        segment = sim.advance(5.0)
        assert segment.duration_minutes >= 0.0
        steps += 1

    assert sim.done
    assert validate_replay(sim.replay()) == []
    result = sim.result()
    assert result.shots[0] >= 0 and result.shots[1] >= 0
    assert 0.0 <= result.possession_home <= 1.0


def test_substitution_restores_stamina():
    sim = PitchSimEngine()
    sim.reset(seed=1)
    for _ in range(12):
        sim.advance(5.0)

    before = sim.state.squads[Side.HOME].stamina.min()
    sim.set_tactic(Side.HOME, Tactic.from_names(substitution="bench_1"))
    after = sim.state.squads[Side.HOME]
    assert after.subs_used == 1
    assert after.stamina.max() >= 0.94
    assert after.stamina.max() > before
