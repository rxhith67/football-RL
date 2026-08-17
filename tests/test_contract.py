"""Contract tests — these guard the interfaces that owners A, B and C share.

If a change to `pitchsim.types` or `pitchsim.replay` breaks one of these, it has
broken at least one teammate's workstream. Run before every push.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchsim import (
    HEAD_SIZES,
    MASK_LENGTH,
    REPLAY_SCHEMA_VERSION,
    Side,
    StubPitchSim,
    Tactic,
    WindowStats,
    validate_replay,
)


# --------------------------------------------------------------------------
# Tactic space
# --------------------------------------------------------------------------


def test_head_sizes_match_plan():
    assert HEAD_SIZES == (5, 3, 3, 3, 6)
    assert MASK_LENGTH == 20
    assert int(np.prod(HEAD_SIZES)) == 810


def test_tactic_roundtrip():
    t = Tactic.from_names("3-5-2", "high", "gegenpress", "direct", "bench_2")
    assert Tactic.from_array(t.as_array()) == t
    assert t.as_names()["formation"] == "3-5-2"


def test_tactic_rejects_out_of_range():
    with pytest.raises(ValueError):
        Tactic(formation=5)
    with pytest.raises(ValueError):
        Tactic(press=3)


def test_one_hot_is_14_dim_and_excludes_substitution():
    a = Tactic.from_names("4-3-3", "high", "gegenpress", "direct", "none")
    b = Tactic.from_names("4-3-3", "high", "gegenpress", "direct", "bench_3")
    assert a.one_hot().shape == (14,)
    assert a.one_hot().sum() == 4.0
    np.testing.assert_array_equal(a.one_hot(), b.one_hot())


def test_churn_indicator_ignores_substitution():
    a = Tactic.from_names("4-4-2", "mid", "balanced", "balanced", "none")
    b = Tactic.from_names("4-4-2", "mid", "balanced", "balanced", "bench_1")
    c = Tactic.from_names("4-4-2", "low", "balanced", "balanced", "none")
    assert not a.style_differs_from(b)
    assert a.style_differs_from(c)


# --------------------------------------------------------------------------
# Stamina convention — the bug the original draft shipped (design note 1, row 6)
# --------------------------------------------------------------------------


def test_stamina_is_freshness_not_tiredness():
    """1 = fresh. A fresh squad must incur NO fatigue penalty."""
    from pitchsim.types import SquadState

    fresh = SquadState(stamina=np.ones(11, dtype=np.float32))
    spent = SquadState(stamina=np.full(11, 0.10, dtype=np.float32))
    assert fresh.fatigue_penalty() == 0.0
    assert spent.fatigue_penalty() > 0.0
    assert fresh.n_below(0.35) == 0
    assert spent.n_below(0.35) == 11


def test_window_stats_shape():
    assert WindowStats().as_array().shape == (WindowStats.N_FEATURES,)
    assert len(WindowStats.feature_names()) == WindowStats.N_FEATURES


# --------------------------------------------------------------------------
# Sim interface
# --------------------------------------------------------------------------


def _play_match(sim, seed=0, home=None, away=None, record=False):
    sim.reset(seed=seed, home_tactic=home, away_tactic=away, record=record)
    n = 0
    while not sim.done and n < 50:
        sim.advance(5.0)
        n += 1
    return sim.result()


def test_stub_completes_a_match():
    sim = StubPitchSim()
    result = _play_match(sim, seed=1)
    assert sim.done
    assert sim.state.minute == pytest.approx(90.0)
    assert result.shots[0] >= 0 and result.shots[1] >= 0
    assert 0.0 < result.possession_home < 1.0


def test_stub_is_deterministic_given_seed():
    a = _play_match(StubPitchSim(), seed=7)
    b = _play_match(StubPitchSim(), seed=7)
    assert a.score == b.score
    assert a.xg == b.xg


def test_stub_is_flagged_as_a_stub():
    """No stub output may ever be reported as a result."""
    assert StubPitchSim.IS_STUB is True


def test_advance_returns_variable_dwell_time():
    """tau_k must be reported so the SMDP discount gamma^tau is well defined."""
    sim = StubPitchSim()
    sim.reset(seed=3)
    segment = sim.advance(5.0)
    assert segment.duration_minutes > 0
    assert segment.trigger in {
        "scheduled", "goal_for", "goal_against", "half_time", "red_card", "done"
    }


def test_pressing_lowers_ppda_and_drains_stamina():
    """Directional sanity: the stub must respond to tactics in the right direction."""
    ppda, stamina = {}, {}
    for press in ("contain", "gegenpress"):
        sim = StubPitchSim()
        sim.reset(seed=11, home_tactic=Tactic.from_names(press=press))
        for _ in range(9):
            sim.advance(5.0)
        ppda[press] = sim.state.stats[Side.HOME].ppda
        stamina[press] = float(np.mean(sim.state.squads[Side.HOME].stamina))

    assert ppda["gegenpress"] < ppda["contain"], "pressing must reduce PPDA"
    assert stamina["gegenpress"] < stamina["contain"], "pressing must cost stamina"


def test_patient_tempo_raises_possession():
    poss = {}
    for tempo in ("patient", "direct"):
        sim = StubPitchSim()
        sim.reset(seed=5, home_tactic=Tactic.from_names(tempo=tempo))
        for _ in range(18):
            sim.advance(5.0)
        poss[tempo] = sim.result().possession_home
    assert poss["patient"] > poss["direct"]


# --------------------------------------------------------------------------
# Replay schema — the interface layer's contract
# --------------------------------------------------------------------------


def test_replay_validates():
    sim = StubPitchSim()
    _play_match(sim, seed=2, record=True)
    replay = sim.replay()
    assert validate_replay(replay) == []
    assert replay["schema_version"] == REPLAY_SCHEMA_VERSION
    assert len(replay["ticks"]) > 0
    assert len(replay["traces"]["xg_home"]) > 0
    assert replay["result"] is not None


def test_replay_requires_record_flag():
    sim = StubPitchSim()
    _play_match(sim, seed=2, record=False)
    with pytest.raises(RuntimeError):
        sim.replay()


def test_validate_replay_catches_bad_schema():
    assert validate_replay({"schema_version": 999}) != []
