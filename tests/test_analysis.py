from __future__ import annotations

import json

import numpy as np

import analysis.aggregate_evaluation as agg_eval
from agents.league import PFSPLeague
from analysis.egta import alpha_rank, find_cycles, nash, payoff_matrix
from analysis.self_play_elo import generations_report
from envs.coach_env import VecCoachEnv
from envs.archetypes import ScriptedCoaches
from pitchsim import Tactic


def test_significant_cycle_and_game_theory_solutions():
    rps = np.array([[0.0, 1.0, -1.0],
                    [-1.0, 0.0, 1.0],
                    [1.0, -1.0, 0.0]])
    stderr = np.full_like(rps, 0.1)
    np.fill_diagonal(stderr, 0.0)
    assert len(find_cycles(rps, stderr)) == 1

    equilibrium = nash(rps)
    np.testing.assert_allclose(equilibrium["row"], np.full(3, 1 / 3), atol=1e-6)
    np.testing.assert_allclose(alpha_rank(rps), np.full(3, 1 / 3), atol=1e-6)


def test_generic_payoff_matrix_is_antisymmetric():
    strategies = [0.0, 1.0, 3.0]
    matrix = payoff_matrix(strategies, lambda a, b: a - b)
    np.testing.assert_allclose(matrix, -matrix.T)


def test_pfsp_league_roundtrip(tmp_path):
    manifest = tmp_path / "league.json"
    checkpoint = tmp_path / "policy.zip"
    checkpoint.touch()
    league = PFSPLeague(path=manifest)
    league.register(checkpoint)
    loaded = PFSPLeague.load(manifest)
    assert loaded.sample(np.random.default_rng(0)) == checkpoint.resolve()
    loaded.record(checkpoint, 1.0)
    assert PFSPLeague.load(manifest).entries[0].games == 1


def test_stage_zero_uses_fixed_balanced_opponent():
    env = VecCoachEnv(n_envs=2, belief_mode="blind", opponent="fixed", seed=0)
    env.reset()
    expected = Tactic.from_names().as_array()
    np.testing.assert_array_equal(env.engine.tactics[:, 1], np.tile(expected, (2, 1)))


def test_flat_gamma_ablation_is_not_a_no_op_after_loading():
    """A loaded checkpoint must not silently keep its parent's discounting.

    `load` restores `smdp` from the saved attributes, so without an explicit
    switch the flat-gamma ablation trains gamma^tau and saves a bit-identical
    policy under a tag claiming otherwise.
    """
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

    from agents.smdp_ppo import SMDPMaskableRolloutBuffer, SMDPMaskablePPO

    env = VecCoachEnv(n_envs=2, belief_mode="blind", opponent="fixed", seed=0)
    model = SMDPMaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8,
                            gamma_per_minute=0.995, smdp=True, seed=0, verbose=0)
    assert isinstance(model.rollout_buffer, SMDPMaskableRolloutBuffer)

    model.set_discounting(False)
    assert model.smdp is False
    assert not isinstance(model.rollout_buffer, SMDPMaskableRolloutBuffer)
    assert isinstance(model.rollout_buffer, MaskableRolloutBuffer)

    model.set_discounting(True)
    assert model.smdp is True
    assert isinstance(model.rollout_buffer, SMDPMaskableRolloutBuffer)


def test_generations_report_requires_rising_and_enough_generations(tmp_path):
    missing = tmp_path / "self_play_elo.json"
    assert generations_report(missing) == (False, f"no round-robin evidence; run: python -m analysis.self_play_elo")

    too_few = tmp_path / "two_gens.json"
    too_few.write_text(json.dumps({
        "tags": ["gen0", "gen1"],
        "ratings": {"gen0": 1500.0, "gen1": 1550.0},
    }), encoding="utf-8")
    ok, detail = generations_report(too_few)
    assert not ok and "need >= 3" in detail

    rising = tmp_path / "rising.json"
    rising.write_text(json.dumps({
        "tags": ["gen0", "gen1", "gen2"],
        "ratings": {"gen0": 1450.0, "gen1": 1500.0, "gen2": 1550.0},
    }), encoding="utf-8")
    assert generations_report(rising)[0]

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({
        "tags": ["gen0", "gen1", "gen2"],
        "ratings": {"gen0": 1500.0, "gen1": 1600.0, "gen2": 1400.0},
    }), encoding="utf-8")
    assert not generations_report(flat)[0]


def test_ppo_vs_random_report_checks_non_overlapping_cis(tmp_path, monkeypatch):
    monkeypatch.setattr(agg_eval, "ROOT", tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    tag = "stage1_bayesian_smdp_seed0"
    (data_dir / f"eval_{tag}.json").write_text(json.dumps({
        tag: {"points_per_match": 2.2, "ppm_ci": [1.9, 2.5]},
    }), encoding="utf-8")
    (data_dir / "eval_baselines.json").write_text(json.dumps({
        "runs": {"seed_0": {"random": {"points_per_match": 1.3, "ppm_ci": [1.1, 1.4]}}},
    }), encoding="utf-8")

    ok, detail = agg_eval.ppo_vs_random_report(tag)
    assert ok and "PPO ppm" in detail

    (data_dir / "eval_baselines.json").write_text(json.dumps({
        "runs": {"seed_0": {"random": {"points_per_match": 1.3, "ppm_ci": [1.1, 2.4]}}},
    }), encoding="utf-8")
    assert not agg_eval.ppo_vs_random_report(tag)[0]


def test_scripted_coach_marks_only_realised_tactic_changes():
    coaches = ScriptedCoaches(np.array([5]))  # balanced 4-4-2
    rng = np.random.default_rng(0)
    coaches.decide(np.array([50.0]), np.array([-1]), rng)
    assert coaches.regime_switched_now[0]
    first_epoch = coaches.last_switch_epoch[0]

    # The same trailing rule is still active, but the realised tactic is
    # unchanged and therefore must not be counted as another switch.
    coaches.decide(np.array([55.0]), np.array([-1]), rng)
    assert not coaches.regime_switched_now[0]
    assert coaches.last_switch_epoch[0] == first_epoch
