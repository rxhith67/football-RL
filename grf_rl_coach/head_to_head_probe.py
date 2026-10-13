"""Read-only runtime feasibility check for a Q-coached right side versus native left.

Run: bash run.sh head_to_head_probe.py
This does not train, change a checkpoint, or modify the engine.
"""
import json
from pathlib import Path

import numpy as np
from gfootball.env import observation_rotation

from coach_env import AI, GRFDepthCoachEnv, apply_depth, live_depth, raw


def main():
    env = GRFDepthCoachEnv()
    try:
        env.reset(seed=991337)
        assert abs(live_depth(env.env, 0) - 0.71) < 1e-3
        assert abs(live_depth(env.env, 1) - 0.71) < 1e-3
        original = env.env.unwrapped._env.observation()
        flipped = observation_rotation.flip_observation(original, env.env.unwrapped._config)
        view = raw([flipped])
        assert view["owner"] == (-1 if original["ball_owned_team"] == -1 else 1 - original["ball_owned_team"])
        assert view["score"] == original["score"][::-1]
        assert np.allclose(view["left"], -original["right_team"])
        assert np.allclose(view["right"], -original["left_team"])
        assert np.allclose(view["ball"][:2], -original["ball"][:2])
        apply_depth(env.env, 1, 0.0)
        assert abs(live_depth(env.env, 1)) < 1e-3
        assert abs(live_depth(env.env, 0) - 0.71) < 1e-3
        sent = {"calls": 0, "non_builtin_actions": 0}
        real_step = env.env.step
        def spy(actions):
            sent["calls"] += 1
            sent["non_builtin_actions"] += sum(a != AI for a in actions)
            assert len(actions) == 11
            return real_step(actions)
        env.env.step = spy
        old_l = np.asarray(original["left_team"], float)
        old_r = np.asarray(original["right_team"], float)
        old_b = np.asarray(original["ball"], float)
        for _ in range(100):
            obs, _, _, _ = env.env.step([AI] * 11)
        assert abs(live_depth(env.env, 1)) < 1e-3
        assert abs(live_depth(env.env, 0) - 0.71) < 1e-3
        apply_depth(env.env, 1, None)
        for _ in range(20):
            obs, _, _, _ = env.env.step([AI] * 11)
        assert abs(live_depth(env.env, 1) - 0.71) < 1e-3
        assert sent == {"calls": 120, "non_builtin_actions": 0}
        result = {"seed": 991337, "steps": 120, "action_list": "11 left builtin_ai; right has no external controls",
                  "engine_step_calls": sent["calls"], "non_builtin_actions_sent": sent["non_builtin_actions"],
                  "right_observation_matches_official_180_rotation": True,
                  "team_1_override_applied_without_changing_team_0": True,
                  "team_1_override_clears_to_native": True,
                  "left_player_motion_norm": float(np.linalg.norm(obs[0]["left_team"] - old_l)),
                  "right_player_motion_norm": float(np.linalg.norm(obs[0]["right_team"] - old_r)),
                  "ball_motion_norm": float(np.linalg.norm(obs[0]["ball"] - old_b)),
                  "live_depth_left_right": [float(live_depth(env.env, 0)), float(live_depth(env.env, 1))]}
        Path("results/head_to_head_probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
