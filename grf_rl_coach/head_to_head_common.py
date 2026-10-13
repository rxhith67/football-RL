"""Frozen Q coach versus untouched native AI, with the Q team on either side.

The left team's eleven external action slots always receive action_builtin_ai.
The right team has zero external action slots. Only the Q team's native tactical
depth setting is changed; the opposing team's depth override is cleared at reset.
"""
from pathlib import Path

import numpy as np
from gfootball.env import observation_rotation

from coach_env import (AI, ACTION_DEPTH, ACTION_NAMES, GRFDepthCoachEnv,
                       apply_depth, live_depth, raw)
from policies import q_action


CHECKPOINT = Path("checkpoints/q_it100_final.npz")
NATIVE_DEPTH = 0.71
AI_ACTIONS = [AI] * 11


def own_view(env, coached_side, agent_obs=None):
    """Use the existing training feature frame: our team attacks left to right.

    GRF's official flip_observation performs a 180-degree rotation and swaps
    team, score, and possession labels for right-side play.
    """
    if coached_side == 0:
        if agent_obs is None:
            agent_obs = env.env.unwrapped.observation()
        return raw(agent_obs)
    core = env.env.unwrapped._env.observation()
    rotated = observation_rotation.flip_observation(core, env.env.unwrapped._config)
    return raw([rotated])


def run_match(env, coached_side, seed, q):
    assert coached_side in (0, 1)
    assert q.shape == (243, 3)
    env.noop_calls = False
    env.reset(seed=int(seed))  # also clears both teams' inherited overrides
    assert abs(live_depth(env.env, 0) - NATIVE_DEPTH) < 1e-3
    assert abs(live_depth(env.env, 1) - NATIVE_DEPTH) < 1e-3
    env.last_action = 1
    view = own_view(env, coached_side)
    features = env._features(view, 0.0)
    assert features.shape == (14,)
    decisions = []
    total_steps = 0
    done = False
    opponent = 1 - coached_side
    expected = {0: 0.0, 1: NATIVE_DEPTH, 2: 1.0}

    while not done:
        state_id = int(env.discretize(features))
        action = int(q_action(q, state_id, tie_default=True))
        assert action in (0, 1, 2)
        # Never write to the opponent's team controller after reset.
        apply_depth(env.env, coached_side, ACTION_DEPTH[action])
        env.last_action = action
        start = total_steps
        own_possession = 0
        n = 0
        score_start = list(view["score"])
        for _ in range(100):
            obs, _, done, _ = env.env.step(AI_ACTIONS)
            n += 1
            global_state = raw(obs)
            own_possession += int(global_state["owner"] == coached_side)
            if done:
                break
        # Match the existing training environment's 100-step decisions and
        # one-step terminal stub, giving 30 decisions over 3,001 engine steps.
        while not done and global_state["steps_left"] <= 1 and n < 105:
            obs, _, done, _ = env.env.step(AI_ACTIONS)
            n += 1
            global_state = raw(obs)
            own_possession += int(global_state["owner"] == coached_side)
        total_steps += n
        assert abs(live_depth(env.env, coached_side) - expected[action]) < 1e-3
        assert abs(live_depth(env.env, opponent) - NATIVE_DEPTH) < 1e-3
        decisions.append({"decision": len(decisions) + 1, "engine_step_start": start,
                          "engine_steps": n, "state_id": state_id,
                          "observation": [float(x) for x in features],
                          "action": action, "action_name": ACTION_NAMES[action],
                          "live_coached_depth_end": float(live_depth(env.env, coached_side)),
                          "live_native_depth_end": float(live_depth(env.env, opponent)),
                          "score_own_opp_start": score_start,
                          "score_left_right_end": list(global_state["score"])})
        view = own_view(env, coached_side, obs)
        features = env._features(view, own_possession / n)
        assert features.shape == (14,) and np.isfinite(features).all()

    assert total_steps == 3001 and len(decisions) == 30
    score = list(global_state["score"])
    own_goals, opp_goals = (score[0], score[1]) if coached_side == 0 else (score[1], score[0])
    counts = [sum(d["action"] == a for d in decisions) for a in range(3)]
    return {"seed": int(seed), "coached_side": "left" if coached_side == 0 else "right",
            "score_left_right": score, "rl_goals": int(own_goals), "native_goals": int(opp_goals),
            "rl_goal_difference": int(own_goals - opp_goals),
            "result": "win" if own_goals > opp_goals else ("loss" if own_goals < opp_goals else "draw"),
            "engine_steps": total_steps, "action_counts_compact_default_stretched": counts,
            "decisions": decisions, "all_left_actions_builtin_ai": True,
            "right_external_action_slots": 0, "opponent_tactical_setter_calls_after_reset": 0}
