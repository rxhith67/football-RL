"""Small, isolated check that GRF supplies a genuine rendered frame in WSL."""
import json
from pathlib import Path

import cv2
import numpy as np
import gfootball.env as football_env
from gfootball.env import football_action_set


out = Path(__file__).resolve().parent / "native_render_probe"
out.mkdir(exist_ok=True)
env = football_env.create_environment(
    env_name="11_vs_11_hard_stochastic",
    representation="raw",
    rewards="scoring",
    number_of_left_players_agent_controls=1,
    render=True,
    write_video=False,
    other_config_options={"action_set": "v2", "display_game_stats": False},
)
try:
    obs = env.reset()
    ai = football_action_set.action_set_v2.index(football_action_set.action_builtin_ai)
    for _ in range(10):
        obs, _, _, _ = env.step([ai])
    frame = obs[0].get("frame")
    result = {
        "frame_present": frame is not None,
        "shape": list(np.shape(frame)) if frame is not None else None,
        "dtype": str(np.asarray(frame).dtype) if frame is not None else None,
    }
    if frame is not None:
        rgb = np.asarray(frame)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        result["unique_color_count"] = int(len(np.unique(rgb.reshape(-1, 3), axis=0)))
        result["frame_file"] = str(out / "native_frame.png")
        assert cv2.imwrite(str(out / "native_frame.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
finally:
    env.close()
