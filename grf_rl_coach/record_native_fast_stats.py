"""Record a full 2.5x GRF-rendered Q-coach match with exact per-step stats.

Only coach-visible frames are recorded, one per engine step. GRF's built-in
video writer also includes unobserved transition frames, so a separate video
cannot safely be aligned frame-for-frame with the coach's observations.

Run: bash run.sh record_native_fast_stats.py --seed 800001 \
  --out demo/q100_seed800001_native_fast
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from coach_env import ACTION_NAMES, AI, GRFDepthCoachEnv, NATIVE_DEFAULT_DEPTH
import policies


YELLOW = (80, 220, 255)  # BGR
NAVY = (245, 175, 100)
WHITE = (235, 240, 240)
MUTED = (168, 185, 185)
BG = (25, 34, 36)
SOURCE_FPS = 10
OUTPUT_FPS = 25


def clock(steps):
    steps = int(steps)
    return "%02d:%02d.%d" % (steps // 600, (steps // 10) % 60, steps % 10)


def put(frame, value, xy, color=WHITE, scale=0.58, thickness=1):
    cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def render_hud(obs, counts, hold, action, step):
    native = cv2.cvtColor(np.asarray(obs["frame"]), cv2.COLOR_RGB2BGR)
    height, width = native.shape[:2]
    assert width == 800 and height == 450, (width, height)
    image = np.full((height + 190, width, 3), BG, dtype=np.uint8)
    image[:height] = native
    cv2.line(image, (0, height), (width, height), (108, 140, 140), 2)
    y = height
    score = obs["score"]
    put(image, "RL COACH: %s" % action.replace("native_default", "default").upper(),
        (16, y + 27), YELLOW, .62, 2)
    put(image, "SCORE  %d - %d" % (score[0], score[1]), (390, y + 27), WHITE, .62, 2)
    put(image, "2.5x", (720, y + 27), MUTED, .52)

    owner = int(obs["ball_owned_team"])
    if owner == 0:
        owner_label, color = "YELLOW #%d" % int(obs["ball_owned_player"]), YELLOW
    elif owner == 1:
        owner_label, color = "NAVY #%d" % int(obs["ball_owned_player"]), NAVY
    else:
        owner_label, color = "LOOSE BALL", MUTED
    put(image, "BALL NOW:", (16, y + 60), MUTED, .54)
    put(image, owner_label, (142, y + 60), color, .58, 2)
    put(image, "CURRENT HOLD  %s" % (clock(hold) if owner != -1 else "--"),
        (430, y + 60), WHITE, .52)

    owned = counts[0] + counts[1]
    yellow_share = 100 * counts[0] / owned if owned else 0
    navy_share = 100 * counts[1] / owned if owned else 0
    put(image, "BALL CONTROL (GRF ownership; share excludes loose):",
        (16, y + 88), MUTED, .48)
    put(image, "YELLOW / RL: %s  (%4.1f%%)" % (clock(counts[0]), yellow_share),
        (16, y + 116), YELLOW, .54, 2)
    put(image, "NAVY / AI: %s  (%4.1f%%)" % (clock(counts[1]), navy_share),
        (395, y + 116), NAVY, .54, 2)
    put(image, "LOOSE: %s" % clock(counts[2]), (16, y + 146), MUTED, .52)
    put(image, "STEP %d / 3001" % step, (575, y + 146), MUTED, .50)
    cv2.rectangle(image, (16, y + 163), (width - 16, y + 171), (65, 77, 78), -1)
    x = 16 + round((width - 32) * step / 3001)
    cv2.rectangle(image, (16, y + 163), (x, y + 171), YELLOW, -1)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=800001)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/q_it100_final.npz"))
    parser.add_argument("--out", type=Path, default=Path("demo/q100_seed800001_native_fast"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "native_fast_stats.avi"
    assert not output.exists(), "refusing to overwrite an existing recording"
    q = np.load(args.ckpt)["q"]
    env = GRFDepthCoachEnv(render_video=True)
    writer = None
    counts = [0, 0, 0]  # yellow, navy, loose; GRF ball_owned_team per step
    previous_owner = -1
    hold = 0
    frame_count = 0
    calls = 0
    non_builtin = 0
    current_action = 1
    steps = []
    decisions = []
    live_matches = []
    real_step = env.env.step

    def capture(actions):
        nonlocal writer, previous_owner, hold, frame_count, calls, non_builtin
        calls += 1
        non_builtin += sum(action != AI for action in actions)
        obs, reward, done, info = real_step(actions)
        r = obs[0]
        assert "frame" in r, "GRF native renderer did not return a frame"
        owner = int(r["ball_owned_team"])
        assert owner in (-1, 0, 1)
        counts[{0: 0, 1: 1, -1: 2}[owner]] += 1
        hold = hold + 1 if owner != -1 and owner == previous_owner else (1 if owner != -1 else 0)
        previous_owner = owner
        frame_count += 1
        image = render_hud(r, counts, hold, ACTION_NAMES[current_action], frame_count)
        if writer is None:
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"XVID"),
                                     OUTPUT_FPS, (image.shape[1], image.shape[0]))
            assert writer.isOpened()
        writer.write(image)
        if frame_count == 1500:
            assert cv2.imwrite(str(args.out / "native_fast_stats_frame_1500.png"), image)
        steps.append({"step": frame_count, "ball_owned_team": owner,
                      "ball_owned_player": int(r["ball_owned_player"]),
                      "score_left": int(r["score"][0]), "score_right": int(r["score"][1]),
                      "coach_action": ACTION_NAMES[current_action]})
        return obs, reward, done, info

    env.env.step = capture
    try:
        f = env.reset(seed=args.seed)
        done = False
        while not done:
            state = env.discretize(f)
            current_action = int(policies.q_action(q, state))
            f, reward, done, info = env.step(current_action)
            decisions.append({"decision": len(decisions) + 1, "action": ACTION_NAMES[current_action],
                              "end_step": frame_count, "score_end": info["score"],
                              "reward": float(reward), "live_depth": info["live_depth"]})
            expected = (NATIVE_DEFAULT_DEPTH if current_action == 1
                        else 0.0 if current_action == 0 else 1.0)
            live_matches.append(abs(info["live_depth"] - expected) < 1e-4)
        final_score = list(env.prev["score"])
    finally:
        if writer is not None:
            writer.release()
        env.close()

    assert frame_count == calls == len(steps) == 3001
    assert non_builtin == 0 and all(live_matches)
    assert steps[-1]["score_left"] == final_score[0]
    assert steps[-1]["score_right"] == final_score[1]
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3001
    assert abs(cap.get(cv2.CAP_PROP_FPS) - OUTPUT_FPS) < 0.01
    cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
    ok, last_frame = cap.read()
    assert ok
    assert cv2.imwrite(str(args.out / "native_fast_stats_last_frame.png"), last_frame)
    cap.release()
    with (args.out / "step_stats.csv").open("w", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(steps[0]))
        csv_writer.writeheader()
        csv_writer.writerows(steps)
    (args.out / "decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    owned = counts[0] + counts[1]
    summary = {
        "seed": args.seed, "checkpoint": str(args.ckpt),
        "checkpoint_sha256": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "source": "GRF native rendered observation at each coach-visible engine step",
        "frames": frame_count, "fps": OUTPUT_FPS, "playback_speed": OUTPUT_FPS / SOURCE_FPS,
        "final_score_left_right": final_score,
        "non_builtin_ai_player_actions": non_builtin,
        "coach_depth_applied_at_all_decisions": all(live_matches),
        "action_counts": {name: sum(d["action"] == name for d in decisions) for name in ACTION_NAMES},
        "ball_owned_steps_yellow_navy_loose": counts,
        "ball_owned_seconds_yellow_navy_loose": [round(n / 10, 1) for n in counts],
        "owned_time_share_yellow_navy": [counts[0] / owned, counts[1] / owned],
        "measurement": "GRF ball_owned_team; loose excluded from share",
        "note": "Separate demonstration run; not an aggregate evaluation or proof of RL advantage",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
