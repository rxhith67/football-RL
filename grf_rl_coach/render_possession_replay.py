"""Add GRF ball-ownership and time-with-ball HUD to the saved 1-0 replay.

Counts come from the original GRF trace, not an inferred proximity heuristic.
One GRF engine step is 0.1 s of simulated match time. Loose-ball time is
reported separately; the percentage uses only steps with a team owner.

Run: bash run.sh render_possession_replay.py
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from render_clean_replay import pitch_background, render_frame, text


SIZE = (960, 680)
WHITE = (235, 244, 235)
BLUE = (230, 145, 45)
RED = (65, 80, 220)


def clock(ticks):
    tenths = int(ticks) % 10
    seconds = int(ticks) // 10
    return "%02d:%02d.%d" % (seconds // 60, seconds % 60, tenths)


def add_possession_hud(frame, obs, counts, streak):
    image = np.full((SIZE[1], SIZE[0], 3), (28, 37, 32), dtype=np.uint8)
    image[:frame.shape[0]] = frame
    cv2.rectangle(image, (0, 580), (SIZE[0], SIZE[1] - 1), (28, 37, 32), -1)
    cv2.line(image, (0, 580), (SIZE[0], 580), (185, 208, 185), 1)
    owner = int(obs["ball_owned_team"])
    player = int(obs["ball_owned_player"])
    if owner == 0:
        status, color = "BLUE - player %d" % player, BLUE
    elif owner == 1:
        status, color = "RED - player %d" % player, RED
    else:
        status, color = "LOOSE BALL", (204, 210, 205)
    text(image, "BALL NOW:", (22, 607), 0.55, WHITE, 1)
    text(image, status, (145, 607), 0.58, color, 2)
    text(image, "CURRENT HOLD: %s" % (clock(streak) if owner != -1 else "--"),
         (560, 607), 0.50, WHITE, 1)
    blue, red, loose = counts
    owned = blue + red
    blue_share = 100 * blue / owned if owned else 0.0
    red_share = 100 * red / owned if owned else 0.0
    text(image, "TIME WITH BALL:", (22, 635), 0.52, WHITE, 1)
    text(image, "BLUE %s (%.1f%%)" % (clock(blue), blue_share),
         (200, 635), 0.52, BLUE, 1)
    text(image, "RED %s (%.1f%%)" % (clock(red), red_share),
         (482, 635), 0.52, RED, 1)
    text(image, "LOOSE %s" % clock(loose), (750, 635), 0.48, WHITE, 1)
    text(image, "Share excludes loose ball. Clock is GRF match time (0.1 s/step).  Playback 2.5x.",
         (22, 663), 0.42, (195, 210, 195), 1)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=Path("demo/q100_seed800001"))
    args = ap.parse_args()
    folder = args.folder
    dumps = list(folder.glob("episode_done_*.dump"))
    assert len(dumps) == 1
    dump = dumps[0]
    timeline_file = folder / "timeline.json"
    timeline = json.loads(timeline_file.read_text())
    decisions = timeline["decisions"]
    assert len(decisions) == 30 and timeline["verification"]["engine_steps_total"] == 3001
    output = folder / "clean_pitch_possession.avi"
    assert not output.exists(), "refusing to overwrite an existing possession replay"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"XVID"), 25, SIZE)
    assert writer.isOpened()
    base = pitch_background()
    counts = [0, 0, 0]  # blue, red, loose; exact ball_owned_team frames
    previous_owner = -1
    streak = 0
    decision_index = 0
    count = 0
    last_score = None
    try:
        with dump.open("rb") as stream:
            while True:
                try:
                    trace = pickle.load(stream)
                except EOFError:
                    break
                while (decision_index + 1 < len(decisions) and
                       count >= decisions[decision_index + 1]["engine_step_start"]):
                    decision_index += 1
                obs = trace["observation"]
                owner = int(obs["ball_owned_team"])
                assert owner in (-1, 0, 1)
                counts[{0: 0, 1: 1, -1: 2}[owner]] += 1
                streak = streak + 1 if owner != -1 and owner == previous_owner else (1 if owner != -1 else 0)
                previous_owner = owner
                frame = render_frame(base, obs, decisions[decision_index], count, 25)
                frame = add_possession_hud(frame, obs, counts, streak)
                writer.write(frame)
                if count == 1500:
                    assert cv2.imwrite(str(folder / "clean_pitch_possession_frame_1500.png"), frame)
                last_score = list(obs["score"])
                count += 1
    finally:
        writer.release()
    assert count == 3001 and sum(counts) == count
    assert last_score == timeline["final_score"]
    owned = counts[0] + counts[1]
    result = {"source_dump": dump.name,
              "source_dump_sha256": hashlib.sha256(dump.read_bytes()).hexdigest(),
              "timeline_sha256": hashlib.sha256(timeline_file.read_bytes()).hexdigest(),
              "frames": count, "video_fps": 25, "playback_speed": 2.5,
              "final_score_left_right": last_score,
              "ball_owned_engine_steps_blue_red_loose": counts,
              "time_with_ball_seconds_blue_red_loose": [round(v / 10, 1) for v in counts],
              "owned_time_share_blue_red": [counts[0] / owned, counts[1] / owned],
              "measurement": "GRF ball_owned_team each engine step; loose excluded from percentage",
              "rendering": "2D visualization of recorded GRF engine observations, not 3D rendering"}
    (folder / "clean_pitch_possession_info.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
