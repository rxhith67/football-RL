"""Draw a readable 2D pitch replay from a saved, unrendered GRF match trace.

This is a visualization of recorded GRF positions and scores. It does not
simulate, edit the checkpoint, or reconstruct the original 3D renderer.

Run: bash run.sh render_clean_replay.py
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


SIZE = (960, 620)
X0, X1, Y0, Y1 = 90, 870, 80, 580
MID_X, MID_Y = (X0 + X1) // 2, (Y0 + Y1) // 2
WHITE = (235, 244, 235)
BLUE = (230, 145, 45)
RED = (65, 80, 220)


def point(x, y):
    return (int(round(X0 + (float(x) + 1.0) * (X1 - X0) / 2.0)),
            int(round(Y0 + (float(y) + 0.42) * (Y1 - Y0) / 0.84)))


def text(im, message, pos, scale=0.55, color=WHITE, thick=1):
    cv2.putText(im, message, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def pitch_background():
    im = np.full((SIZE[1], SIZE[0], 3), (23, 45, 26), np.uint8)
    cv2.rectangle(im, (X0, Y0), (X1, Y1), (45, 126, 55), -1)
    for stripe in range(10):
        if stripe % 2 == 0:
            a = X0 + stripe * (X1 - X0) // 10
            b = X0 + (stripe + 1) * (X1 - X0) // 10
            cv2.rectangle(im, (a, Y0), (b, Y1), (48, 136, 60), -1)
    cv2.rectangle(im, (X0, Y0), (X1, Y1), WHITE, 2)
    cv2.line(im, (MID_X, Y0), (MID_X, Y1), WHITE, 2)
    cv2.circle(im, (MID_X, MID_Y), 68, WHITE, 2, cv2.LINE_AA)
    cv2.circle(im, (MID_X, MID_Y), 3, WHITE, -1)
    penalty_depth, penalty_half_width = 123, 148
    goal_depth, goal_half_width = 41, 68
    for side_x, direction in ((X0, 1), (X1, -1)):
        cv2.rectangle(im, (min(side_x, side_x + direction * penalty_depth), MID_Y - penalty_half_width),
                      (max(side_x, side_x + direction * penalty_depth), MID_Y + penalty_half_width), WHITE, 2)
        cv2.rectangle(im, (min(side_x, side_x + direction * goal_depth), MID_Y - goal_half_width),
                      (max(side_x, side_x + direction * goal_depth), MID_Y + goal_half_width), WHITE, 2)
        cv2.circle(im, (side_x + direction * 82, MID_Y), 3, WHITE, -1)
        cv2.rectangle(im, (min(side_x, side_x - direction * 14), MID_Y - 27),
                      (max(side_x, side_x - direction * 14), MID_Y + 27), WHITE, 2)
    cv2.rectangle(im, (0, 0), (SIZE[0], Y0 - 1), (28, 37, 32), -1)
    cv2.rectangle(im, (0, Y1 + 1), (SIZE[0], SIZE[1]), (28, 37, 32), -1)
    text(im, "TACTICAL AI  |  GRF 11v11", (24, 25), 0.68, WHITE, 2)
    text(im, "RECORDED GRF MATCH  -  TOP-DOWN REPLAY", (24, 607), 0.47,
         (205, 217, 205), 1)
    return im


def render_frame(base, obs, decision, frame_index, fps):
    im = base.copy()
    score = obs["score"]
    text(im, "BLUE %d  -  %d RED" % (score[0], score[1]), (24, 57), 0.67, WHITE, 2)
    action = decision["action"].replace("_", " ").upper()
    text(im, "COACH: %s" % action, (385, 56), 0.57, (110, 230, 255), 2)
    text(im, "STEP %d/3001  |  DECISION %d/30" % (frame_index, decision["decision"]),
         (630, 25), 0.44, WHITE, 1)
    text(im, "2.5x playback", (806, 607), 0.40, (205, 217, 205), 1)
    owner_team = int(obs["ball_owned_team"])
    owner_player = int(obs["ball_owned_player"])
    for team_index, key, color in ((0, "left_team", BLUE), (1, "right_team", RED)):
        for player_index, (x, y) in enumerate(obs[key]):
            px, py = point(x, y)
            if owner_team == team_index and owner_player == player_index:
                cv2.circle(im, (px, py), 15, (34, 218, 244), 3, cv2.LINE_AA)
            cv2.circle(im, (px + 2, py + 3), 11, (25, 55, 29), -1, cv2.LINE_AA)
            cv2.circle(im, (px, py), 10, color, -1, cv2.LINE_AA)
            cv2.circle(im, (px, py), 10, WHITE, 1, cv2.LINE_AA)
            label = "G" if player_index == 0 else str(player_index)
            text(im, label, (px - (4 if len(label) == 1 else 7), py + 4),
                 0.33, WHITE, 1)
    bx, by = point(obs["ball"][0], obs["ball"][1])
    cv2.circle(im, (bx + 2, by + 3), 7, (18, 50, 25), -1, cv2.LINE_AA)
    cv2.circle(im, (bx, by), 6, (252, 253, 250), -1, cv2.LINE_AA)
    cv2.circle(im, (bx, by), 6, (30, 40, 30), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=Path("demo/q100_seed800001"))
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    assert args.fps == 25, "caption currently specifies 2.5x playback"
    folder = args.folder
    dumps = list(folder.glob("episode_done_*.dump"))
    assert len(dumps) == 1
    dump = dumps[0]
    timeline_file = folder / "timeline.json"
    timeline = json.loads(timeline_file.read_text())
    decisions = timeline["decisions"]
    assert len(decisions) == 30 and timeline["verification"]["engine_steps_total"] == 3001
    output = folder / "clean_pitch.avi"
    assert not output.exists(), "refusing to overwrite an existing clean replay"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"XVID"), args.fps, SIZE)
    assert writer.isOpened()
    base = pitch_background()
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
                frame = render_frame(base, obs, decisions[decision_index], count, args.fps)
                writer.write(frame)
                if count == 1500:
                    assert cv2.imwrite(str(folder / "clean_pitch_frame_1500.png"), frame)
                last_score = list(obs["score"])
                count += 1
    finally:
        writer.release()
    assert count == 3001, count
    assert last_score == timeline["final_score"], (last_score, timeline["final_score"])
    info = {"source_dump": dump.name,
            "source_dump_sha256": hashlib.sha256(dump.read_bytes()).hexdigest(),
            "timeline_sha256": hashlib.sha256(timeline_file.read_bytes()).hexdigest(),
            "frames": count, "video_fps": args.fps, "match_engine_steps_per_second": 10,
            "playback_speed": 2.5, "final_score_left_right": last_score,
            "rendering": "2D visualization of recorded GRF engine observations; not original 3D rendering"}
    (folder / "clean_pitch_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
