"""Inspect GRF recording frame and trace alignment (read-only)."""
import argparse
import pickle
from pathlib import Path

import cv2


parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()
dump, = args.folder.glob("episode_done_*.dump")
video, = args.folder.glob("episode_done_*.avi")
cap = cv2.VideoCapture(str(video))
print("video frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps", cap.get(cv2.CAP_PROP_FPS))
cap.release()
rows = []
with dump.open("rb") as stream:
    while True:
        try:
            rows.append(pickle.load(stream))
        except EOFError:
            break
print("trace rows", len(rows))
for i in [0, 1, 500, 1500, 2500, len(rows) - 1]:
    if i >= len(rows):
        continue
    row = rows[i]
    print(i, "score", row["observation"]["score"], "steps_left",
          row["observation"].get("steps_left"), "frame_cnt",
          row["debug"].get("frame_cnt"), "debug_keys", list(row["debug"])[:15])
