"""Record two predetermined demonstration matches, one with Q on each side.

Run: bash run.sh head_to_head_demo.py --seed 970001
Demonstration seeds are disjoint from the aggregate head-to-head evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from coach_env import GRFDepthCoachEnv
from head_to_head_common import CHECKPOINT, run_match
from head_to_head_eval import FROZEN_SHA256


def annotate_video(folder, match):
    import cv2

    videos = sorted(folder.glob("episode_done_*.avi"))
    assert videos, "GRF did not write its full-match gameplay recording"
    cap = cv2.VideoCapture(str(videos[0]))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(folder / "annotated.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    assert writer.isOpened()
    starts = [d["engine_step_start"] for d in match["decisions"]]
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k = max(0, min(29, int(np.searchsorted(starts, count, side="right") - 1)))
        d = match["decisions"][k]
        own, opp = d["score_own_opp_start"]
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
        label = ("RL %s vs native | %s | decision %d/30 | step %d | score at decision %d-%d" %
                 (match["coached_side"], d["action_name"], k + 1, count, own, opp))
        cv2.putText(frame, label, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 80), 1)
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    assert count == 3001, "recording frame count does not match match steps"
    return {"source_video": videos[0].name, "annotated_video": "annotated.avi", "frames": count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=970001)
    ap.add_argument("--out", type=Path, default=Path("demo/head_to_head_seed970001"))
    args = ap.parse_args()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == FROZEN_SHA256
    assert int(np.load(CHECKPOINT)["iter"]) == 100
    args.out.mkdir(parents=True, exist_ok=False)
    q = np.load(CHECKPOINT)["q"]
    summary = {"label": "demonstration only; excluded from aggregate evaluation",
               "seed": args.seed, "checkpoint_sha256": FROZEN_SHA256, "legs": []}
    for side, label in ((1, "rl_right"), (0, "rl_left")):
        folder = args.out / label
        folder.mkdir()
        env = GRFDepthCoachEnv(decision_steps=100, write_video=True, logdir=folder)
        try:
            match = run_match(env, side, args.seed, q)
        finally:
            env.close()
        (folder / "timeline.json").write_text(json.dumps(match, indent=2) + "\n")
        video = annotate_video(folder, match)
        summary["legs"].append({"coached_side": match["coached_side"],
                                "score_left_right": match["score_left_right"],
                                "rl_goal_difference": match["rl_goal_difference"],
                                "action_counts_compact_default_stretched": match["action_counts_compact_default_stretched"],
                                "video": video})
        print(label, "RL goal difference", match["rl_goal_difference"],
              "actions", match["action_counts_compact_default_stretched"], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("demonstrations saved", args.out, flush=True)


if __name__ == "__main__":
    main()
