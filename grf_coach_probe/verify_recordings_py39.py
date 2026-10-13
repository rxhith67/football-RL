"""Check native GRF AVI files decode and save a representative frame."""

import json
import sys
from pathlib import Path

import cv2


def main():
    directory = Path(sys.argv[1])
    result = []
    for avi in sorted(directory.glob("*.avi")):
        cap = cv2.VideoCapture(str(avi))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
        success, frame = cap.read()
        png = avi.with_suffix(".png")
        if success:
            cv2.imwrite(str(png), frame)
        cap.release()
        result.append({"video": str(avi), "bytes": avi.stat().st_size,
                       "frames": frames, "fps": fps,
                       "width": width, "height": height,
                       "mid_frame_decoded": success,
                       "screenshot": str(png) if success else None})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
