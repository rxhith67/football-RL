#!/usr/bin/env bash
set -euo pipefail

probe="$HOME/grf_coach_probe"
export PATH="$probe/py39/bin:$PATH"
mkdir -p "$probe/logs"
python --version
pip --version

# GRF requires Gym 0.21. The old package builds with these pinned tooling
# versions under Python 3.9, without changing Ubuntu's system Python.
pip install --no-build-isolation \
  'gym==0.21.0' 'numpy==1.23.5' 'psutil==5.9.8' \
  'pygame==2.6.1' 'opencv-python==4.8.1.78' 'absl-py==2.1.0' \
  'six==1.16.0' \
  > "$probe/logs/dependencies.log" 2>&1 || {
    tail -n 80 "$probe/logs/dependencies.log"
    exit 1
  }

python -c 'import gym, numpy, pygame, cv2, psutil, absl; print("gym", gym.__version__, "numpy", numpy.__version__, "pygame", pygame.__version__, "cv2", cv2.__version__)'
