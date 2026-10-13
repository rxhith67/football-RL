#!/usr/bin/env bash
# usage: bash run.sh script.py [args...]   (run from anywhere; executes inside grf_possession/ against the PATCHED engine)
# Install locations (see REPRODUCE.md); override with environment variables if you installed elsewhere.
GRF_PROBE="${GRF_PROBE:-$HOME/grf_coach_probe}"      # Python 3.9 toolchain + stock GRF
GRF_POC="${GRF_POC:-$HOME/grf_native_poc}"           # patched engine copy; overlay/ holds the patched native library
GRF_VENV="${GRF_VENV:-$HOME/grf_rl_env}"             # virtualenv with numpy/matplotlib
HERE="$(cd "$(dirname "$0")" && pwd)"
COACH="$(cd "$HERE/../grf_rl_coach" && pwd)"          # existing project, imported READ-ONLY
export PATH=$GRF_PROBE/py39/bin:$PATH
export LD_LIBRARY_PATH=$GRF_PROBE/py39/lib
export PYTHONPATH=$GRF_POC/overlay:$HERE:$COACH
export PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
$GRF_VENV/bin/python "$@" 2>&1 | grep --line-buffered -v RuntimeWarning