#!/usr/bin/env bash
# usage: bash run.sh script.py [args...]   (run from anywhere; executes inside grf_possession/ against the PATCHED engine)
HERE="$(cd "$(dirname "$0")" && pwd)"
COACH="$(cd "$HERE/../grf_rl_coach" && pwd)"          # existing project, imported READ-ONLY
export PATH=$HOME/grf_coach_probe/py39/bin:$PATH
export LD_LIBRARY_PATH=$HOME/grf_coach_probe/py39/lib
export PYTHONPATH=$HOME/grf_native_poc/overlay:$HERE:$COACH
export PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
$HOME/grf_rl_env/bin/python "$@" 2>&1 | grep --line-buffered -v RuntimeWarning
