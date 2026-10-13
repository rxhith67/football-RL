#!/usr/bin/env bash
# usage: bash run.sh script.py [args...]   -- runs inside WSL against the PATCHED engine only
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH=$HOME/grf_coach_probe/py39/bin:$PATH
export LD_LIBRARY_PATH=$HOME/grf_coach_probe/py39/lib
export PYTHONPATH=$HOME/grf_native_poc/overlay:$HERE
export PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
$HOME/grf_rl_env/bin/python "$@" 2>&1 | grep -v RuntimeWarning
