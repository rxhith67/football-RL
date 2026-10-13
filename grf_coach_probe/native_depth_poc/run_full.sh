#!/usr/bin/env bash
export PATH=$HOME/grf_coach_probe/py39/bin:$PATH LD_LIBRARY_PATH=$HOME/grf_coach_probe/py39/lib PYTHONPATH=$HOME/grf_native_poc/overlay PYTHONDONTWRITEBYTECODE=1
export PROBE_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$(dirname "$0")"
for dd in -1 0.0 1.0; do python full_match_check.py $dd results/full_match_$dd.json > results/full_match_$dd.log 2>&1 & done; wait; echo ok > results/full_done.flag
