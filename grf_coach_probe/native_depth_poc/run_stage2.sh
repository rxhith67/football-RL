#!/usr/bin/env bash
# usage: run_stage2.sh DIRS TAG   (launches 4 replicate processes detached, waits for them)
export PATH=$HOME/grf_coach_probe/py39/bin:$PATH LD_LIBRARY_PATH=$HOME/grf_coach_probe/py39/lib PYTHONPATH=$HOME/grf_native_poc/overlay PYTHONDONTWRITEBYTECODE=1
export PROBE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"; mkdir -p results
for r in 0 1 2 3; do
  python stage2_paired.py "$1" results/$2_rep$r.json $r > results/$2_rep$r.log 2>&1 &
done
wait
echo finished > results/$2_done.flag
