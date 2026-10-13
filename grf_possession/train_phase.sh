#!/usr/bin/env bash
# Sequential training queue, resource-limited. usage: bash train_phase.sh RUN_ID "algo reward" ["algo reward" ...]
#   WORKERS  (default 2)   simulation processes; results do NOT depend on this (each match is a deterministic function of its seed)
#   CPUSET   (optional)    e.g. CPUSET=16-17 pins the whole queue to those logical CPUs (taskset)
#   Resumes automatically from checkpoints/<tag>.npz when it exists; a finished run (iteration 100) is skipped.
cd "$(dirname "$0")"
WORKERS=${WORKERS:-2}
RUN=$1; shift
for cfg in "$@"; do
  set -- $cfg
  TAG="$1_$2_run$RUN"
  PIN=""; [ -n "$CPUSET" ] && PIN="taskset -c $CPUSET"
  RESUME=""
  if [ -f "checkpoints/$TAG.npz" ]; then
    DONE=$(wc -l < "logs/$TAG.jsonl")
    if [ "$DONE" -ge 100 ]; then echo "=== $TAG already complete, skipping"; continue; fi
    RESUME="--resume"
  fi
  echo "=== $(date) start $TAG workers=$WORKERS cpuset=${CPUSET:-all} $RESUME"
  $PIN nice -n 19 bash run.sh train_poss.py --algo $1 --reward $2 --run-id $RUN --iters 100 --episodes-per-iter 12 --workers $WORKERS $RESUME
  echo "=== $(date) finished $TAG"
done
echo PHASE_DONE
