#!/usr/bin/env bash
# FULL-BUDGET TRAINING STAGE (not run until approved): one configuration to 100 iterations x 12 = 1,200 matches.
# usage: bash full_train.sh ALGO REWARD        (q|double_q) (possession|original)
# Same for every configuration: run-id 1 (match seeds 2,010,000-2,011,199), 243 states, 3 actions, 100-iteration epsilon schedule
# 0.5 -> 0.05 over the first 70 iterations, alpha 0.1, gamma 0.99, 12 matches per update batch, greedy dev check every 10 iterations.
# Resource limits: 2 workers, nice 19, 30 s rest after every iteration, resource monitor. Resumes automatically if a checkpoint exists.
# Recovery: checkpoints are written atomically every iteration (tables, counters, learner RNG state); just re-run this same command.
# Does NOT start any other stage.
cd "$(dirname "$0")"
ALGO=$1; REWARD=$2; TAG="${ALGO}_${REWARD}_run1"
if [ -f "checkpoints/$TAG.npz" ]; then
  DONE=$(wc -l < "logs/$TAG.jsonl")
  if [ "$DONE" -ge 100 ]; then echo "$TAG already complete (100 iterations)"; exit 0; fi
  RESUME="--resume"
  cp -n "checkpoints/$TAG.npz" "checkpoints/${TAG}_pre_resume_it${DONE}.npz"     # full backup incl. counters and learner RNG state
  sha256sum checkpoints/$TAG*.npz logs/$TAG.jsonl > "logs/HASHES_before_resume_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
else RESUME=""; fi
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
echo "=== $(date) $TAG resume=[$RESUME]"
nice -n 19 bash run.sh train_poss.py --algo $ALGO --reward $REWARD --run-id 1 --iters 100 --episodes-per-iter 12 --workers 2 --cooldown 30 $RESUME
pkill -f "[m]onitor.sh"
echo "=== $(date) STAGE_DONE $TAG (iteration $(wc -l < logs/$TAG.jsonl) of 100)"
