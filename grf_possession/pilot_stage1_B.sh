#!/usr/bin/env bash
# PILOT STAGE 1: finish Experiment B (Q-learning + possession reward) to exactly 300 training matches (iteration 25).
# Resumes from the iteration-19 checkpoint (228 matches). 2 workers, lowest priority, 30 s rest after each iteration. Does NOT start any other stage.
cd "$(dirname "$0")"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh train_poss.py --algo q --reward possession --run-id 1 --iters 100 --stop-after 25 --episodes-per-iter 12 \
     --workers 2 --cooldown 30 --resume
pkill -f "[m]onitor.sh"
echo "STAGE1_DONE: B stopped at 300 matches; snapshot checkpoints/q_possession_run1_it25.npz"
