#!/usr/bin/env bash
# PILOT STAGE 2: train Experiment C (Double Q-learning + possession reward) for exactly 300 matches (iteration 25),
# same settings and the same match seeds as B (run-id 1). 2 workers, lowest priority, 30 s rest after each iteration. Does NOT start any other stage.
cd "$(dirname "$0")"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh train_poss.py --algo double_q --reward possession --run-id 1 --iters 100 --stop-after 25 --episodes-per-iter 12 \
     --workers 2 --cooldown 30
pkill -f "[m]onitor.sh"
echo "STAGE2_DONE: C stopped at 300 matches; snapshot checkpoints/double_q_possession_run1_it25.npz"
