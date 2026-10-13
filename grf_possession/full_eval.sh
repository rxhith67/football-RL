#!/usr/bin/env bash
# FULL TWO-SIDED EVALUATION (not run until approved): native vs A vs B vs C (and D if it was trained), identical fresh seeds,
# every policy plays the native AI on BOTH sides for every seed (so the left/right strength imbalance cancels in the pair mean).
# Seeds 5,000,000 + 0..(PAIRS-1): never used before. Final iteration-100 snapshots only (no checkpoint selection on test data).
# usage: PAIRS=60 bash full_eval.sh           2 workers, nice 19, 10-match batches with a 60 s rest.
cd "$(dirname "$0")"
PAIRS=${PAIRS:-60}
ARGS="--policy native --policy A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz"
ARGS="$ARGS --policy B_q_possession=checkpoints/q_possession_run1_it100.npz --policy C_dq_possession=checkpoints/double_q_possession_run1_it100.npz"
[ -f checkpoints/double_q_original_run1_it100.npz ] && ARGS="$ARGS --policy D_dq_original=checkpoints/double_q_original_run1_it100.npz"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh evaluate_poss.py --mode headtohead --episodes $PAIRS --seed0 5000000 --out eval_h2h_final --workers 2 --chunk 10 --pause 60 $ARGS
pkill -f "[m]onitor.sh"
OUT_TAG=_final EVAL_IND=none_final EVAL_H2H=eval_h2h_final bash run.sh analyze_poss.py > logs/final_analysis.out 2>&1
echo "EVAL_DONE: see results/possession_tables_final.md"
