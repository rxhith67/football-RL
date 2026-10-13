#!/usr/bin/env bash
# PRELIMINARY two-sided evaluation (20 fresh seed pairs, seeds 6,000,000-6,000,019): native, A, B (final it100), C (final it100).
# Every policy plays both sides of every seed; left legs double as the fixed-left comparison. Greedy, no exploration. No model D.
# 2 workers, nice 19, 10-match batches with 60 s rest. The planned 60-pair evaluation (full_eval.sh, seeds 5,000,000+) is separate.
cd "$(dirname "$0")"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh evaluate_poss.py --mode headtohead --episodes 20 --seed0 6000000 --out eval_h2h_prelim --workers 2 --chunk 10 --pause 60 \
  --policy native --policy A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz \
  --policy B_q_possession=checkpoints/q_possession_run1_it100.npz --policy C_dq_possession=checkpoints/double_q_possession_run1_it100.npz
pkill -f "[m]onitor.sh"
echo "=== $(date) PRELIM_EVAL_DONE"