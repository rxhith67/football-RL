#!/usr/bin/env bash
# FULL 800-MATCH EVALUATION: native, A, B (final it100), C (final it100) x 100 fresh seed pairs (seeds 7,000,000-7,000,099) x both sides.
# Greedy, no exploration, no model D. Worker count is read from logs/eval100_workers.txt before EACH policy (default 3); lowest priority;
# 10-match batches with a 60 s rest. Each policy's 200 matches are cached when complete, so a re-run skips finished policies.
cd "$(dirname "$0")"
OUT=eval_final100
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
for POL in "native" "A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz" "B_q_possession=checkpoints/q_possession_run1_it100.npz" "C_dq_possession=checkpoints/double_q_possession_run1_it100.npz"; do
  W=$(cat logs/eval100_workers.txt 2>/dev/null || echo 3)
  echo "=== $(date +%T) policy ${POL%%=*} workers=$W"
  nice -n 19 bash run.sh evaluate_poss.py --mode headtohead --episodes 100 --seed0 7000000 --out $OUT --workers $W --chunk 10 --pause 60 --policy "$POL"
done
pkill -f "[m]onitor.sh"
echo "=== $(date) EVAL100_DONE"