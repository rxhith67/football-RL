#!/usr/bin/env bash
# SMALL PRELIMINARY EVALUATION (NOT YET APPROVED TO RUN): native GRF vs A (original Q) vs B (possession-reward Q, 300-match pilot).
# 30 fresh seeds (3,100,000-3,100,029), identical for every policy, left side only. Policy C can be added later to the same output
# folder with the same seeds (results are cached per policy). 2 workers, lowest priority, 10-match batches with 60 s rests.
# Uses the immutable iteration-25 snapshot of B, so later training cannot change what was evaluated.
cd "$(dirname "$0")"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh evaluate_poss.py --mode independent --episodes 30 --seed0 3100000 --out pilot_eval --workers 2 --chunk 10 --pause 60 \
     --policy native \
     --policy A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz \
     --policy B_q_possession=checkpoints/q_possession_run1_it25.npz
pkill -f "[m]onitor.sh"
OUT_TAG=_pilot_AB EVAL_IND=pilot_eval EVAL_H2H=none_pilot bash run.sh analyze_poss.py > logs/pilot_AB_analysis.out 2>&1
echo "EVAL_AB_DONE: see results/possession_tables_pilot_AB.md (preliminary)"
