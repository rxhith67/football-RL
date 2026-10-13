#!/usr/bin/env bash
# PILOT STAGE 3: preliminary evaluation, 30 fresh seeds (3,100,000-3,100,029), same seeds for every policy, left side only.
# Policies: native GRF, A (original Q, frozen checkpoint), B and C (iteration-25 snapshots from stages 1 and 2).
# 2 workers, lowest priority, batches of 10 matches with a 60 s rest between batches. Does NOT start head-to-head or anything else.
cd "$(dirname "$0")"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh evaluate_poss.py --mode independent --episodes 30 --seed0 3100000 --out pilot_eval --workers 2 --chunk 10 --pause 60 \
     --policy native \
     --policy A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz \
     --policy B_q_possession=checkpoints/q_possession_run1_it25.npz \
     --policy C_dq_possession=checkpoints/double_q_possession_run1_it25.npz
pkill -f "[m]onitor.sh"
OUT_TAG=_pilot EVAL_IND=pilot_eval EVAL_H2H=none_pilot bash run.sh analyze_poss.py > logs/pilot_analysis.out 2>&1
echo "STAGE3_DONE: see results/possession_tables_pilot.md (preliminary)"
