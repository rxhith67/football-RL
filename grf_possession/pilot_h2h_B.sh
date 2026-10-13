#!/usr/bin/env bash
# PRELIMINARY HEAD-TO-HEAD (NOT YET APPROVED TO RUN): Model B (iteration-25 snapshot, greedy, no exploration) vs native GRF AI.
# 30 fresh seeds (3,200,000-3,200,029); for every seed two full matches: B on the left / native on the right, then native on the left / B on the right.
# Both teams keep GRF's built-in player AI; only B's own team's defensive-depth parameter is set (opponent override cleared, never written).
# 2 workers, lowest priority, 10-match batches (5 seed pairs) with a 60 s rest after each batch. Results: results/eval_h2h_pilot/ (new folder).
cd "$(dirname "$0")"
(INTERVAL=30 setsid bash monitor.sh > /dev/null 2>&1 < /dev/null &)
nice -n 19 bash run.sh evaluate_poss.py --mode headtohead --episodes 30 --seed0 3200000 --out eval_h2h_pilot --workers 2 --chunk 10 --pause 60 \
     --policy B_q_possession=checkpoints/q_possession_run1_it25.npz
pkill -f "[m]onitor.sh"
OUT_TAG=_h2h_B EVAL_IND=none_pilot EVAL_H2H=eval_h2h_pilot bash run.sh analyze_poss.py > logs/pilot_h2h_B_analysis.out 2>&1
echo "H2H_B_DONE: see results/possession_tables_h2h_B.md (preliminary)"
