#!/usr/bin/env bash
# End-to-end smoke test: 2 tiny iterations of each configuration, separate run id (99) and tags so nothing real is touched.
cd "$(dirname "$0")"
for cfg in "q possession" "double_q possession" "double_q original"; do
  set -- $cfg
  nice -n 10 bash run.sh train_poss.py --algo $1 --reward $2 --run-id 99 --iters 2 --episodes-per-iter 6 --workers 6 --eval-every 2 --tag smoke_$1_$2
done
echo SMOKE_DONE
