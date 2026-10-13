#!/usr/bin/env bash
# Throwaway interruption test on its own tag (selftest_stop, run-id 98): clean stop-after, hard kill mid-iteration, resume.
# 2 workers, lowest priority. Touches nothing else.
cd "$(dirname "$0")"
TAG=selftest_stop
COMMON="--algo double_q --reward possession --run-id 98 --iters 100 --episodes-per-iter 2 --workers 2 --dev-seeds 2 --eval-every 1000 --tag $TAG"
echo "### A: clean stop after 2 iterations"
nice -n 19 bash run.sh train_poss.py $COMMON --stop-after 2 --cooldown 5 2>&1 | cut -c1-110
echo "### B: resume to iteration 4, then hard-kill (SIGKILL) the whole process group 40 s in, mid-iteration"
setsid nice -n 19 bash run.sh train_poss.py $COMMON --stop-after 4 --resume > logs/selftest_B.out 2>&1 &
PG=$!
sleep 40
kill -9 -- -$PG 2>/dev/null
sleep 3
echo "processes left after kill: $(ps -eo cmd | grep -c '[m]ultiprocessing.spawn')"
echo "### C: checkpoint must still load and resume cleanly to iteration 3"
nice -n 19 bash run.sh train_poss.py $COMMON --stop-after 3 --resume 2>&1 | cut -c1-110
echo SELFTEST_DONE
