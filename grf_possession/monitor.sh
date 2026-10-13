#!/usr/bin/env bash
# Lightweight resource logger (negligible CPU): one CSV row every INTERVAL seconds -> logs/resource_usage.csv
# columns: time, cpu_busy_percent_of_all_24_threads, load1, ram_used_mb, simulation_processes, gpu_util, gpu_temp_c
cd "$(dirname "$0")"
INTERVAL=${INTERVAL:-30}
[ -f logs/resource_usage.csv ] || echo "time,cpu_busy_pct,load1,ram_used_mb,sim_procs,gpu_util,gpu_temp_c" > logs/resource_usage.csv
while true; do
  read -r _ u1 n1 s1 i1 w1 q1 sq1 _ < /proc/stat; sleep 2
  read -r _ u2 n2 s2 i2 w2 q2 sq2 _ < /proc/stat
  busy=$(( (u2+n2+s2+q2+sq2) - (u1+n1+s1+q1+sq1) )); tot=$(( busy + (i2+w2) - (i1+w1) ))
  pct=$(awk -v b=$busy -v t=$tot 'BEGIN{printf "%.1f", (t>0?100*b/t:0)}')
  load=$(cut -d' ' -f1 /proc/loadavg); ram=$(free -m | awk 'NR==2{print $3}')
  procs=$(ps -eo cmd | grep -c "[m]ultiprocessing.spawn")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  echo "$(date +%T),$pct,$load,$ram,$procs,${gpu:-na,na}" >> logs/resource_usage.csv
  sleep $INTERVAL
done
