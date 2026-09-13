#!/usr/bin/env bash
# Usage: run_bench_monitored.sh <label> <container> [bench args...]
# Runs bench_steps.py while sampling GPU util, NVMe I/O, engine thread placement and major page faults.
set -u
LABEL=$1; CONT=$2; shift 2
cd ~/research
PID=$(pgrep -f "VLLM::EngineCore" | head -1)
echo "ENGINE_PID=$PID label=$LABEL container=$CONT"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw --format=csv,noheader -l 1 > logs/gpu-util-$LABEL.log 2>&1 &
P1=$!
iostat -dx nvme0n1 1 > logs/iostat-$LABEL.log 2>&1 &
P2=$!
( for i in $(seq 1 600); do ps -L -o tid,psr,pcpu,comm -p $PID --no-headers | awk -v t=$(date +%s) '$3>5 {print t, $0}'; sleep 1; done ) > logs/threads-$LABEL.log 2>&1 &
P3=$!
( for i in $(seq 1 600); do awk -v t=$(date +%s) '{print t, "majflt", $12, "minflt", $10}' /proc/$PID/stat; sleep 1; done ) > logs/faults-$LABEL.log 2>&1 &
P4=$!
sleep 2
python3 bench_steps.py --label "$LABEL" "$@" 2>&1
sleep 2
kill $P1 $P2 $P3 $P4 2>/dev/null
echo
echo "=== GPU util: mean util.gpu over samples with power > 30W ==="
awk -F', ' '{gsub(/ %| W| MHz/,""); if ($4+0 > 30) {s+=$1; n++}} END {if (n) printf "mean_util=%.1f%% over %d busy samples\n", s/n, n; else print "no busy samples"}' logs/gpu-util-$LABEL.log
echo "=== iostat: read IOPS / kB/s during run ==="
grep nvme0n1 logs/iostat-$LABEL.log | awk '{if($2>mr)mr=$2; if($3>mk)mk=$3; s+=$2; n++} END {print "max r/s", mr, "max rkB/s", mk, "mean r/s", s/n}'
echo "=== major faults of the engine during the run ==="
awk 'NR==1{m0=$3} {m=$3} END {print "majflt delta:", m-m0}' logs/faults-$LABEL.log
echo "=== hot engine threads: (comm, cpu, %cpu bucket) counts ==="
awk '{printf "%s cpu%s %s\n", $5, $3, ($4>80?">80%":($4>40?"40-80%":"5-40%"))}' logs/threads-$LABEL.log | sort | uniq -c | sort -rn | head -8
