#!/usr/bin/env bash
# profile_run.sh <label> [port] -- start torch profiler, run a short code generation, stop, list trace
LABEL=$1; PORT=${2:-9115}
cd ~/research
curl -s -X POST localhost:$PORT/start_profile >/dev/null && echo "profiler started"
python3 bench_steps.py --base http://127.0.0.1:$PORT --cases ${CASES:-code} --reps 1 --max-tokens ${TOK:-120} 2>&1 | tail -3
curl -s -X POST localhost:$PORT/stop_profile >/dev/null && echo "profiler stopped"
sleep 10
ls -la profiles/$LABEL/ | tail -5
