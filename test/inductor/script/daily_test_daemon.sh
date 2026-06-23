#!/bin/bash
# Lightweight daily test scheduler (no cron/systemd needed)
# Runs run_daily_test.sh at 00:03 every day
# Usage: nohup bash daily_test_daemon.sh &

LOCKFILE=/tmp/daily_test_daemon.lock
SCRIPT=/models/torch-inductor/h00925030/test-daily/pytorch/test/inductor/scripts/run_daily_test.sh
LOG=/models/torch-inductor/h00925030/debug/pytorch/analysis_report/daily_test.log

if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Daemon already running (PID $PID)"
        exit 1
    fi
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

echo "[$(date)] Daily test daemon started" >> "$LOG"

while true; do
    NOW=$(date +%H%M)
    if [ "$NOW" = "0003" ]; then
        echo "[$(date)] Triggering daily test..." >> "$LOG"
        bash "$SCRIPT" >> "$LOG" 2>&1
        echo "[$(date)] Daily test completed" >> "$LOG"
        # Sleep past the minute to avoid re-triggering
        sleep 120
    fi
    sleep 30
done
