#!/bin/bash
# Auto-queue: wait for p=5 to finish, then run p=20, then p=40.
# Polls the CSV row count rather than tmux state (more robust).

set -uo pipefail
cd ~/nikolija-neurips
export PATH="$HOME/broadenv/bin:$PATH"

LOG_DIR=logs
mkdir -p "$LOG_DIR"

wait_for_csv() {
    local csv="$1"
    local target_rows="$2"
    local label="$3"
    echo "$(date -Iseconds) [queue] waiting for $label ($csv >= $target_rows rows)"
    local last_rows=-1
    local stuck_count=0
    while true; do
        local rows
        rows=$(tail -n +2 "$csv" 2>/dev/null | wc -l)
        if [ "$rows" -ge "$target_rows" ]; then
            echo "$(date -Iseconds) [queue] $label DONE at $rows/$target_rows rows"
            return 0
        fi
        if [ "$rows" = "$last_rows" ]; then
            stuck_count=$((stuck_count + 1))
        else
            stuck_count=0
        fi
        last_rows=$rows
        if [ "$stuck_count" -gt 20 ]; then
            # 20 * 5min = 100 min with no progress; assume crashed
            echo "$(date -Iseconds) [queue] WARNING: $label stuck at $rows for 100 min, proceeding anyway"
            return 0
        fi
        sleep 300
    done
}

# Wait for p=5 to finish (90 rows expected)
wait_for_csv results/p5/p_recovery_sweep.csv 90 "p=5"
sleep 60

# Verify GPUs free
echo "$(date -Iseconds) [queue] GPU state before p=20:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

echo "$(date -Iseconds) [queue] launching sweep-p20"
make sweep-p20 GPUS=0,1,2,3 2>&1 | tee "$LOG_DIR/sweep_p20_$(date +%Y%m%dT%H%M%S).log"

# p=20 expects 48 rows (powers 8-13, seeds 8)
wait_for_csv results/p20/p_recovery_sweep.csv 48 "p=20"
sleep 60

echo "$(date -Iseconds) [queue] launching sweep-p40"
make sweep-p40 GPUS=0,1,2,3 2>&1 | tee "$LOG_DIR/sweep_p40_$(date +%Y%m%dT%H%M%S).log"

# p=40 expects 20 rows (powers 9-12, seeds 5)
wait_for_csv results/p40/p_recovery_sweep.csv 20 "p=40"

echo "$(date -Iseconds) [queue] ALL DONE"
