#!/bin/bash
# RunPod bootstrap for the p=40 sweep on 8×A100 80GB SXM.
#
# Usage on a fresh RunPod pod (PyTorch 2.1 + CUDA 11.8 template recommended):
#   curl -sSL https://raw.githubusercontent.com/kvrancic/nikolija-neurips/main/scripts/runpod_bootstrap.sh | bash
# OR after cloning:
#   bash scripts/runpod_bootstrap.sh
#
# What this does:
#   1. Clone repo (if not already in /workspace/nikolija-neurips)
#   2. pip install requirements
#   3. Launch the full p=40 sweep on all 8 GPUs in tmux
#   4. Print the rsync command to copy results back when done

set -euo pipefail

REPO_URL="https://github.com/kvrancic/nikolija-neurips.git"
WORKDIR=/workspace/nikolija-neurips

if [ ! -d "$WORKDIR" ]; then
    cd /workspace
    git clone "$REPO_URL" nikolija-neurips
fi
cd "$WORKDIR"
git pull --ff-only || true

pip install --quiet -r requirements.txt

# Detect GPU count
NGPU=$(nvidia-smi -L | wc -l)
echo "Detected $NGPU GPUs"
GPUS=$(seq -s, 0 $((NGPU - 1)))
RESTARTS=$NGPU

mkdir -p logs results
SESSION=crl_p40_runpod

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already exists. Attach with: tmux attach -t $SESSION"
    exit 0
fi

LOG=logs/sweep_p40_runpod_$(date +%Y%m%dT%H%M%S).log

tmux new-session -d -s "$SESSION" \
    "cd $WORKDIR && make sweep-p40-full GPUS=$GPUS RESTARTS=$RESTARTS 2>&1 | tee $LOG"

echo ""
echo "=== p=40 sweep launched on $NGPU GPUs in tmux session '$SESSION' ==="
echo "Attach:   tmux attach -t $SESSION"
echo "Tail log: tail -f $WORKDIR/$LOG"
echo "CSV:      $WORKDIR/results/p40/p_recovery_sweep.csv"
echo ""
echo "When done, copy results back:"
echo "  rsync -av $WORKDIR/results/p40/ <your-laptop>:~/nikolija-results/p40/"
echo "or push to GitHub from inside the pod:"
echo "  cd $WORKDIR && git add -f results/p40 && git commit -m 'results: p40 RunPod' && git push"
