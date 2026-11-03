#!/usr/bin/env bash
# Sync the code, data and paper tree to gpu_212 and launch the complete
# benchmark inside a detached screen session named "Benchmark".
#
# Usage (from the gcmts repository):
#   bash scripts/sync_and_launch_gpu212.sh
#
# Excludes virtualenvs, caches, previous outputs and build artefacts. The remote
# previous outputs (if any) are moved aside, not deleted. The remote needs a
# Python environment with torch; run `uv sync` there once if it is missing.
set -euo pipefail

HOST="${HOST:-gpu_212}"
REMOTE="${REMOTE:-gcmts}"
SESSION="${SESSION:-Benchmark}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$(dirname "$ROOT")"

echo "== rsync code and data to $HOST:~/$REMOTE =="
rsync -av \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'outputs' \
  --exclude 'outputs_shared' \
  --exclude 'outputs_timing' \
  --exclude 'outputs_shared_timing' \
  --exclude 'data/synthetic_smoke' \
  "$ROOT/" "$HOST:~/$REMOTE/"

echo "== rsync the paper tree (for table generation) =="
rsync -av \
  --exclude '.git' \
  --exclude 'main.pdf' \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.bcf' \
  --exclude '*.blg' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.fls' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.run.xml' \
  --exclude '*.toc' \
  "$PAPER/latex/" "$HOST:~/latex/"

echo "== prepare the remote tree =="
ssh "$HOST" "cd ~/$REMOTE && mkdir -p outputs/logs && \
  for d in outputs outputs_shared outputs_timing outputs_shared_timing; do \
    if [ -d \"\$d\" ]; then mv \"\$d\" \"\${d}_backup_\$(date +%s)\"; fi; \
  done"

echo "== launch screen session '$SESSION' =="
ssh "$HOST" "cd ~/$REMOTE && screen -L -dmS $SESSION env GPU=1 DEVICE=cuda JOBS=2 bash scripts/run_benchmark_complete.sh"

echo
echo "Started. Attach with:  ssh $HOST -t 'screen -r $SESSION'"
echo "Detach with Ctrl-A D. Check with: ssh $HOST 'screen -ls'"
echo "Logs: ~/$REMOTE/outputs/logs/ and ~/$REMOTE/screenlog.0"
