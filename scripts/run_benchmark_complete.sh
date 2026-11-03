#!/usr/bin/env bash
# Complete benchmark run for the GPU host: pending blocks, new representatives,
# sequential timing, analysis and tables.
#
# Usage (inside the screen session):
#   GPU=1 DEVICE=cuda JOBS=2 bash scripts/run_benchmark_complete.sh
#
# The run is resumable. Each stage writes its own log under outputs/logs/.
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-10}"
EPOCHS="${EPOCHS:-500}"

if [ -n "$GPU" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export GPU DEVICE JOBS

mkdir -p outputs/logs

if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  PY=".venv/bin/python"
elif uv run --no-sync python -c "import torch" >/dev/null 2>&1; then
  PY="uv run --no-sync python"
elif python -c "import torch" >/dev/null 2>&1; then
  PY="python"
else
  echo "No Python interpreter with torch found." >&2
  exit 1
fi

echo "== complete benchmark: device=$DEVICE gpu=$GPU jobs=$JOBS =="
echo "interpreter: $PY"
echo "started: $(date -Is)"

echo "== 1/3 pending blocks (grid, IHDP, rendered images) =="
bash scripts/run_pending_tasks.sh

echo "== 2/3 six new representatives =="
$PY scripts/run_new_representatives.py \
  --device "$DEVICE" --seeds "$SEEDS" --epochs "$EPOCHS" \
  --out outputs/new_representatives.json

echo "== 3/3 sequential timing, merge, statistics and tables =="
bash scripts/run_pending_timing.sh

echo "finished: $(date -Is)" | tee outputs/BENCHMARK_DONE
