#!/usr/bin/env bash
# Complete the benchmark after the rendered-image suite, all on GPU 1:
# two grid shards and the six new representatives run concurrently, then the
# statistics, audit and tables are regenerated.
#
# The image suite (batch 64) and IHDP 100 are expected to be complete; the
# caller finishes them first. Wall-clock timing is deferred: set RUN_TIMING=1
# to run the sequential timing pass, which must not overlap anything.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  PY=".venv/bin/python"
elif uv run --no-sync python -c "import torch" >/dev/null 2>&1; then
  PY="uv run --no-sync python"
else
  PY="python"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GCMTS_PREFETCH=1
export GCMTS_LIGHT_BATCH=1
export CUDA_VISIBLE_DEVICES=1
mkdir -p outputs/logs

GRID_FLAGS=(
  --device cuda --grid --shard-total 2 --out-dir outputs_shared
  --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic
  --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99
  --kl-warmup-epochs 50
)

echo "== grid (2 shards) and representatives concurrently on GPU 1 =="
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_crl_shared.py "${GRID_FLAGS[@]}" \
  --shard-index 0 --stem crl_shared_grid_a \
  > outputs/logs/pending_grid_a.log 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_crl_shared.py "${GRID_FLAGS[@]}" \
  --shard-index 1 --stem crl_shared_grid_b \
  > outputs/logs/pending_grid_b.log 2>&1 &
PID_B=$!
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_new_representatives.py \
  --device cuda --seeds 10 --epochs 500 \
  --out outputs/new_representatives.json \
  > outputs/logs/pending_reps.log 2>&1 &
PID_R=$!

status=0
wait "$PID_A" || status=1
wait "$PID_B" || status=1
wait "$PID_R" || status=1
if [ "$status" -ne 0 ]; then
  echo "a grid shard or the representative run failed; see outputs/logs/." >&2
  exit 1
fi

if [ "${RUN_TIMING:-0}" = "1" ]; then
  echo "== sequential timing pass (explicitly requested) =="
  bash scripts/run_pending_timing.sh "" cuda
else
  echo "== analysis and tables (wall-clock times deferred) =="
  "$PY" scripts/statds_report.py --outputs outputs --shared outputs_shared \
    --out docs/statistical_analysis.md --json-out docs/statistical_analysis.json
  "$PY" scripts/reference_floors.py --outputs outputs --out docs/reference_floors.md
  "$PY" scripts/verify_outputs.py --pools --report docs/audit_pending.md || true
  "$PY" scripts/make_paper_tables.py --shared-dir outputs_shared \
    --outputs-dir outputs --latex-dir ../latex
fi

echo "== complete =="
date -Is
