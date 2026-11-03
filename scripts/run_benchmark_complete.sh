#!/usr/bin/env bash
# Complete the benchmark after the rendered-image suite: grid on both GPUs,
# six new representatives split across the GPUs, then the sequential timing
# pass and the table regeneration.
#
# The image suite (batch 64) and IHDP 100 are expected to be complete; if they
# are not, the caller should finish them first. Timing is deliberately
# sequential and starts only after both GPUs are idle.
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
mkdir -p outputs/logs

GRID_FLAGS=(
  --device cuda --grid --shard-total 2 --out-dir outputs_shared
  --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic
  --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99
  --kl-warmup-epochs 50
)

echo "== grid: shard A on GPU 1, shard B on GPU 0 =="
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_crl_shared.py "${GRID_FLAGS[@]}" \
  --shard-index 0 --stem crl_shared_grid_a \
  > outputs/logs/pending_grid_a.log 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_crl_shared.py "${GRID_FLAGS[@]}" \
  --shard-index 1 --stem crl_shared_grid_b \
  > outputs/logs/pending_grid_b.log 2>&1 &
PID_B=$!
status=0
wait "$PID_A" || status=1
wait "$PID_B" || status=1
if [ "$status" -ne 0 ]; then
  echo "grid shard failed; see outputs/logs/pending_grid_*.log" >&2
  exit 1
fi

echo "== new representatives: three per GPU =="
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/run_new_representatives.py \
  --device cuda --seeds 10 --epochs 500 \
  --methods CaDRe CNF Becker \
  --out outputs/new_representatives_gpu1.json \
  > outputs/logs/pending_reps_gpu1.log 2>&1 &
REP1=$!
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_new_representatives.py \
  --device cuda --seeds 10 --epochs 500 \
  --methods VGAN-CHG SCARF RCA-VAE \
  --out outputs/new_representatives_gpu0.json \
  > outputs/logs/pending_reps_gpu0.log 2>&1 &
REP0=$!
status=0
wait "$REP1" || status=1
wait "$REP0" || status=1
if [ "$status" -ne 0 ]; then
  echo "new-representative run failed; see outputs/logs/pending_reps_*.log" >&2
  exit 1
fi

"$PY" - <<'PY'
import json
from pathlib import Path

from gcmts.evaluation.report import save_rows

rows = []
for name in ("new_representatives_gpu1.json", "new_representatives_gpu0.json"):
    path = Path("outputs") / name
    if path.exists():
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
if rows:
    save_rows(rows, Path("outputs"), stem="new_representatives")
    print(f"merged {len(rows)} new-representative rows")
PY

echo "== sequential timing pass =="
bash scripts/run_pending_timing.sh "" cuda

echo "== complete =="
date -Is
