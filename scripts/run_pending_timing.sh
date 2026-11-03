#!/usr/bin/env bash
# Sequential timing pass (T3), queued after the main pending driver.
#
# Usage: bash scripts/run_pending_timing.sh <pid-of-main-driver> <device>
#
# Waits for the main driver to exit, then times every block that appears in the
# tables on an otherwise idle machine. Writes to outputs_timing/ and
# outputs_shared_timing/ and merges only the `seconds` field into the main rows.
set -euo pipefail

cd "$(dirname "$0")/.."

MAIN_PID="${1:-}"
DEVICE="${2:-mps}"

if [ -n "$MAIN_PID" ]; then
  echo "waiting for main driver pid $MAIN_PID"
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 60
  done
  echo "main driver finished; starting the timing pass"
fi

if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  PY=".venv/bin/python"
else
  PY="uv run --no-sync python"
fi

export GCMTS_PREFETCH=1
export GCMTS_LIGHT_BATCH=1
unset GCMTS_TIMING

CRL_FLAGS=(
  --device "$DEVICE" --seeds 10 --epochs-scale 1.0 --data-source pool
  --dataset-dir data/synthetic --scheduler adaptive --early-stopping-patience 300
  --ema-decay 0.99 --kl-warmup-epochs 50
)

echo "== CRL timing (sequential) =="
for suite in npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 \
             pcl kuramoto lotka_volterra duffing fitzhugh_nagumo chua \
             van_der_pol pendulum mountain_car varma ou; do
  echo "-- $suite"
  $PY scripts/run_crl_shared.py "${CRL_FLAGS[@]}" --suite "$suite" \
    --out-dir outputs_shared_timing
done

echo "== static timing (sequential) =="
for dataset in ihdp acic twins news; do
  reps=100
  [ "$dataset" = "news" ] && reps=50
  echo "-- $dataset ($reps)"
  $PY scripts/run_ctee_static.py --device "$DEVICE" --dataset "$dataset" \
    --replications "$reps" --out-dir outputs_timing
done
for dataset in lalonde_rct jobs_rct nhefs oregon_rct; do
  echo "-- $dataset (1)"
  $PY scripts/run_ctee_static.py --device "$DEVICE" --dataset "$dataset" \
    --replications 1 --out-dir outputs_timing
done

echo "== merge, statistics, tables, audit =="
$PY scripts/merge_timings.py --main outputs outputs_shared \
  --timing outputs_timing outputs_shared_timing
$PY scripts/statds_report.py --outputs outputs --shared outputs_shared \
  --out docs/statistical_analysis.md --json-out docs/statistical_analysis.json
$PY scripts/verify_outputs.py --pools --report docs/audit_pending.md || true
$PY scripts/make_paper_tables.py --shared-dir outputs_shared \
  --outputs-dir outputs --latex-dir ../latex

echo "timing pass done."
