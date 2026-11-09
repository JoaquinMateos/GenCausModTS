#!/usr/bin/env bash
# Final 5-seed benchmark: official video releases first, then the corrected
# NCTRL and the remaining temporal/control suites.  Run from gcmts/.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs_final_v3}
BASE=${BASE:-outputs_final_v2}
LOG=${LOG:-logs/final_v3.log}
SEEDS=${SEEDS:-5}
SCALE=${SCALE:-0.01}
mkdir -p "$OUT" "$(dirname "$LOG")"
.venv/bin/python scripts/prepare_final_outputs.py --source "$BASE" --dest "$OUT" >> "$LOG" 2>&1 || true

COMMON=(--seeds "$SEEDS" --epochs-scale "$SCALE" --batch-size 32 --device cuda \
  --hpo-config configs/crl_hpo_global.json --early-stopping-patience 50 \
  --restore-best-weights --force --out-dir "$OUT")

run_suite () {
  local suite=$1
  echo "START $suite $(date)" >> "$LOG"
  taskset -c 0-3 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    GCMTS_TDRL_PRIOR_CHUNK=8 GCMTS_TDRL_VECTORIZE=1 CUDA_VISIBLE_DEVICES=1 \
    .venv/bin/python scripts/run_crl_shared.py --suite "$suite" --resume "${COMMON[@]}" \
    >> "$LOG" 2>&1 || echo "FAILED $suite $(date)" >> "$LOG"
}

# Official visual block (real Zenodo data; official convolutional CITRIS/iCITRIS).
for suite in pong pinball ball_in_boxes causal3d_img; do
  run_suite "$suite"
done

# Unfinished vector suites from the earlier five-seed pass.
for suite in duffing fitzhugh_nagumo chua van_der_pol pendulum mountain_car varma ou voronoi sde child; do
  run_suite "$suite"
done

# Completed vector suites: only the corrected NCTRL rows are missing, because
# prepare_final_outputs.py removed the stale ones.
for suite in npvar npvar_linear causal3d cartpole npvar_hard lorenz lorenz96 pcl kuramoto lotka_volterra; do
  run_suite "$suite"
done

echo "DONE $(date)" >> "$LOG"
