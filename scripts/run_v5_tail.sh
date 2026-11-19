#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs_final_v5}
LOG=${LOG:-logs/v5.log}
COMMON=(--seeds 5 --epochs-scale 0.01 --batch-size 32 --device cuda \
  --hpo-config configs/crl_hpo_global.json --early-stopping-patience 50 \
  --restore-best-weights --no-baselines --out-dir "$OUT")
run_suite() {
  local suite=$1 cores=$2
  echo "START $suite $(date)" >> "$LOG"
  taskset -c "$cores" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_crl_shared.py \
    --suite "$suite" "${COMMON[@]}" >> "$LOG" 2>&1 || echo "FAILED $suite $(date)" >> "$LOG"
}
run_suite kuramoto 0-1 & A=$!
run_suite lotka_volterra 2-3 & B=$!
wait "$A" "$B"
echo "SHARED_DONE $(date)" >> "$LOG"
taskset -c 0-3 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_crl_benchmarks.py \
  --seeds 10 --device cuda --out-dir outputs_crl_benchmarks_v5 >> "$LOG" 2>&1 || echo "FAILED validation $(date)" >> "$LOG"
echo "VALIDATION_DONE $(date)" >> "$LOG"
taskset -c 0-1 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_crl_shared.py --grid \
  --seeds 5 --epochs-scale 0.01 --batch-size 32 --device cuda \
  --hpo-config configs/crl_hpo_global.json --early-stopping-patience 50 \
  --restore-best-weights --no-baselines --shard-index 0 --shard-total 2 \
  --stem crl_shared_grid_a --out-dir outputs_final_grid_v5 >> "$LOG" 2>&1 & G1=$!
taskset -c 2-3 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_crl_shared.py --grid \
  --seeds 5 --epochs-scale 0.01 --batch-size 32 --device cuda \
  --hpo-config configs/crl_hpo_global.json --early-stopping-patience 50 \
  --restore-best-weights --no-baselines --shard-index 1 --shard-total 2 \
  --stem crl_shared_grid_b --out-dir outputs_final_grid_v5 >> "$LOG" 2>&1 & G2=$!
wait "$G1" "$G2"
echo "FINAL_V5_DONE $(date)" >> "$LOG"
