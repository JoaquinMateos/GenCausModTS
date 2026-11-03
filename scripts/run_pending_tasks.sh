#!/usr/bin/env bash
# Run every pending benchmark task that can be executed from a copied tree.
#
# It assumes the tree already contains the code at this revision and data/
# (ACIC, static, real, semi_synthetic); the synthetic pools are regenerated if
# missing. outputs/ and outputs_shared/ are resumed.
#
# Blocks (independent; run concurrently up to JOBS):
#   ihdp      IHDP extended to 100 replications
#   image     rendered-image Temporal Causal3DIdent (10 seeds x 5 methods)
#   grid_a    18-problem grid, shard 0
#   grid_b    18-problem grid, shard 1
#
# Usage (from the repository root):
#   GPU=1 DEVICE=cuda JOBS=4 bash scripts/run_pending_tasks.sh
#   JOBS=2 bash scripts/run_pending_tasks.sh
#   DEVICE=cuda bash scripts/run_pending_tasks.sh          # sequential
#   SKIP_GRID=1 bash scripts/run_pending_tasks.sh
#   SKIP_IMAGE=1 bash scripts/run_pending_tasks.sh
#   RUN_TIMING=1 JOBS=1 bash scripts/run_pending_tasks.sh  # timing must be sequential
#   SKIP_POOLS=1 bash scripts/run_pending_tasks.sh         # pools already copied
#   DRY_RUN=1 bash scripts/run_pending_tasks.sh            # print commands only
#
# JOBS>1 shares one GPU. Check memory first (nvidia-smi); the image suite is the
# heaviest process. If the GPU has little free memory, run with JOBS=2.
#
# Every step is resumable; rerunning the script continues where it stopped.

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-mps}"
GPU="${GPU:-}"
JOBS="${JOBS:-1}"
IMAGE_BATCH="${IMAGE_BATCH:-64}"
SKIP_GRID="${SKIP_GRID:-0}"
SKIP_IMAGE="${SKIP_IMAGE:-0}"
RUN_TIMING="${RUN_TIMING:-0}"
LATEX_DIR="${LATEX_DIR:-../latex}"
DRY_RUN="${DRY_RUN:-0}"

case "$DEVICE" in
  *:*) if [ -n "$GPU" ]; then
         echo "Use GPU=<n> or DEVICE=cuda:<n>, not both." >&2
         exit 2
       fi ;;
esac
if [ -n "$GPU" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

export GCMTS_PREFETCH=1
export GCMTS_LIGHT_BATCH=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+ %s\n' "$*"
  else
    "$@"
  fi
}

# ------------------------------------------------------------------ interpreter
if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  PY=".venv/bin/python"
elif uv run --no-sync python -c "import torch" >/dev/null 2>&1; then
  PY="uv run --no-sync python"
elif python -c "import torch" >/dev/null 2>&1; then
  PY="python"
else
  echo "No Python interpreter with torch found (.venv, uv run, or python)." >&2
  echo "Run 'uv sync --extra dev --extra real --extra progress' first." >&2
  exit 1
fi

CRL_FLAGS=(
  --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic
  --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99
  --kl-warmup-epochs 50
)
CTEE_FLAGS=(--device "$DEVICE" --out-dir outputs)

BASE_SUITES="npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 pcl \
             kuramoto lotka_volterra duffing fitzhugh_nagumo chua van_der_pol \
             pendulum mountain_car varma ou child sde voronoi"

mkdir -p outputs/logs
echo "== device=$DEVICE gpu=${GPU:-default} jobs=$JOBS image_batch=$IMAGE_BATCH grid=$SKIP_GRID image=$SKIP_IMAGE timing=$RUN_TIMING =="
echo "interpreter: $PY"

# ------------------------------------------------------------------- preflight
echo "== preflight =="
if [ "${SKIP_POOLS:-0}" = "1" ] || \
   { [ -f data/synthetic/npvar/seed_00.pt ] && [ -f data/synthetic/grid_d3_D6_s2_r2/seed_00.pt ]; }; then
  echo "pools already present; skipping generation (set SKIP_POOLS=0 to force a check)"
else
  echo "generating the missing base and grid pools"
  run $PY scripts/generate_synthetic_data.py --seeds 10 --suites $BASE_SUITES --with-grid
fi

if [ ! -f data/synthetic/causal3d_img/seed_00.pt ]; then
  echo "generating the rendered-image pools"
  run $PY scripts/generate_synthetic_data.py --suites causal3d_img --seeds 10 \
    --train-size 8192 --eval-size 1024
fi

if [ ! -f data/acic/manifest.json ]; then
  echo "ACIC manifest missing; writing it for this snapshot"
  run $PY scripts/write_acic_manifest.py --data-dir data/acic
else
  digest=$(python3 -c 'import json; print(json.load(open("data/acic/manifest.json"))["digest"][:12])')
  echo "ACIC manifest present (digest $digest)"
fi

run $PY scripts/verify_outputs.py --pools-only

case "$DEVICE" in
  cuda*)
    run $PY -c "import os, torch; print('CUDA visible:', os.environ.get('CUDA_VISIBLE_DEVICES', 'all'), '->', torch.cuda.get_device_name(0), 'count', torch.cuda.device_count())" || \
      echo "warning: could not query the CUDA device (the runs will still try)"
    ;;
esac

# ------------------------------------------------------------------ job bodies
job_cmd() {
  case "$1" in
    ihdp)
      run $PY scripts/run_ctee_static.py "${CTEE_FLAGS[@]}" --dataset ihdp \
        --replications 100 --force
      ;;
    image)
      run $PY scripts/run_crl_shared.py --device "$DEVICE" --suite causal3d_img \
        --out-dir outputs_shared --batch-size "$IMAGE_BATCH" "${CRL_FLAGS[@]}"
      ;;
    grid_a)
      run $PY scripts/run_crl_shared.py --device "$DEVICE" --grid \
        --shard-index 0 --shard-total 2 --stem crl_shared_grid_a \
        --out-dir outputs_shared "${CRL_FLAGS[@]}"
      ;;
    grid_b)
      run $PY scripts/run_crl_shared.py --device "$DEVICE" --grid \
        --shard-index 1 --shard-total 2 --stem crl_shared_grid_b \
        --out-dir outputs_shared "${CRL_FLAGS[@]}"
      ;;
    *) echo "unknown job $1" >&2; return 2 ;;
  esac
}

JOB_PIDS=()
JOB_NAMES=()

start_job() {
  local name="$1"
  if [ "$DRY_RUN" = "1" ]; then
    echo "--- $name"
    job_cmd "$name"
    return 0
  fi
  ( job_cmd "$name" ) > "outputs/logs/pending_${name}.log" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("$name")
  echo "started $name -> outputs/logs/pending_${name}.log (pid $!)"
}

wait_jobs() {
  local failed=()
  for index in "${!JOB_PIDS[@]}"; do
    if ! wait "${JOB_PIDS[$index]}"; then
      failed+=("${JOB_NAMES[$index]}")
    fi
  done
  JOB_PIDS=()
  JOB_NAMES=()
  if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED jobs: ${failed[*]}" >&2
    for name in "${failed[@]}"; do
      echo "--- tail of outputs/logs/pending_${name}.log" >&2
      tail -20 "outputs/logs/pending_${name}.log" >&2 || true
    done
    return 1
  fi
  return 0
}

# ------------------------------------------------------------- run the blocks
JOB_LIST=(ihdp)
[ "$SKIP_IMAGE" = "0" ] && JOB_LIST+=(image)
if [ "$SKIP_GRID" = "0" ]; then
  JOB_LIST+=(grid_a grid_b)
fi

echo "== blocks: ${JOB_LIST[*]} (concurrency $JOBS) =="
offset=0
while [ "$offset" -lt "${#JOB_LIST[@]}" ]; do
  batch=("${JOB_LIST[@]:offset:JOBS}")
  for name in "${batch[@]}"; do
    start_job "$name"
  done
  wait_jobs || exit 1
  offset=$((offset + JOBS))
done

# --------------------------------------------------------------- CPS (manual)
echo "== CPS =="
if [ -f data/static/cps.csv ]; then
  run $PY scripts/run_ctee_static.py "${CTEE_FLAGS[@]}" --dataset cps \
    --replications 100
else
  echo "data/static/cps.csv absent: CPS remains a manual panel (docs/OPEN_TASKS.md T7)."
fi

# --------------------------------------------------------- sequential timing
if [ "$RUN_TIMING" = "1" ]; then
  echo "== sequential timing pass =="
  unset GCMTS_TIMING
  for suite in npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 \
               pcl kuramoto lotka_volterra duffing fitzhugh_nagumo chua \
               van_der_pol pendulum mountain_car varma ou; do
    run $PY scripts/run_crl_shared.py --device "$DEVICE" --suite "$suite" \
      --out-dir outputs_shared_timing "${CRL_FLAGS[@]}"
  done
  for dataset in ihdp acic twins news; do
    reps=100; [ "$dataset" = "news" ] && reps=50
    run $PY scripts/run_ctee_static.py --device "$DEVICE" --dataset "$dataset" \
      --replications "$reps" --out-dir outputs_timing
  done
  for dataset in lalonde_rct jobs_rct nhefs oregon_rct; do
    run $PY scripts/run_ctee_static.py --device "$DEVICE" --dataset "$dataset" \
      --replications 1 --out-dir outputs_timing
  done
  run $PY scripts/merge_timings.py --main outputs outputs_shared \
    --timing outputs_timing outputs_shared_timing
else
  echo "timing pass skipped (set RUN_TIMING=1 to run it; keep JOBS=1)."
fi

# ------------------------------------------------------ analysis and tables
echo "== statistics, floors, tables, audit =="
run $PY scripts/statds_report.py --outputs outputs --shared outputs_shared \
  --out docs/statistical_analysis.md --json-out docs/statistical_analysis.json
run $PY scripts/reference_floors.py --outputs outputs --out docs/reference_floors.md
run $PY scripts/verify_outputs.py --pools --report docs/audit_pending.md || true
run $PY scripts/make_paper_tables.py --shared-dir outputs_shared \
  --outputs-dir outputs --latex-dir "$LATEX_DIR"

echo
echo "Done."
echo "  audit:   docs/audit_pending.md (must report FAIL=0)"
echo "  stats:   docs/statistical_analysis.md"
echo "  tables:  $LATEX_DIR/pic/experiments/*.tex"
echo
echo "Not automatable here (docs/OPEN_TASKS.md):"
echo "  T7  CPS needs data/static/cps.csv (manual request)."
echo "  T9--T14 second representatives per branch: new implementations (docs/OPEN_TASKS.md)."
echo "  T17 tagged release with DOI."
