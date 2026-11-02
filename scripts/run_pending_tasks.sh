#!/usr/bin/env bash
# Run every pending benchmark task that can be executed from a copied tree.
#
# It assumes the tree already contains:
#   - the code at this revision (git pull),
#   - data/ (synthetic pools, ACIC, Twins, News, real cohorts),
#   - outputs/ and outputs_shared/ from the 2026-11 run (the script resumes).
#
# What it runs, in order:
#   1. preflight: dataset manifest + pool finiteness
#   2. T2  IHDP extended to 100 replications (resumes from 30)
#   3. T8  rendered-image Temporal Causal3DIdent (pools + 10 seeds x 5 methods)
#   4. T1  18-problem grid on the fresh pools, two shards
#   5. T7  CPS, only if data/static/cps.csv is present
#   6. T3  sequential timing pass (optional: RUN_TIMING=1)
#   7. T4  statistics, floors, tables, audit
#
# Usage (from the repository root):
#   bash scripts/run_pending_tasks.sh                 # device=mps
#   DEVICE=cuda bash scripts/run_pending_tasks.sh     # on the 2080
#   SKIP_GRID=1 bash scripts/run_pending_tasks.sh     # everything except the grid
#   RUN_TIMING=1 bash scripts/run_pending_tasks.sh    # also time the tables
#   SKIP_POOLS=1 bash scripts/run_pending_tasks.sh    # pools already copied
#
# Every step is resumable; rerunning the script continues where it stopped.

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-mps}"
SKIP_GRID="${SKIP_GRID:-0}"
SKIP_IMAGE="${SKIP_IMAGE:-0}"
RUN_TIMING="${RUN_TIMING:-0}"
LATEX_DIR="${LATEX_DIR:-../latex}"
PY="uv run --no-sync python"

export GCMTS_PREFETCH=1
export GCMTS_LIGHT_BATCH=1

DRY_RUN="${DRY_RUN:-0}"
run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+ %s\n' "$*"
  else
    "$@"
  fi
}

CRL_FLAGS=(
  --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic
  --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99
  --kl-warmup-epochs 50
)

echo "== device=$DEVICE grid=$SKIP_GRID image=$SKIP_IMAGE timing=$RUN_TIMING =="

# ---------------------------------------------------------------- 1. preflight
echo "== 1/7 preflight =="
BASE_SUITES="npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 pcl \
             kuramoto lotka_volterra duffing fitzhugh_nagumo chua van_der_pol \
             pendulum mountain_car varma ou child sde voronoi"
if [ "${SKIP_POOLS:-0}" = "1" ] || \
   { [ -f data/synthetic/npvar/seed_00.pt ] && [ -f data/synthetic/grid_d3_D6_s2_r2/seed_00.pt ]; }; then
  echo "pools already present; skipping generation (set SKIP_POOLS=0 to force a check)"
else
  echo "generating the missing pools (existing finite pools are skipped)"
  run $PY scripts/generate_synthetic_data.py --seeds 10 --suites $BASE_SUITES --with-grid
fi
if [ ! -f data/acic/manifest.json ]; then
  echo "ACIC manifest missing; writing it for this snapshot"
  run $PY scripts/write_acic_manifest.py --data-dir data/acic
else
  digest=$(python3 -c 'import json; print(json.load(open("data/acic/manifest.json"))["digest"][:12])')
  echo "ACIC manifest present (digest $digest)"
fi

# ------------------------------------------------------- 2. IHDP 100 replications
echo "== 2/7 IHDP, 100 replications (resumes from the stored rows) =="
run $PY scripts/run_ctee_static.py --device "$DEVICE" --dataset ihdp \
  --replications 100 --force --out-dir outputs

# ------------------------------------- 3. rendered-image Temporal Causal3DIdent
if [ "$SKIP_IMAGE" = "0" ]; then
  echo "== 3/7 rendered-image Causal3DIdent =="
  run $PY scripts/generate_synthetic_data.py --suites causal3d_img --seeds 10 \
    --train-size 8192 --eval-size 1024
  run $PY scripts/verify_outputs.py --pools-only
  run $PY scripts/run_crl_shared.py --device "$DEVICE" --suite causal3d_img \
    --out-dir outputs_shared "${CRL_FLAGS[@]}"
else
  echo "== 3/7 rendered-image Causal3DIdent skipped (SKIP_IMAGE=1) =="
fi

# ----------------------------------------------------- 4. the 18-problem grid
if [ "$SKIP_GRID" = "0" ]; then
  echo "== 4/7 grid shards (pools generated in preflight) =="
  run $PY scripts/verify_outputs.py --pools-only
  for shard in 0 1; do
    if [ "$shard" = "0" ]; then stem=crl_shared_grid_a; else stem=crl_shared_grid_b; fi
    run $PY scripts/run_crl_shared.py --device "$DEVICE" --grid \
      --shard-index "$shard" --shard-total 2 --stem "$stem" \
      --out-dir outputs_shared "${CRL_FLAGS[@]}"
  done
else
  echo "== 4/7 grid skipped (SKIP_GRID=1) =="
fi

# ------------------------------------------------------------ 5. CPS (manual)
echo "== 5/7 CPS =="
if [ -f data/static/cps.csv ]; then
  run $PY scripts/run_ctee_static.py --device "$DEVICE" --dataset cps \
    --replications 100 --out-dir outputs
else
  echo "data/static/cps.csv absent: CPS remains a manual panel (see docs/OPEN_TASKS.md T7)."
fi

# --------------------------------------------------------- 6. sequential timings
if [ "$RUN_TIMING" = "1" ]; then
  echo "== 6/7 sequential timing pass (this is the long one) =="
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
  echo "== 6/7 timing pass skipped (set RUN_TIMING=1 to run it) =="
fi

# ------------------------------------------------------ 7. analysis and tables
echo "== 7/7 statistics, floors, tables, audit =="
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
echo "Not automatable here (see docs/OPEN_TASKS.md):"
echo "  T7  CPS needs data/static/cps.csv (manual request)."
echo "  T9  II.D representatives, T10 adjacent representatives: new implementations."
echo "  T11 tagged release with DOI."
