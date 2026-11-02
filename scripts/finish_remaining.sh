#!/usr/bin/env bash
# Finish the remaining benchmark blocks after the 2026-10 audit.
#
# What it does, in order:
#   1. archives the CRL stems produced before the per-method sampler rewind
#      (A.9 Lotka-Volterra is already post-rewind and is kept);
#   2. archives the ACIC/Twins/News static stems produced before the outcome
#      masking fix (IHDP and the real cohorts are already masked and final);
#   3. reruns ACIC (100 replications), Twins (100) and News (50) with masking;
#   4. reruns the seventeen shared CRL suites with the sampler rewind;
#   5. optionally regenerates the grid pools and reruns the 18-problem grid;
#   6. audits the outputs and regenerates the tables.
#
# Usage (from gcmts/):
#   bash scripts/finish_remaining.sh              # mps, no grid
#   bash scripts/finish_remaining.sh cuda         # on the 2080
#   bash scripts/finish_remaining.sh mps grid     # include the grid (very long on mps)
#
# The script is resumable: every runner skips rows already present, so an
# interrupted run can be restarted with the same command.

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${1:-mps}"
WANT_GRID="${2:-nogrid}"
export GCMTS_PREFETCH=1
export GCMTS_LIGHT_BATCH=1

STAMP="$(date +%Y%m%d_%H%M%S)"
SHARED_ARCHIVE="outputs_shared/rerun_${STAMP}"
STATIC_ARCHIVE="outputs/rerun_${STAMP}"
CRL_MARKER="outputs_shared/.post_audit_rerun"
STATIC_MARKER="outputs/.post_audit_rerun"

CRL_SUITES=(
  npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 pcl kuramoto
  duffing fitzhugh_nagumo chua van_der_pol pendulum mountain_car varma ou
)

CRL_FLAGS=(
  --device "$DEVICE" --seeds 10 --epochs-scale 1.0 --data-source pool
  --dataset-dir data/synthetic --scheduler adaptive --early-stopping-patience 300
  --ema-decay 0.99 --kl-warmup-epochs 50 --out-dir outputs_shared
)

echo "== device=$DEVICE grid=$WANT_GRID stamp=$STAMP =="

# ---------------------------------------------------------------- archiving
if [ ! -e "$CRL_MARKER" ]; then
  mkdir -p "$SHARED_ARCHIVE"
  shopt -s nullglob
  for suite in "${CRL_SUITES[@]}"; do
    for path in "outputs_shared/crl_shared_${suite}" "outputs_shared/crl_shared_${suite}".json \
                "outputs_shared/crl_shared_${suite}"_config.json \
                "outputs_shared/crl_shared_${suite}"_progress.json \
                "outputs_shared/crl_shared_${suite}"_stats.json \
                "outputs_shared/crl_shared_${suite}"_agg.csv \
                "outputs_shared/shared_${suite}"_*; do
      [ -e "$path" ] || [ -d "$path" ] || continue
      mv "$path" "$SHARED_ARCHIVE"/ || true
    done
  done
  touch "$CRL_MARKER"
  echo "archived pre-rewind CRL stems -> $SHARED_ARCHIVE"
else
  echo "CRL stems already archived ($CRL_MARKER); resuming in place"
fi

if [ ! -e "$STATIC_MARKER" ]; then
  mkdir -p "$STATIC_ARCHIVE"
  shopt -s nullglob
  for dataset in acic twins news; do
    for path in "outputs/ctee_static_${dataset}"*; do
      [ -e "$path" ] || continue
      mv "$path" "$STATIC_ARCHIVE"/ || true
    done
  done
  touch "$STATIC_MARKER"
  echo "archived pre-mask static stems -> $STATIC_ARCHIVE"
else
  echo "static stems already archived ($STATIC_MARKER); resuming in place"
fi

# ------------------------------------------------------- masked static panels
echo "== masked static panels =="
uv run --no-sync python scripts/run_ctee_static.py --device "$DEVICE" \
  --dataset acic --replications 100 --out-dir outputs
uv run --no-sync python scripts/run_ctee_static.py --device "$DEVICE" \
  --dataset twins --replications 100 --out-dir outputs
uv run --no-sync python scripts/run_ctee_static.py --device "$DEVICE" \
  --dataset news --replications 50 --out-dir outputs

# ----------------------------------------------------------- CRL with rewind
echo "== shared CRL suites with sampler rewind =="
for suite in "${CRL_SUITES[@]}"; do
  echo "-- $suite"
  uv run --no-sync python scripts/run_crl_shared.py --suite "$suite" "${CRL_FLAGS[@]}"
done

# --------------------------------------------- single-method blocks (child, sde, voronoi)
# CEGEN (sde) reproducibly aborts the Metal driver on MPS, so it runs on CPU.
echo "== single-method blocks =="
for suite in child voronoi; do
  uv run --no-sync python scripts/run_crl_shared.py --suite "$suite" "${CRL_FLAGS[@]}"
done
uv run --no-sync python scripts/run_crl_shared.py --suite sde \
  --device cpu --seeds 10 --epochs-scale 1.0 --data-source pool \
  --dataset-dir data/synthetic --scheduler adaptive --early-stopping-patience 300 \
  --ema-decay 0.99 --kl-warmup-epochs 50 --out-dir outputs_shared

# ------------------------------------------------------------------- the grid
if [ "$WANT_GRID" = "grid" ]; then
  echo "== grid: pools =="
  uv run --no-sync python scripts/generate_synthetic_data.py --with-grid --seeds 10
  uv run --no-sync python scripts/verify_outputs.py --pools-only

  echo "== grid: two shards =="
  uv run --no-sync python scripts/run_crl_shared.py --grid --shard-index 0 \
    --shard-total 2 --stem crl_shared_grid_a "${CRL_FLAGS[@]}"
  uv run --no-sync python scripts/run_crl_shared.py --grid --shard-index 1 \
    --shard-total 2 --stem crl_shared_grid_b "${CRL_FLAGS[@]}"
fi

# ------------------------------------------------------------ audit + tables
echo "== audit and tables =="
uv run --no-sync python scripts/verify_outputs.py --pools \
  --report "docs/audit_${STAMP}.md"
uv run --no-sync python scripts/statds_report.py --outputs outputs \
  --shared outputs_shared --out docs/statistical_analysis.md \
  --json-out docs/statistical_analysis.json
uv run --no-sync python scripts/reference_floors.py --outputs outputs \
  > docs/reference_floors.md
uv run --no-sync python scripts/make_paper_tables.py --shared-dir outputs_shared \
  --outputs-dir outputs --latex-dir ../latex

echo
echo "Done. Audit: docs/audit_${STAMP}.md (must report FAIL=0)."
echo "Then rebuild the paper:  cd ../latex && latexmk -pdf main.tex"
