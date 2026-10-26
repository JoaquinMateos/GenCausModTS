#!/usr/bin/env bash
# End-to-end benchmark run for the GTX 2080 / RTX 2080 Ti host.
#
# The audited protocol changes the generators, metrics and EMA setting, so old
# rows are NOT compatible.  This script archives an existing outputs tree and
# starts fresh; it never appends to a pre-audit run.
#
# Usage (from gcmts/):
#   bash scripts/run_2080.sh                 # full run, 10 seeds, 100 replications
#   JOBS=4 bash scripts/run_2080.sh          # lower CPU concurrency
#   SEEDS=5 REPLICATIONS=30 bash scripts/run_2080.sh
#   SKIP_DOWNLOAD=1 bash scripts/run_2080.sh
#
# Requirements: a CUDA GPU visible to torch, `uv`, and ~10 GB free for the
# canonical static data + ACIC mirror.

set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-10}"
REPLICATIONS="${REPLICATIONS:-100}"
JOBS="${JOBS:-8}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "== 1/6 environment =="
uv sync --extra dev --extra real
uv run --no-sync python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "== 2/6 archive old outputs (never resume across the protocol change) =="
for directory in outputs outputs_shared; do
  if [ -d "$directory" ] && [ ! -e "${directory}_pre_audit_${STAMP}" ]; then
    mv "$directory" "${directory}_pre_audit_${STAMP}"
    echo "archived $directory -> ${directory}_pre_audit_${STAMP}"
  fi
done
mkdir -p outputs outputs_shared outputs/logs

echo "== 3/6 canonical datasets =="
if [ "$SKIP_DOWNLOAD" = "0" ]; then
  uv run --no-sync python scripts/download_datasets.py
  # ACIC 2018 Zenodo mirror (~1.4 GB archive; 100 replications converted).
  if [ ! -f data/acic/metadata.csv ]; then
    uv run --no-sync python scripts/fetch_acic_zenodo.py --download --n-replications 100
  fi
else
  echo "SKIP_DOWNLOAD=1: using the existing data/ tree"
fi

echo "== 4/6 synthetic pools =="
uv run --no-sync python scripts/generate_synthetic_data.py \
  --seeds $(seq 0 $((SEEDS - 1))) \
  --suites npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 pcl \
           kuramoto lotka_volterra duffing fitzhugh_nagumo chua van_der_pol \
           pendulum mountain_car varma ou voronoi sde child

echo "== 5/6 full benchmark (${SEEDS} seeds, ${REPLICATIONS} replications, ${JOBS} workers) =="
# The orchestrator writes outputs/ and outputs_shared/ and regenerates every
# LaTeX table as a final barrier.  EMA is 0.99 by default and KL warm-up is 50
# epochs; both are uniform across methods.
uv run --no-sync python scripts/run_full_benchmark.py \
  --device cuda --seeds "$SEEDS" --replications "$REPLICATIONS" \
  --jobs "$JOBS" --with-grid

echo "== 6/6 regenerate tables (idempotent; the barrier already did it) =="
uv run --no-sync python scripts/make_paper_tables.py \
  --shared-dir outputs_shared --outputs-dir outputs --latex-dir ../latex

echo
echo "Done. Artifacts: outputs/, outputs_shared/, ../latex/pic/experiments/*.tex"
echo "If you also need wall-clock times, rerun the relevant steps with --jobs 1"
echo "and --keep-timing (the parallel orchestrator suppresses timings)."
