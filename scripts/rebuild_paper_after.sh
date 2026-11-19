#!/usr/bin/env bash
# Rebuild the paper once the timing pass has regenerated the tables.
# Usage: bash scripts/rebuild_paper_after.sh <pid>
#
# The table generation is a barrier: make_paper_tables.py refuses to write
# unless every declared source run finished cleanly (--require-finished is the
# default). Set TABLE_ALLOW_PARTIAL=1 to write PARTIAL tables from an
# unfinished tree instead (they are stamped as such in the LaTeX header).
set -euo pipefail

cd "$(dirname "$0")/.."

PID="${1:-}"
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi

SHARED_DIR="${SHARED_DIR:-outputs_final_v5}"
GRID_DIR="${GRID_DIR:-outputs_final_grid_v5}"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
LATEX_DIR="${LATEX_DIR:-../latex}"
PY="${PY:-.venv/bin/python}"

TABLE_ARGS=(
  --shared-dir "$SHARED_DIR"
  --grid-dir "$GRID_DIR"
  --outputs-dir "$OUTPUTS_DIR"
  --latex-dir "$LATEX_DIR"
  --allow-mixed
)
if [ "${TABLE_ALLOW_PARTIAL:-0}" = "1" ]; then
  TABLE_ARGS+=(--allow-partial)
fi

# Regenerates the CRL, CTEE, SDD, adjacent, real and visual tables. The visual
# step is wired into this barrier (C17) and is skipped only if its checked-in
# tables are absent or the run is unfinished.
"$PY" scripts/make_paper_tables.py "${TABLE_ARGS[@]}"

cd "$LATEX_DIR"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
echo "paper rebuilt: $(pwd)/main.pdf"
