#!/usr/bin/env bash
# Rebuild the paper once the timing pass has regenerated the tables.
# Usage: bash scripts/rebuild_paper_after.sh <pid>
set -euo pipefail

cd "$(dirname "$0")/.."

PID="${1:-}"
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi

cd ../latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
echo "paper rebuilt: $(pwd)/main.pdf"
