#!/usr/bin/env bash
# Resume the v5 validation as two disjoint case shards on GPU 1, merge the
# rows, then run the corrected two-shard grid. Existing validation rows are
# preserved in the shard that owns their cases.
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-logs/v5.log}
BASE=outputs_crl_benchmarks_v5
A=outputs_crl_validation_a
B=outputs_crl_validation_b
mkdir -p "$A" "$B"

.venv/bin/python - "$BASE" "$A" "$B" <<'PY'
import json, sys
from pathlib import Path
base, a, b = map(Path, sys.argv[1:])
rows = json.loads((base / "crl_benchmarks.json").read_text())
a_cases = {
    "ivae_linear", "ivae_nonlinear", "leap_temporal", "slow_flows",
    "citris_interventional", "icitris_instantaneous", "child_hierarchical",
}
b_cases = {
    "tdrl_modular", "nctrl_regimes", "mosaic_modules", "cegen_sde",
}
for d, cases in ((a, a_cases), (b, b_cases)):
    subset = [r for r in rows if r.get("case") in cases]
    (d / "crl_benchmarks.json").write_text(json.dumps(subset, indent=2))
    print(d, len(subset))
PY

run_shard() {
  local out=$1 cores=$2; shift 2
  echo "START validation shard $out $(date)" >> "$LOG"
  taskset -c "$cores" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_crl_benchmarks.py \
    --seeds 10 --device cuda --out-dir "$out" --force --only "$@" >> "$LOG" 2>&1 || \
    echo "FAILED validation shard $out $(date)" >> "$LOG"
  echo "DONE validation shard $out $(date)" >> "$LOG"
}

run_shard "$A" 0-1 ivae_linear ivae_nonlinear leap_temporal slow_flows \
  citris_interventional icitris_instantaneous child_hierarchical & PA=$!
run_shard "$B" 2-3 tdrl_modular nctrl_regimes mosaic_modules cegen_sde & PB=$!
wait "$PA" "$PB"

# Merge shards into the canonical validation file, preserving one row per
# (case, seed). The canonical directory still contains the original seed-0 rows.
.venv/bin/python - "$A" "$B" "$BASE" <<'PY'
import json, sys
from pathlib import Path
a, b, base = map(Path, sys.argv[1:])
rows = json.loads((base / "crl_benchmarks.json").read_text())
for d in (a, b):
    p = d / "crl_benchmarks.json"
    if p.exists():
        rows.extend(json.loads(p.read_text()))
by_key = {}
for row in rows:
    by_key[(row.get("case"), row.get("seed"))] = row
merged = [by_key[k] for k in sorted(by_key, key=lambda k: (str(k[0]), int(k[1])))]
(base / "crl_benchmarks.json").write_text(json.dumps(merged, indent=2))
print("merged validation rows", len(merged))
PY
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
