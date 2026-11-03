# Complete benchmark run on gpu_212

The run uses the `Benchmark` screen session, `JOBS=2`, `GPU=1`, and the folder
`~/gcmts` on `gpu_212`.

## 0. Access

The local key `~/.ssh/id_ed25519.pub` is currently rejected by `gpu_212`
(`Permission denied (publickey,password)`). Authorise it once from a terminal
where the password is available:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub gpu_212
ssh gpu_212 'hostname; nvidia-smi -L'
```

## 1. Sync and launch

From `gcmts/`:

```bash
bash scripts/sync_and_launch_gpu212.sh
```

The script rsyncs `gcmts/` to `gpu_212:~/gcmts/`, rsyncs the paper tree to
`gpu_212:~/latex/`, moves any previous remote outputs aside, and starts

```bash
screen -L -dmS Benchmark env GPU=1 DEVICE=cuda JOBS=2 \
  bash scripts/run_benchmark_complete.sh
```

If the remote has no Python environment with torch, create it once:

```bash
ssh gpu_212 'cd ~/gcmts && uv sync --extra dev --extra real --extra progress'
```

## 2. What runs, in order

1. `scripts/run_pending_tasks.sh` — grid (two shards), IHDP 100, rendered-image
   Causal3DIdent, analysis and tables. `JOBS=2` batches `(IHDP, image)` and then
   `(grid_a, grid_b)`.
2. `scripts/run_new_representatives.py` — the six new representatives, 10 seeds,
   500 epochs, written to `outputs/new_representatives.json` plus the aggregate
   CSV.
3. `scripts/run_pending_timing.sh` — the sequential timing pass over the 19 CRL
   suites and the static/real panels, then `merge_timings.py`, statistics,
   floors, audit and tables. This stage must not overlap anything.

The run is resumable: relaunching the same script continues from the stored
rows. The completion marker is `outputs/BENCHMARK_DONE`.

## 3. Monitor

```bash
ssh gpu_212 'screen -ls'
ssh gpu_212 -t 'screen -r Benchmark'      # Ctrl-A D to detach
ssh gpu_212 'cd ~/gcmts && tail -f outputs/logs/pending_driver.log'
ssh gpu_212 'nvidia-smi'
```

Per-job logs: `~/gcmts/outputs/logs/pending_{ihdp,image,grid_a,grid_b,timing}.log`.
Screen output: `~/gcmts/screenlog.0`.

## 4. Copy back

Run these from the paper repository root on the local machine. The destinations
are the same relative paths the local tree uses.

```bash
# Outputs, including outputs/new_representatives.json and the aggregate CSV
for d in outputs outputs_shared outputs_timing outputs_shared_timing; do
  rsync -av "gpu_212:~/gcmts/$d/" "gcmts/$d/"
done

# Audit, statistics and reference floors
rsync -av gpu_212:~/gcmts/docs/audit_pending.md \
          gpu_212:~/gcmts/docs/statistical_analysis.md \
          gpu_212:~/gcmts/docs/statistical_analysis.json \
          gpu_212:~/gcmts/docs/reference_floors.md \
          gcmts/docs/

# Tables generated on the GPU host
rsync -av gpu_212:~/latex/pic/experiments/ latex/pic/experiments/

# Logs (optional, for provenance)
rsync -av gpu_212:~/gcmts/outputs/logs/ gcmts/outputs/logs/
```

Then, locally:

```bash
cd gcmts
.venv/bin/python scripts/verify_outputs.py --pools --report docs/audit_pending.md
.venv/bin/python scripts/statds_report.py --outputs outputs --shared outputs_shared \
  --out docs/statistical_analysis.md --json-out docs/statistical_analysis.json
.venv/bin/python scripts/make_paper_tables.py --shared-dir outputs_shared \
  --outputs-dir outputs --latex-dir ../latex
```

The main-text evidence set is fixed in `docs/BENCHMARK_STRUCTURE.md`; the tables
produced by the run feed that structure, and the rest goes to the appendix.
