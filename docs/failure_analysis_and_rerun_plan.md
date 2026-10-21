# Failure analysis and rerun plan

**Run analysed:** the ten-seed full-suite run launched on 2026-10-16 and finished
2026-10-20 (3 d 19 h), from `errors.log`:

```
--device cuda --gpus 1 --jobs 4 --seeds 10 --with-grid --data-source pool
--scheduler adaptive --early-stopping-patience 300 --restore-best-weights
--kl-warmup-epochs 50
```

The environment also set `GCMTS_PREFETCH=1`, `GCMTS_LIGHT_BATCH=1` and
`GCMTS_EMA_DECAY=0.99` (visible in every trainer log line; they are **not** in
the orchestrator command).

**Outcome:** 36 steps, 25 completed, 11 exited non-zero. The completed blocks are
usable and are now reported in the paper (`latex/sections/09_comparative_experiments.tex`).
The 11 failures reduce to **six independent root causes**. None invalidates a
completed result; each is a localised code or data defect.

> **Implementation status.** R1, R2, R3, R4, R6 and R7 are **fixed** in this
> tree and covered by `tests/test_regressions.py`. R5 (Oregon) is a manual data
> download because the Dataverse is behind a WAF challenge; the runner raises an
> actionable message when the archive is missing. **All eleven blocked steps
> have now been rerun successfully:** the six CRL suites (50 rows each), the
> ten-seed sanity block (110 rows), SDD (41 rows), NHEFS (6 rows) and Oregon
> (6 rows). The LaTeX tables have been regenerated. The only pending items are
> the rendered-image Causal3DIdent variant, the ACIC panels and a tagged release.

The `Overall 5248/4693 runs (112%)` line is a dashboard accounting artefact, not
a result: the footer sums the `done` counters of every `*_progress.json`, while
`total` is each step's current planned count. After a failed step is restarted
(or when a previous run's progress file is still present) `done` can exceed
`total`. It does not affect rows or metrics.

---

## 1. Failure map

| # | Failed step(s) | First symptom in the log | Root cause | Class |
|---|---|---|---|---|
| R1 | A.10 Duffing, A.11 FitzHugh–Nagumo, A.13 van der Pol, A.14 pendulum, A.15 mountain-car | `IndexError: index 2 is out of bounds for axis 1 with size 2` in `R2SepMetric` / `DCIDisentanglementMetric` | TDRL allocates `fix+dyn+obs = 1+1+1 = 3` latents for `d=2`; the metrics assume a square `d×d` matrix | code |
| R2 | A.9 Lotka–Volterra | `epoch 1/1500 │ val_loss=nan`; then `ValueError: matrix contains invalid numeric entries` in `linear_sum_assignment` | Euler integration of the competitive Lotka–Volterra drift produces non-finite states; the fixed pool contains NaN and MCC rejects it | code + data |
| R3 | CRL sanity (matching-benchmark) | `RuntimeError: Expected all tensors to be on the same device … cuda:0 and cpu` in `graph_ops.shd` | `aligned_shd` mixes a CUDA estimate (`get_latent_adjacency`, e.g. LEAP) with a CPU ground-truth adjacency | code |
| R4 | SDD invariance | `RuntimeError: … cuda:0 and cpu` in `SYNC.loss` → `mse_loss(outputs["outcome"], target)` | `GCMTS_LIGHT_BATCH=1` copies only `x`/`context`; SYNC's loss also reads `outcome` (and `domain`) | code + env |
| R5 | CTEE real `oregon_rct` | `FileNotFoundError: data/real/dataverse_files/Oregon/oregon_puf.zip` | The Dataverse archive was never downloaded (manual source) | data |
| R6 | CTEE real `nhefs` | `ValueError: Input y contains NaN` in `Ridge.fit` | `wt82_71` has missing entries and `load_nhefs` does not drop them | data |
| R7 | Regenerate LaTeX tables | `FileNotFoundError: ../latex/pic/experiments/crl_summary.tex` | The GPU host has no `latex/` tree; `make_paper_tables.py` did not create the output directory | code (reporting) |

R7 also uncovered four reporting-only defects that are **already fixed** in this
working tree: missing output directory, deterministic baselines treated as
incomplete (which suppressed the whole external-validity table), unescaped
underscores in dataset/metric labels (LaTeX fatal error), and blank lines inside
the external-validity `tabular`.

---

## 2. Root-cause detail and fixes

### R1 — TDRL latent budget for `d = 2`

**Evidence.** `outputs/logs/{09,10,12,13,14}_*.error` all fail in the metric
layer, after iVAE and LEAP have completed. The failures are exactly the five
suites whose `latent_dim` is 2 (`_PHYSICS_DIMS["duffing"|"fitzhugh_nagumo"|
"van_der_pol"] = 2`, `pendulum`, `mountain_car`).

**Cause.** `scripts/run_crl_shared.py::build_methods`:

```python
fix = max(d - 2, 1)      # d=2 -> 1
obs = 1
dyn = max(d - fix - obs, 1)   # d=2 -> 1
# fix + dyn + obs = 3, but the true latent is d = 2
```

`TDRL` sets `latent_dim = fix + dyn + obs` and encodes three components. The
metric helpers `_r2_matrix` / `_importance_matrix` build a matrix with
`k = tgt.shape[-1]` (or `x.shape[1]`) and then index with a Hungarian column that
can equal 2, hence the `size 2` `IndexError`.

**Fix (two layers).**

1. Make the allocator sum to `d`. Suggested helper:

   ```python
   def _tdrl_blocks(d: int) -> tuple[int, int, int]:
       if d >= 3:
           fix = max(d - 2, 1)
           obs = 1
           return fix, d - fix - obs, obs      # fix + dyn + obs == d
       return 1, 1, 0                         # d == 2: fixed + dynamic
   ```

   `TDRL` supports a zero-size observation block (`obs_prior=None`, empty slice
   in `_group_prior`), so `(1, 1, 0)` is safe; `(1, 0, 1)` also works. Do **not**
   keep 3 latents for a 2-factor problem.

2. Harden the metrics so a dimension mismatch can never crash a suite:
   - `_r2_matrix`: build `matrix = np.zeros((tgt_dim, pred_dim))`, loop over both
     dimensions, and only score matched rows (or return `nan` for unmatched).
   - `_importance_matrix`: use `k_pred = x.shape[1]`, `k_true = y.shape[1]`,
     allocate `(k_true, k_pred)`, and loop `for i in range(k_true)`.
   - `R2Sep`: skip `j == col[i]` only when `col[i]` is within range.
   - `mcc`: return `nan` (and log) when either latent contains non-finite values,
     instead of passing a NaN cost matrix to `linear_sum_assignment`.

**Verify.** Re-run one suite and inspect that TDRL logs `(D=4, fix=1, dyn=1,
obs=0)` for Duffing and that the `RESULT` line has finite `r2_sep`/`dci`.

### R2 — Non-finite Lotka–Volterra process

**Evidence.** `outputs/logs/08_*Lotka-Volterra*.log`, line 10 trains iVAE and
line 26 already reports `val_loss=nan`; the run then emits hundreds of
`Non-finite loss … skipping step` warnings, early-stops at epoch 301 with
`best val_loss inf`, and the MCC call fails with
`ValueError: matrix contains invalid numeric entries`.

**Cause.** `src/gcmts/data/physics.py::_lotka_volterra` uses an explicit Euler
step (`dt=0.01`, 8 substeps) on a competitive drift whose interaction matrix has
`-0.85` on the diagonal and `0.15` off-diagonal. For some regime parameters
(`p ∈ [0.6, 1.4]`) and initial states the trajectory leaves the stable region and
overflows; `states` then contains `inf`, the standardisation `(states-mean)/std`
turns it into `NaN`, and the pool is written with NaNs. Every later method then
trains on NaNs.

**Fix.**

1. Sanitise the generator:
   - clip the state to a physical box after each substep (e.g. `state.clamp_(1e-3, 50)`),
     and/or reduce `dt`/increase `substeps` for `lotka_volterra`;
   - after integration, replace non-finite trajectories: resample or drop them
     (`finite = torch.isfinite(states).all(dim=(1, 2)); keep = finite`).
2. Gate the pool writer: in `scripts/generate_synthetic_data.py::generate_one`,
   assert `torch.isfinite(train.x).all()` (and `z`, `eval`) and raise with the
   suite/seed if not, so a bad pool can never be consumed silently.
3. Guard `mcc` as in R1.

**Verify and regenerate.**

```bash
cd gcmts
uv run python scripts/generate_synthetic_data.py \
    --suites lotka_volterra --seeds 10 \
    --train-size 65536 --eval-size 4096 --out-dir data/synthetic
uv run python - <<'PY'
import glob, torch
for path in glob.glob("data/synthetic/lotka_volterra/seed_*.pt"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for split in ("train", "eval"):
        assert torch.isfinite(payload[split]["x"]).all(), (path, split)
print("Lotka-Volterra pools are finite")
PY
```

### R3 — Device mismatch in `shd` / `aligned_shd`

**Evidence.** `outputs/logs/20_CRL_sanity_matching-benchmark.log`, failing after
LEAP training in `graph_ops.aligned_shd → shd`, `cuda:0` vs `cpu`.

**Cause.** `run_crl_benchmarks.py` passes `adj_est` from
`model.get_latent_adjacency()` (some methods return it on CPU) and `adj_true`
from the moved evaluation batch; `aligned_shd` permutes `adj_est` on its own
device and then calls `shd(permuted, adj_true)` without aligning devices.

**Fix.** In `src/gcmts/core/graph_ops.py`, make both functions device-agnostic:

```python
def shd(adj_est, adj_true, *, threshold=1e-3):
    est = _binarise(adj_est, threshold)
    true = _binarise(adj_true.to(est.device), threshold)
    return int((est != true).sum().item())

# aligned_shd: at the top
adj_true = adj_true.to(adj_est.device)
```

(Alternatively move both to CPU inside `shd`.) Add a unit test mixing a CPU and
a CUDA adjacency (skip if CUDA is unavailable).

### R4 — Light-batch drops tensors that a loss reads

**Evidence.** `outputs/logs/27_SDD_invariance.log`, trainer header
`prefetch=True, light_batch=True, ema=0.99`; failure at
`sync.py:203 mse_loss(outputs["outcome"], target)` with `target` on CPU.

**Cause.** `src/gcmts/core/trainer.py`:

```python
TRAIN_BATCH_KEYS = frozenset({"x", "context"})
```

`move_batch(..., keys=TRAIN_BATCH_KEYS)` is applied when `GCMTS_LIGHT_BATCH=1`.
SYNC (and any effect model) also reads `batch.outcome` / `batch.domain`, which
therefore stay on the host.

**Fix (either, preferably both).**

1. Extend the key set to everything a training loss may read:
   ```python
   TRAIN_BATCH_KEYS = frozenset({"x", "context", "outcome", "domain"})
   ```
   (These tensors are small; `z`, `adjacency`, `counterfactual` remain
   evaluation-only.)
2. Disable light batching for the SDD/effect families (their batches are not the
   CRL `x`/`context` contract), e.g. `light_batch=False` in
   `run_sdd_benchmarks.run_case` and `run_ctee_static`.

Until the fix is verified, rerun SDD with `GCMTS_LIGHT_BATCH=0`.

### R5 — Missing Oregon archive

**Evidence.** `FileNotFoundError: data/real/dataverse_files/Oregon/oregon_puf.zip`.

**Fix.** Download the Oregon Health Insurance Experiment public-use files from
`https://dataverse.harvard.edu/dataverse/oregon` (manual; no stable direct URL),
and place the archive at:

```
data/real/dataverse_files/Oregon/oregon_puf.zip
```

The loader `gcmts.data.real_panels.load_oregon` expects
`OHIE_Public_Use_Files/OHIE_Data/oregonhie_descriptive_vars.dta` and
`…/oregonhie_ed_vars.dta` inside the archive. Verify with:

```bash
unzip -l data/real/dataverse_files/Oregon/oregon_puf.zip | grep oregonhie
```

### R6 — NHEFS missing outcome values

**Evidence.** `ValueError: Input y contains NaN` in
`_ridge_t_learner(...).fit(x[t > 0], y[t > 0])`.

**Cause.** `wt82_71` (weight change) has missing entries; `load_nhefs` keeps them.

**Fix.** In `src/gcmts/data/real_panels.py::load_nhefs`, drop incomplete rows
before returning the cohort:

```python
needed = [*covariates, "qsmk", "wt82_71"]
frame = frame.dropna(subset=needed)
x = frame[covariates].to_numpy(dtype=np.float64)
t = frame["qsmk"].to_numpy(dtype=np.int64)
y = frame["wt82_71"].to_numpy(dtype=np.float64)
```

Log the number of dropped rows so the cohort size is auditable. Add a unit test
that a frame with a missing outcome is filtered rather than passed to sklearn.

### R7 — Table generation on a host without `latex/`

**Evidence.** `FileNotFoundError: ../latex/pic/experiments/crl_summary.tex`.

**Cause.** `make_paper_tables.py` wrote into `latex_dir/pic/experiments/` without
creating it; the GPU host only has the `gcmts/` tree.

**Fix (already applied here).** In `main()`:

```python
(args.latex_dir / "pic" / "experiments").mkdir(parents=True, exist_ok=True)
```

The same patch set also:
- treats `Persistence`/`AR(1)` as complete with one deterministic run
  (`_case_complete`), so the external-validity table is no longer suppressed;
- escapes `_` in dataset/metric labels (`r2\_diag`, `beijing\_pm25`, …), which was
  a fatal LaTeX error;
- removes the empty strings that produced blank lines inside the
  `external_validity` `tabular`.

---

## 3. Ordered rerun procedure

### Step 0 — apply and test the fixes

Edit:

- `scripts/run_crl_shared.py` (TDRL allocator);
- `src/gcmts/evaluation/metrics.py` (`_r2_matrix`, `_importance_matrix`, NaN guards);
- `src/gcmts/core/utils.py` (`mcc` NaN guard);
- `src/gcmts/core/graph_ops.py` (`shd`, `aligned_shd` device alignment);
- `src/gcmts/core/trainer.py` (`TRAIN_BATCH_KEYS`);
- `src/gcmts/data/physics.py` + `scripts/generate_synthetic_data.py` (finite LV);
- `src/gcmts/data/real_panels.py` (`load_nhefs` dropna);
- `scripts/make_paper_tables.py` (already patched).

Then:

```bash
cd gcmts
uv run ruff check src scripts tests
uv run pytest -q -m "not slow"
```

### Resume guard: why a naive rerun exits with code 2

Every runner fingerprints its output in `<stem>_config.json`. After a code or
configuration change it **refuses to append** to rows written by the previous
revision (exit 2, `Refusing to resume … stored code=…, current code=…`). This is
what the first rerun hit: the old rows were produced at `d63caa2`, the fixes are
at `a023d5c`, and the explicit `--ema-decay 0.99` also changed the recorded
configuration (the original run supplied it through `GCMTS_EMA_DECAY`).

Two ways out:

- **Clean (recommended):** delete the affected `<stem>.json` and
  `<stem>_config.json` before rerunning, so the new rows never mix with the old
  protocol.
- **Quick:** pass `--force` to the runner (the orchestrator exports
  `GCMTS_FORCE=1` to its workers). This appends and *skips* `(case, seed)` pairs
  already present, so a partial old block is retained.

The commands below delete the stale rows of exactly the eleven blocked blocks.

### Step 1 — regenerate the Lotka–Volterra pools

```bash
cd gcmts
uv run python scripts/generate_synthetic_data.py \
    --suites lotka_volterra --seeds 10 \
    --train-size 65536 --eval-size 4096 --out-dir data/synthetic
```

Run the finiteness check shown in R2 before continuing.

### Step 2 — rerun the six blocked CRL suites

Fast path through the orchestrator (writes `outputs_shared/` and the tables).
The `rm` lines are the clean way to satisfy the resume guard:

```bash
cd gcmts
for suite in lotka_volterra duffing fitzhugh_nagumo van_der_pol pendulum mountain_car; do
  rm -f "outputs_shared/crl_shared_${suite}".json \
        "outputs_shared/crl_shared_${suite}"_config.json
done
uv run python scripts/run_full_benchmark.py \
    --device cuda --gpus 1 --jobs 4 --seeds 10 --force \
    --data-source pool --scheduler adaptive --early-stopping-patience 300 \
    --restore-best-weights --kl-warmup-epochs 50 --ema-decay 0.99 \
    --suites lotka_volterra duffing fitzhugh_nagumo van_der_pol pendulum mountain_car
```

Per-suite equivalent, useful when a single suite still fails:

```bash
for suite in lotka_volterra duffing fitzhugh_nagumo van_der_pol pendulum mountain_car; do
  rm -f "outputs_shared/crl_shared_${suite}".json \
        "outputs_shared/crl_shared_${suite}"_config.json
  uv run python scripts/run_crl_shared.py \
      --device cuda --suite "$suite" --seeds 10 --epochs-scale 1.0 --force \
      --data-source pool --dataset-dir data/synthetic \
      --scheduler adaptive --early-stopping-patience 300 \
      --restore-best-weights --kl-warmup-epochs 50 --ema-decay 0.99 \
      --out-dir outputs_shared
done
```

### Step 3 — rerun the CRL sanity block

```bash
uv run python scripts/run_crl_benchmarks.py --device cuda --seeds 10 --out-dir outputs --force
```

### Step 4 — rerun SDD (with the light-batch guard until verified)

```bash
GCMTS_LIGHT_BATCH=0 uv run python scripts/run_sdd_benchmarks.py \
    --device cuda --seeds 10 --out-dir outputs --force
```

### Step 5 — real effect cohorts

```bash
# after placing oregon_puf.zip under data/real/dataverse_files/Oregon/
uv run python scripts/run_ctee_static.py --device cuda --dataset oregon_rct --replications 1 --out-dir outputs --force
uv run python scripts/run_ctee_static.py --device cuda --dataset nhefs      --replications 1 --out-dir outputs --force
```

### Step 6 — regenerate the LaTeX tables

```bash
uv run python scripts/make_paper_tables.py \
    --shared-dir outputs_shared --outputs-dir outputs --latex-dir ../latex
cd ../latex && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Step 7 — verify completeness

```bash
# no step should still be missing
grep -L "RESULT" outputs/logs/*.log || true

# each CRL suite should have 10 seeds x every method
uv run python - <<'PY'
import glob, json, os
for path in sorted(glob.glob("outputs_shared/crl_shared_*.json")):
    if any(tag in path for tag in ("agg", "config", "stats", "progress")):
        continue
    rows = json.load(open(path))
    seeds = {r.get("seed") for r in rows}
    methods = sorted({r.get("method") for r in rows})
    ok = len(seeds) == 10
    print(f"{'OK ' if ok else 'BAD'} {os.path.basename(path):40s} seeds={len(seeds)} methods={methods}")
PY

# finite metrics only
uv run python - <<'PY'
import glob, json, math
bad = []
for path in glob.glob("outputs_shared/crl_shared_*.json"):
    if any(t in path for t in ("agg", "config", "stats", "progress")):
        continue
    for row in json.load(open(path)):
        for key in ("mcc", "r2_diag", "r2_sep", "forecast1", "forecast4"):
            value = row.get(key)
            if value is not None and not math.isfinite(value):
                bad.append((path, key, row.get("method"), row.get("seed")))
print("non-finite metric rows:", len(bad))
for item in bad[:20]:
    print(" ", item)
PY
```

Expected final state: every CRL shared suite with `seeds=10`, SDD with 40 runs,
sanity with its default replications, and the two real cohorts with one row per
method. Then regenerate the tables and rebuild the PDF.

---

## 4. What the completed results already say

The completed blocks are not a stopgap: they already support the paper's central
claims and are now written into Section 9.

- **CRL, complete suites (11).** Mean MCC: NCTRL 0.716, LEAP 0.697, MOSAIC 0.682,
  TDRL 0.659, iVAE 0.475. The ordering is mechanism-dependent: all methods tie on
  NP/VAR (0.86), while NCTRL wins on Lorenz-63/96 and Chua, LEAP on PCL and
  Kuramoto, MOSAIC on Causal3DIdent and VARMA, and iVAE trails on nonlinear
  processes but is competitive on OU.
- **CRL, 18-problem grid.** MCC is *not* significantly different (Friedman
  χ²=7.16, p=0.128; CD=1.44). One-step forecasting is almost perfectly separated
  (χ²=72, p=8.6×10⁻¹⁵; ranks LEAP 1, NCTRL 2, TDRL 3, iVAE 4, MOSAIC 5), with
  grid-mean MSE 4.52/4.62/5.64/9.11/9.20. A representation metric that is nearly
  tied can hide a two-fold difference in downstream utility.
- **CTEE temporal.** CRN dominates the harmonic benchmark (PEHE 0.36, ATE 0.17);
  on tumour growth all methods have PEHE ≈ 6.8–6.9 (positivity violation), but
  trajectory metrics disagree; the linear-SEM panel is unstable at every seed
  (PEHE 10.5–11.6 with sd 20–22).
- **CTEE static.** A ridge T-learner is best or near-best on all five panels;
  CEPAE is the best learned method; CRN collapses to a near-zero effect
  (PEHE ≈ 1, ATE ≈ 1) on every static panel. The LaLonde cohort confirms the
  scale problem: ATE errors 1653–1796 for the neural estimators versus 213
  (ridge) and 528 (nearest neighbour).
- **External validity.** Persistence/AR(1) beat every representation learner at
  one step on 10 of 11 panels; the exception is traffic (LEAP 0.312 versus 0.494).
  This is the review's utility-vs-identifiability warning in its sharpest form.
- **Adjacent.** GBTPL path-F1 0.79 (source accuracy 0.17), LacaDM hypervolume
  8.43±8.96, CausalConceptTS faithfulness 0.04±0.03, CausalGAN CF MAE 0.41.

The six missing physics/control blocks are exactly the ones that would test
whether the mechanism-dependent ordering above survives two-dimensional latent
spaces; until R1 and R2 are fixed they must not be reported as low scores.
