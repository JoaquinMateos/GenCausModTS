# Open tasks and plans

Status snapshot: 2026-11-02, library `62bc6cb`, paper `58a95c9`.

The corrected benchmark is complete except for the items below. Every task has a
concrete command, an acceptance test, and a cost estimate measured on the two
hosts used so far (Apple M4 / MPS and RTX 2080 Ti / CUDA).

Reference state:

- Shared CRL, 18 suites, fresh pools + per-method sampler rewind: NCTRL 0.782,
  LEAP 0.761, TDRL 0.741, MOSAIC 0.730, iVAE 0.581 (Friedman χ²=42.09,
  p=1.6×10⁻⁸).
- Static CATE, masked: IHDP 30 replications, ACIC 100, Twins 100, News 50.
- Single-method blocks: CHiLD 0.357, CITRIS 0.952, iCITRIS 0.851, CEGEN
  one-step MSE 1.136 (CPU).
- Audit `docs/audit_2026-11.md`: FAIL=0, one REVIEW (real-panel rows have no
  seed field).

---

## P0 — required before the paper's results are final

### T1. Rerun the 18-problem grid on fresh pools with the rewind

**Why.** The grid rows (MCC tie p=0.285; one-step χ²=72.0) predate the pool
regeneration and the sampler rewind, and the paper still marks them as
provisional. A.1 moved by 0.14 MCC on fresh pools, so the grid may move too; the
grid supports the paper's central "representation metrics tie, utility separates"
contrast.

**Plan.** Pools are already generated locally (18 configs × 10 seeds). Gate them,
then run the two standard shards on the GPU host — on M4 the grid costs 1–2
weeks.

```bash
cd ~/gcmts && git pull
export GCMTS_PREFETCH=1 GCMTS_LIGHT_BATCH=1

uv run --no-sync python scripts/generate_synthetic_data.py --with-grid --seeds 10
uv run --no-sync python scripts/verify_outputs.py --pools-only   # must be FAIL=0

for shard in 0 1; do
  stem=$([ "$shard" = "0" ] && echo crl_shared_grid_a || echo crl_shared_grid_b)
  uv run --no-sync python scripts/run_crl_shared.py --device cuda --grid \
    --shard-index "$shard" --shard-total 2 --stem "$stem" \
    --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic \
    --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99 \
    --kl-warmup-epochs 50 --out-dir outputs_shared
done
```

**Acceptance.** `crl_shared_grid_a/b.json` have 450 rows each (10 seeds × 9
configs × 5 methods), 0 errors, all finite; `statds_report.py` regenerates both
grid blocks; the paper's grid paragraph is rewritten from the new values and its
red caveat is removed.

**Cost.** ≈3–4 days on the 2080 (the earlier grid took 78 h per shard with the
shards in parallel); 1–2 weeks on M4. **Disk:** ~28 GB of pools.

**Risk.** Grid models are larger (d≤8, D≤24); watch memory at `--jobs` > 2.

### T2. Extend IHDP to 100 replications

**Why.** The paper's protocol says 100 replications, ACIC/Twins use 100 and News
50, but the masked IHDP rerun has 30. A 100-replication table removes the last
asymmetry among the static panels.

**Plan.** The runner resumes from the 30 stored replications.

```bash
uv run --no-sync python scripts/run_ctee_static.py --device cuda \
  --dataset ihdp --replications 100 --force --out-dir outputs
uv run --no-sync python scripts/statds_report.py --outputs outputs \
  --shared outputs_shared --out docs/statistical_analysis.md
uv run --no-sync python scripts/make_paper_tables.py --shared-dir outputs_shared \
  --outputs-dir outputs --latex-dir ../latex
```

**Acceptance.** 600 rows, config `replications=100`, static-table caption updates
automatically, IHDP block and paper paragraph regenerate; `verify_outputs`
FAIL=0.

**Cost.** ≈8 h on the 2080, ≈8–13 h on M4 (measured 4.3 min per replication).

### T3. Wall-clock times for the Time columns

**Why.** `seconds` is NaN in every table because the orchestrator sets
`GCMTS_TIMING=0` for parallel runs. The paper's Time columns are `--`.

**Plan.** Time only the blocks that appear in the tables, on an idle machine,
sequentially, into a separate tree; merge only the `seconds` field. The
orchestrator's `--jobs 1 --keep-timing` exists but writes into the main tree, so
the per-block route below is preferred.

```bash
unset GCMTS_TIMING                      # default is 1; just do not set it to 0

# CRL: one call per suite, into the timing tree
for s in npvar npvar_linear npvar_hard cartpole causal3d lorenz lorenz96 pcl \
         kuramoto lotka_volterra duffing fitzhugh_nagumo chua van_der_pol \
         pendulum mountain_car varma ou; do
  uv run --no-sync python scripts/run_crl_shared.py --device cuda --suite "$s" \
    --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic \
    --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99 \
    --kl-warmup-epochs 50 --out-dir outputs_shared_timing
done

# static and real cohorts, same idea
uv run --no-sync python scripts/run_ctee_static.py --device cuda --dataset ihdp \
  --replications 100 --out-dir outputs_timing
uv run --no-sync python scripts/run_ctee_static.py --device cuda --dataset acic \
  --replications 100 --out-dir outputs_timing
# ... twins, news, and the four real cohorts

# merge only the seconds field into the main rows
uv run --no-sync python scripts/merge_timings.py \
  --main outputs outputs_shared --timing outputs_timing outputs_shared_timing
```

`merge_timings.py` matches `(case, method, seed)` (or `(case, method)` for the
real panels), fills only non-finite values, and reports any block that has no
timing counterpart.

**Acceptance.** Every learned row has a finite `seconds`; the table generator
reads them; the caption states the hardware and that times are sequential.

**Cost.** One sequential pass over CRL (850 runs) plus the static panels (2 400
runs): ≈3–4 days on the 2080, longer on M4. Do not run concurrently with
anything else.

**Risk.** Timing is only meaningful on an idle machine; record the host and the
torch/CUDA versions in the report.

### T4. Freeze the paper after T1–T3

Regenerate tables, rebuild, remove the provisional caveats, update Appendix H
and Appendix I, and re-run the audit.

**Acceptance.** 85 pages, 0 errors, 0 overfull/underfull, 0 undefined; audit
FAIL=0; every table caption reports the actual replication/seed count.

---

## P1 — reproducibility and coverage of the current benchmark

### T5. Pin the ACIC snapshot

**Why.** ACIC replication `i` is the `i`-th directory in sorted-hash order, so a
different download changes the index→replication map. The archived 2080 rows and
the current rows are not paired (ridge on "replication 0": 0.836 vs 1.545).

**Plan.** Add a manifest with the SHA-256 of every replication directory (or of
`x.csv`/`z.csv`/`y.csv`/`mu0.csv`/`mu1.csv`) to `data/acic/manifest.json`; make
`ACICLoader` verify it and fail loudly on mismatch; store the manifest hash in
the run config.

**Acceptance.** Loading a different ACIC snapshot raises with an actionable
message; the config records the hash; the paper's ACIC caveat can be replaced by
"pinned snapshot".

**Cost.** ≈1 day (code + tests).

### T6. MPS driver workarounds

**Why.** CEGEN (`sde`) reproducibly aborts the Metal driver
(`commit command buffer with uncommitted encoder`); CITRIS needed a float64→CPU
fix (done in `9e4f8df`).

**Plan.** (a) Keep the CITRIS fix. (b) `finish_remaining.sh` now runs the
single-method blocks and forces `sde` onto CPU (done). (c) Optionally bisect the
CEGEN forward pass (SDE solver / float64 use) and guard it for MPS so the CUDA
device can be used everywhere.

**Acceptance.** The one-shot script completes on MPS without manual intervention;
the log records which blocks ran on CPU and why.

**Cost.** ≈half a day (b); 1–2 days (c).

### T7. CPS panel (optional, manual data)

**Why.** The paper lists CPS as an optional manual panel; it is absent from the
static table.

**Plan.** Request `cps.csv` from the CFRNet authors (no public mirror); place it
at `data/static/cps.csv` with columns `t, yf, ycf, mu0, mu1` plus covariates;
run 100 replications.

```bash
uv run --no-sync python scripts/run_ctee_static.py --device cuda \
  --dataset cps --replications 100 --out-dir outputs
```

**Acceptance.** 600 rows, finite PEHE, table caption includes CPS.

**Risk.** Data availability/licensing is the blocker, not compute.

### T8. Rendered-image Temporal Causal3DIdent

**Why.** The current `causal3d` suite is the *factor process* (four observed
scalars). The literature's Temporal Causal3DIdent feeds rendered RGB frames to
the representation learner, and the paper currently says that variant is
unimplemented, so its claim about high-dimensional visual observations is
untested.

**What already exists.** `src/gcmts/data/causal3dident.py` contains a complete
`Causal3DIdentImageGenerator`: six factors (object x/y, rotation, scale, hue,
spotlight x), a lagged acyclic mechanism with recorded single-target
interventions (`context["I"]`), and a differentiable 32x32 renderer (rotated
coloured object under a coloured spotlight, coloured background). The renderer is
a *surrogate* for Blender, with the same causal structure but simpler images.
`CNNEncoder`/`CNNDecoder` exist in `core/backbones.py` and are tested; the
generator is tested in `tests/test_data.py`. What is missing is the suite wiring
and image-capable methods.

**Work items.**

1. **Register the suite.** Add `causal3d_img` to `SUITES` in
   `scripts/run_crl_shared.py` with
   `Causal3DIdentImageGenerator(horizon=8, image_size=32, seed=seed)`,
   `observed_dim = 3*32*32`, `latent_dim = 6`, `max_lag = 1`; add
   `CrlBenchmark("A.4b", "Temporal Causal3DIdent (rendered images)", "temporal", True)`
   to `CRL_BENCHMARKS`. Add `image_shape: tuple[int, int, int] | None = None`
   to the `Suite` dataclass and set it to `(3, 32, 32)` for this suite, so
   `build_methods` can pass it to every method; return the known graph from the
   generator (`Batch(..., adjacency=self.adjacency)`) so SHD works, shape
   `(6, 6, 1)`.
2. **Make the five methods image-capable.** Add
   `image_shape: tuple[int, int, int] | None = None` and
   `image_feature_dim: int = 64` to `iVAE`, `LEAP`, `TDRL`, `NCTRL`, `MOSAIC`.
   When set, insert a shared `CNNEncoder(3, image_feature_dim)` that maps each
   flattened frame `(B*T, 3*32*32)` to features, and feed those features to the
   existing `GaussianHead`; keep the decoder outputting flattened pixels so the
   existing reconstruction losses and metrics are unchanged. Use the same
   encoder width, decoder width and epoch budget for every method (fairness).
   Put the shared code in one helper to avoid five divergent implementations.
3. **Pool format.** Full 32x32 float pools at the current 65 536x16 schedule are
   ~13 GB per seed. Use a reduced image pool (e.g. 8 192 train / 1 024 eval
   trajectories, horizon 8) and store uint8 if `save_dataset`/`batch_from_state`
   gain dtype support, otherwise float32 (~1.6 GB per seed). Document the
   reduced schedule in the table caption.
4. **Protocol and metrics.** `--suite causal3d_img --seeds 10`; report MCC,
   R^2_diag/sep and DCI on the six factors, SHD against the known graph, and
   (optionally) intervention-target AUROC from `context["I"]`. Report pixel
   forecasting MSE but mark it as a utility diagnostic, not a soundness metric.
   Epoch budget: start at 1 000 epochs; the measured CNN throughput is ~12.9k
   frames/s on M4, so one 2 000-epoch x 16-frame run is ~2.6 h and 50 runs are
   ~5.5 days on M4 (~1.5 days on the 2080). Horizon 8 + 1 000 epochs halves it.
5. **Tests.** Generator determinism and image range; adjacency shape; one
   forward/backward per method with `image_shape`; a one-seed smoke run at
   `--epochs-scale 0.01`; pool round-trip with the chosen dtype.
6. **Paper.** Add the dataset row, the results block, the Appendix H table and
   the audit note; state explicitly that the renderer is a surrogate, not the
   Blender teapot, so numbers are not comparable to the published Causal3DIdent
   results.

**Commands (after the code is in).**

```bash
uv run --no-sync python scripts/generate_synthetic_data.py \
  --suites causal3d_img --seeds 10 --train-size 8192 --eval-size 1024
uv run --no-sync python scripts/verify_outputs.py --pools-only
uv run --no-sync python scripts/run_crl_shared.py --device cuda \
  --suite causal3d_img --seeds 10 --epochs-scale 1.0 --data-source pool \
  --dataset-dir data/synthetic --scheduler adaptive \
  --early-stopping-patience 300 --ema-decay 0.99 --kl-warmup-epochs 50 \
  --out-dir outputs_shared
```

**Acceptance.** 50 rows (10 seeds x 5 methods), finite metrics, one table with
MCC/R^2/DCI/SHD, the paper's "unimplemented" sentence replaced, and the surrogate
caveat stated.

**Cost.** ≈2–3 weeks of implementation (suite + five methods + dtype + tests),
then ≈1.5–5 days of compute depending on horizon/epochs and host.

**Risks.** (a) The surrogate renderer is not the Blender dataset; the paper must
not compare absolute MCC with published Causal3DIdent numbers. (b) Images are
memory-heavy; the reduced pool must be reported. (c) A CNN adds parameters; keep
the same backbone and budget across methods so the comparison is fair.

---

## P2 — coverage of the literature (adjacent branches)

### T9. II.D representatives beyond CausalGAN

Missing: counterfactual generative networks (Sauer et al. 2021), causal market
simulators (Thumm et al. 2025), drift-aware simulation (Badre et al. 2025), and
LCD (Lorch et al. 2026) if the reviewed variant is not the implemented one.

**Plan.** One method per iteration: implement the mechanism, define a benchmark
that exposes its object (interventional law or counterfactual trajectory) with
ground truth, add it to `run_adjacent_benchmarks.py` or a new
`run_mechanism_benchmarks.py`, and report its headline metric. Do not add a
method to the table without a ground-truth protocol.

**Acceptance.** A per-method row with a benchmark, a metric and 10 seeds;
`verify_outputs` FAIL=0.

**Cost.** 1–3 weeks per method depending on code availability.

### T10. III.A/B/C representatives beyond the current one per branch

- III.A (decision): AdaDiffuser, sequential recourse (Gumucio et al.), causal-aware
  RL (Cao et al.).
- III.B (structure): root-cause analysis (Adesunkanmi et al.).
- III.C (explanation): Lopez-Alcaraz et al. 2025, Becker et al. 2026, ERCI, CIER.

**Plan.** Prioritise methods with public code and a ground-truth benchmark; one
new representative per branch per release; keep the per-object (not
head-to-head) framing.

**Acceptance.** Two representatives per branch with their own protocols, or an
explicit note that no shared benchmark exists.

**Cost.** 1–3 weeks per method.

---

## P3 — release engineering

### T11. Tagged release with DOI

**Plan.** Freeze `outputs/`, `outputs_shared/`, the manifests (`data/synthetic`,
`data/acic`), the audit reports and the paper tables; tag `v1.0`; archive on
Zenodo; add the DOI to the paper and `CITATION.cff`.

**Acceptance.** A DOI resolves; the archived tree reproduces every table with
the stored rows; `docs/audit_*.md` is included.

**Cost.** 1–2 days.

---

## Ordering and dependencies

| Order | Task | Depends on | Parallel with |
|---|---|---|---|
| 1 | T2 IHDP 100 | — | T1 |
| 2 | T1 grid | pools | T2 |
| 3 | T4 paper freeze | T1, T2 | — |
| 4 | T3 timings | idle host | — |
| 5 | T5 ACIC pin, T6 MPS | — | T1/T2 |
| 6 | T7 CPS, T8 rendered images | data / implementation | — |
| 7 | T9/T10 adjacent methods | implementation | — |
| 8 | T11 release | all | — |

**Definition of done for the paper.** T1, T2, T4 and T3 (if the Time columns are
kept) are complete; `verify_outputs` reports FAIL=0; every table caption matches
the stored configuration; the grid and static caveats are removed or replaced by
a pinned-snapshot statement.
