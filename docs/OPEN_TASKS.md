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

## How to run everything

Every runnable task below is orchestrated by one script:

```bash
bash scripts/run_pending_tasks.sh                 # device=mps
DEVICE=cuda bash scripts/run_pending_tasks.sh     # on the 2080 (default GPU)
GPU=1 DEVICE=cuda bash scripts/run_pending_tasks.sh  # physical GPU 1
SKIP_GRID=1 bash scripts/run_pending_tasks.sh     # skip the 3-4 day grid
SKIP_IMAGE=1 bash scripts/run_pending_tasks.sh    # skip the image suite
RUN_TIMING=1 bash scripts/run_pending_tasks.sh    # add the sequential timing pass
DRY_RUN=1 bash scripts/run_pending_tasks.sh       # print the command sequence only
```

The driver prints the interpreter it selected (`.venv/bin/python` when it has
torch, otherwise `uv run --no-sync python`, otherwise the active `python`) and
fails early with a clear message when none has torch.

`GPU=<n>` sets `CUDA_VISIBLE_DEVICES` before any Python process starts, so
physical GPU `n` becomes logical `cuda:0` for the whole run; the preflight prints
the visible device and its name. `DEVICE=cuda:<n>` also works, but prefer `GPU`
because a few library calls (profiling) synchronise the current device. Do not
pass both.

It is resumable and idempotent. The copied tree must contain `data/acic`,
`data/static`, `data/real` and `data/semi_synthetic`; the synthetic pools
(including the 18-config grid, ~28 GB) are regenerated in about a minute if
absent, and the image pools take about 30 s and 8.5 GB. `outputs/` and
`outputs_shared/` are resumed, so the existing 2026-11 rows are reused.

Implemented in this revision: the image suite and image path for all five CRL
methods (T8), the ACIC manifest and loader check (T5), the timing merge (T3),
and the CEGEN-on-CPU fallback (T6).

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

### T5. Pin the ACIC snapshot — **implemented**

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

### T6. MPS driver workarounds — **implemented in the driver**

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

### T8. Rendered-image Temporal Causal3DIdent — **implemented, run pending**

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

## P2 — coverage of the literature (complete benchmark)

### Coverage target

The benchmark is *complete* when every first-level branch of the taxonomy
(I.A--I.E, II.A--II.D, III.A--III.C) has at least **two independent
representatives**, each with a ground-truth protocol and ten seeds, and every
sub-branch has at least one. Baselines and reference predictors do not count
toward the two. Under this target the current matrix is complete for I.A, I.B,
I.C, I.D, II.A and II.B, and incomplete for six branches:

| Branch | Representative today | Missing | Task |
|---|---|---|---|
| I.E structural & functional constraints | MOSAIC | one mechanism-sparsity method | T9 |
| II.C latent-confounder & environment | CaTSG | one latent-confounder method | T10 |
| II.D given mechanisms & simulators | CausalGAN | one counterfactual generative method | T11 |
| III.A decision-making | LacaDM (surrogate) | one decision/recourse method | T12 |
| III.B structure discovery | GBTPL | one root-cause method | T13 |
| III.C explanation & attribution | CausalConceptTS (surrogate) | one attribution method | T14 |

The pending compute (T1 grid, T2 IHDP, T3 timing, T8 image suite) does not change
this table: it finalises values for methods already present.

Every task below follows one rule: do not add a method to Table 5 without a
benchmark that exposes its object with ground truth. If the original code is not
available, implement a compact reimplementation and label it as a surrogate in
the table, as already done for LacaDM and CausalConceptTS.

### T9. Second representative for I.E

**Candidates.** A mechanism-sparsity or differentiable-discovery method from the
I.E set (LaCIM; the differentiable causal-discovery baseline of Prashant et al.;
the general causal-representation method of Fu et al.).

**Plan.** Implement the mechanism, reuse the MOSAIC regime-conditioned series and
missing-data series so the comparison is within-benchmark, and score SHD and
module recovery against the known structure.

**Acceptance.** Ten seeds, finite SHD/module recovery, `verify_outputs` FAIL=0,
a second row in the I.E block and in Table 5.

**Cost.** 1--2 weeks.

### T10. Second representative for II.C

**Candidates.** FedCausalDiff (Li et al. 2026), the variational-GAN latent-confounder
model (Zhou et al. 2026), or the causal time-series estimator (Xia et al. 2025).

**Plan.** Implement the estimator on the harmonic and tumour benchmarks, where the
counterfactual is known, and report PEHE, MMD² and policy value.

**Acceptance.** Ten seeds, ground-truth counterfactuals, `verify_outputs` FAIL=0,
a second row in the II.C block and in Table 5.

**Cost.** 2--3 weeks.

### T11. Second representative for II.D

**Candidates.** Counterfactual generative networks (Sauer \& Geiger 2021) with
public code; CGN (Kocaoglu et al.) as the fallback.

**Plan.** Implement the method on the binary SCM already used for CausalGAN and on
one counterfactual image or sequence benchmark; reuse the rendered-image path of
A.4b for the latter. Report interventional-law fidelity and CF MAE with ground
truth.

**Acceptance.** Ten seeds, finite interventional/CF metrics, `verify_outputs`
FAIL=0, a second row in the II.D block and in Table 5.

**Cost.** 2--3 weeks.

### T12. Second representative for III.A

**Candidates.** Sequential recourse (Gumucio et al. 2026), AdaDiffuser (Feng et al.
2026), or causal-aware RL (Cao et al. 2026).

**Plan.** Use a multi-objective MDP or a recourse environment with a ground-truth
counterfactual policy; report hypervolume/regret, constraint violation and policy
validity under intervention.

**Acceptance.** Ten seeds, a second row in the III.A block and in Table 5.

**Cost.** 2--3 weeks.

### T13. Second representative for III.B

**Candidate.** Root-cause analysis (Adesunkanmi et al. 2025).

**Plan.** Use a synthetic graph with known propagation (the GBTPL protocol) or a
SWaT/WADI subset with known paths; report path-F1, source accuracy and structural
violations.

**Acceptance.** Ten seeds, a second row in the III.B block and in Table 5.

**Cost.** 1--2 weeks.

### T14. Second representative for III.C

**Candidates.** ERCI (Wang et al. 2025), CIER (Wang et al. 2024), or Becker et al.
2026. CausalConceptTS already implements the Lopez-Alcaraz protocol, so it does
not count as a second method.

**Plan.** Use a concept-annotated synthetic classifier with known attributions;
report faithfulness, intervention efficacy and explanation stability.

**Acceptance.** Ten seeds, a second row in the III.C block and in Table 5.

**Cost.** 1--3 weeks.

### T15. Audit-named II.A/II.D methods (fidelity, optional)

PFD-BDCM and CaPaint (II.A), CGN and LCD/CLIPR (II.D) are named as unimplemented
in the audit appendix. They are not needed for the two-per-branch target: II.A
already has GANITE and Wu-IPTW, and II.D gains a second method in T11. Implement
them only if the review's coverage claim is extended to those specific methods;
otherwise record the exclusion explicitly in the audit and in Table 5.

**Cost.** 1--3 weeks per method.

### T16. Replace the adjacent surrogates (fidelity, optional)

LacaDM and CausalConceptTS are compact surrogates. If the authors' code is
available, replace them and rerun the adjacent block; if not, keep the surrogate
label in Table 5 and in the caption. GBTPL is a full reimplementation and is not
affected.

**Cost.** 1--2 weeks for both.

---

## P3 — release engineering

### T17. Tagged release with DOI

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
| 7 | T9--T14 second representatives | implementation | T1--T3 |
| 8 | T15/T16 fidelity (optional) | T9--T14 | — |
| 9 | T17 release | all | — |

**Definition of done for the paper.** T1, T2, T4 and T3 (if the Time columns are
kept) are complete, and T9--T14 are complete, so every first-level branch has at
least two representatives; `verify_outputs` reports FAIL=0; every table caption
matches the stored configuration; the grid and static caveats are removed or
replaced by a pinned-snapshot statement. T15 and T16 are fidelity work and do not
block the coverage claim.
