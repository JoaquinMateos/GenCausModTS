# Next Experiments — Plan

Status after implementing the seven red taxonomy targets and the rigor pass.
All methods are unit-tested (`uv run pytest -m "not slow"` is green) and the
taxonomy is now covered 12/12 (`uv run python scripts/taxonomy_coverage.py`).
The items below are the **benchmark runs** still required before the paper's
result tables can be filled. Ordered by priority.

## 0. Preconditions

1. Freeze a tagged release with a DOI (referenced in `09`).
2. Re-seed everything: the trainer now seeds `torch` when `seed` is set and
   caches the data iterator; all runners must pass a fixed `seed` schedule.
3. Confirm the four new generators expose ground truth under R1:
   `HierarchicalTemporalGenerator` (latents), `VoronoiGenerator` (graph +
   intervention targets), `PropagationGenerator` (path + source + graph),
   `BinarySCMGenerator` (counterfactual), `ConceptTimeSeriesGenerator`
   (causal concepts).

## 1. Adjacent branches (III.A–III.C + II.D) — new, blocking a table

Runner: `scripts/run_adjacent_benchmarks.py` (extend to 10 seeds and executor).

- **III.A GBTPL** on `PropagationGenerator` (D=6, T=16; sweep noise
  `sigma in {0, 0.1, 0.2}` and missing rates `{0,10%,30%}`); metrics
  path-F1, source accuracy, structural-violation rate, temporal deviation.
- **III.B LacaDM** on `MultiObjectiveMDPGenerator`; metrics hypervolume,
  sparsity, expected utility; compare against PCN/random-policy baselines.
- **III.C CausalConceptTS** on `ConceptTimeSeriesGenerator` (causal concepts
  known); metrics faithfulness, intervention efficacy, stability across seeds;
  compare causal vs associational (PredDiff) attribution.
- **II.D CausalGAN** on `BinarySCMGenerator`; metrics CF MAE, PEHE,
  interventional-law fidelity; compare against `CausalController`-only and
  observational-only ablations.

Deliverable: fill `latex/pic/experiments/adjacent.tex` (currently a target
protocol) and add a results table for II.D in `ctee.tex`; regenerate with
`scripts/make_paper_tables.py` (extend it to emit these rows).

## 2. Core CRL shared suites (pending re-runs)

Run the shared ten-seed protocol in `scripts/run_crl_shared.py`:

- **Suite A (CartPole/Pendulum)** — ten-seed re-run with the corrected
  structural-recovery estimator.
- **Suite A' (Temporal Causal3DIdent factor process)** — same; the
  `crl_shared_causal3d_agg.csv` partial (3/5 methods) must be completed with
  NCTRL and MOSAIC and to ten seeds.
- **Suite B (Voronoi / Causal Pinball)** — now includes **CITRIS and iCITRIS**
  (I.D.1/I.D.2); enables SHD, edge AUPRC and the triplet/two-sample metrics.
- **Suite C (linear SDE)** — CEGEN ten-seed run.

Also add **CHiLD** to the shared suite where a hierarchical generator is used,
and report MCC, R2_diag/R2_sep, DCI for the new CHiLD/iCITRIS blocks.

## 3. Effect-estimation suites (II)

- **Wu-IPTW** on the harmonic and tumour benchmarks (already wired into
  `scripts/run_ctee_benchmarks.py`): PEHE, eps_ATE, CF MAE/MBE, MMD2, axioms.
- Re-run all five estimators on the **static IHDP 100-replication** protocol
  (`scripts/run_ctee_static.py`) including Wu-IPTW.
- Re-run the ACIC 2016/2018 loaders through the same interface.
- Report the positivity-violation case (tumour no-treatment extrapolation)
  with Wu-IPTW's propensity weights as direct evidence.

## 4. Rigor re-runs after the trainer fixes

The trainer changes (seeded globals, cached iterator, skip-not-break on
non-finite loss, `estimate_effect` dispatching L2/L3) alter the RNG stream, so
**every existing ten-seed number must be regenerated**. In particular:

- `scripts/run_ctee_benchmarks.py`, `scripts/run_sdd_benchmarks.py`,
  `scripts/run_crl_shared.py`, `scripts/run_real_benchmarks.py`.
- Verify the context one-hot cardinality fix (`prepare_context(..., n_classes)`)
  does not change IVAE/LEAP/TDRL/MOSAIC results when all regimes are present.

## 5. Statistical comparison and tables

- Complete the **eighteen-problem Friedman/Nemenyi grid** for the CRL family;
  regenerate `latex/pic/experiments/stats.tex` and the CD diagrams.
- Recompute Holm-corrected paired Wilcoxon blocks for CRL, CTEE and SDD.
- Regenerate `matrix.tex`/`coverage` numbers via
  `scripts/taxonomy_coverage.py`; regenerate all experiment tables with
  `scripts/make_paper_tables.py`.

## 6. Sanity / matching-benchmark checks (Appendix)

For each newly implemented method add a matching-benchmark sanity check
(train, shape, and latent/path recovery on the generator whose assumptions it
satisfies): CHiLD, iCITRIS, Wu-IPTW, CausalGAN, GBTPL, LacaDM,
CausalConceptTS. Report in `app:sanity`.

## 7. Known rigor items intentionally deferred (document, then fix)

From the audit, the following are **not** yet fixed and must either be fixed or
stated as limitations before submission:

- `GANITE` adversarial objective is a single-coupled minimisation; needs
  alternating generator/discriminator updates (trainer support).
- `MOSAIC` uses an unconditional prior and an identity transition despite the
  regime-conditioned identifiability claim.
- ELBO reconstruction/KL reduction mismatch (`gaussian_reconstruction_nll`
  averages over `B*T*D`, KL averages over `B`) biases the effective KL weight.
- `CaTSG`/`CEGEN` re-estimate region boundaries per batch.
- `SlowFlows`/`CITRIS` reconstruction metric on an invertible map is
  identically zero.

## 8. Commands (quick reference)

```bash
cd gcmts
uv run python scripts/taxonomy_coverage.py
uv run python scripts/run_crl_benchmarks.py
uv run python scripts/run_crl_shared.py
uv run python scripts/run_ctee_benchmarks.py
uv run python scripts/run_ctee_static.py
uv run python scripts/run_sdd_benchmarks.py
uv run python scripts/run_adjacent_benchmarks.py
uv run python scripts/make_paper_tables.py
uv run pytest -m "not slow"
```
