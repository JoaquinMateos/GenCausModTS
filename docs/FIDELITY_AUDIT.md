# Fidelity audit and remaining work

Status: 2026-11-05. Scope: implementations, architectures, hyperparameters, metrics and result claims in the benchmark paper. This file replaces the historical defect narrative as the working checklist; the paper itself now reports only the current state.

## Fixed in this pass

| Item | Change | Test |
|---|---|---|
| iCITRIS native graph | Instantaneous gates now use the benchmark convention `A[child, parent]`; transition and temporal expansion use the same convention; allowed lower-triangular gates are initialised active | `tests/test_adjacent.py::test_icitris_native_graph_orientation` |
| Time-varying SEM | One initial covariate state is shared by the propensity pass, factual trajectory and counterfactual; the companion-radius cap is iterated on the actual companion matrix | `tests/test_data.py::test_time_varying_sem_is_stable_and_paired` |
| CTEE temporal evaluation | A fixed evaluation batch is drawn from a separate seeded generator before training and shared by every method for a dataset/seed | one-seed full-budget check (`outputs_sem_check`) |
| CRL fresh-data evaluation | The fresh-data path now pre-draws one evaluation batch per suite/seed instead of sampling after training | full test suite |
| Static CTEE rewind | Each method and baseline gets a fresh train/test generator with the same seed and held-out units | static smoke run |
| Instantaneous SHD | Latents are aligned to the true factor order before estimating the instantaneous graph; post-hoc permutation is no longer applied to a causal-order estimator | full test suite |
| GANITE static adaptation | The factual-outcome slot is a scalar vector, supervised loss is on the outcome coordinate, and the ITE loss is the difference of potential outcomes | `tests/test_effect.py`, static smoke run |
| Table 23 / claims | The fidelity table now states that LEAP/TDRL/NCTRL/MOSAIC are simplified reimplementations and that current values are diagnostics, not replications | paper build |
| SDD claim | The paper no longer says VREx fails; it states that the DGP is a mechanism-shift stress test, not a positive invariance test | paper build |
| Adjacent/SDD labels | Surrogate/partial status is explicit in Table 5, the adjacent caption and the SDD caption | paper build |
| Pool provenance | Pools now carry a generator-revision hash; a finite stale pool is rejected | `tests/test_synthetic_store.py`, `tests/test_regressions.py` |
| Batch preservation | `instantaneous_adjacency` and `target` survive pooling, subsetting and device moves | `tests/test_regressions.py` |
| New-representative SHD | Latents are aligned before `SHDMetric(aligned=False)`; the earlier optimistic default is removed | completed; canonical JSON updated |
| LaLonde covariates | `educ/hisp/marr` are mapped to the canonical names | `tests/test_real.py` |

## Audit items already fixed or verified

| Audit item | Status | Evidence |
|---|---|---|
| 1. nilpotent temporal generator | fixed in default path | `data/synthetic.py` lower-triangular positive-persistence VAR; stored pool radii/autocorrelation checks |
| 2a. Voronoi instantaneous upper triangle | fixed | `data/voronoi.py` lower-triangular instantaneous mask; generator test |
| 3. NCTRL regime radii | fixed | `data/synthetic.py` rescales regime matrices to radius <= 0.8 |
| 5. SDD zero-average spurious coefficients | fixed for 5-domain design | `data/synthetic_sdd.py` coefficients `[1,.8,.6,.4]` train, `-1` test |
| 6. IHDP train/test split | fixed | `run_ctee_static.py` loads canonical train/test files |
| 7. synthetic static panels | fixed for executed panels | canonical Twins/News/ACIC/Jobs loaders; `static_effect.py` is a dead fallback |
| 8a/8b/8c/8d/8e/8f. metrics | fixed/partial | unclipped R2, one-edit SHD, KSG MIG, median MMD, worst-corner hypervolume, `ddof=1`; see residual notes below |
| 10a. EMA | fixed | `EMA_DECAY=0.99` applied by all runners |
| 11. static outcome leakage | fixed at input | `IHDPGenerator` masks the outcome coordinate; no method reads `batch.counterfactual` |
| 12. training sampler rewind | fixed for training | `PooledGenerator.reset()` before each method |
| 14. CHiLD negative KL | fixed | trajectory-level KL in `child.py` |

## Material gaps still open

### P0 — shared temporal CRL

- **LEAP** is a Gaussian-transition surrogate, not the paper's inverse-transition + spline flow with noise discriminator, TC term, mask L1 and bidirectional GRU.
- **TDRL** uses Gaussian group priors, not modular inverse-transition flows; the encoder conditions on `u`, and the observation-shift block has no observation shift.
- **NCTRL** uses a Gaussian prior rather than the ARHMM/flow prior, does not use the filtered regime posterior in `transition()`, and is not tested on a persistent HMM generator.
- **MOSAIC** is single-stage, without the dense Stage-1 decoder, temporal prior, regime embedding, beta/gamma weights, sparsity schedule or entropy penalty.
- **iVAE** is a reduced architecture; on `r=1` suites the auxiliary variable is constant and the method reduces to an unconditional VAE.

These are the largest fidelity gaps. The shared ranking is a comparison of simplified representatives, not of the published methods. The paper now says this in §9.3 and Table 23.

### P0/P1 — static CTEE

- The static losses still reconstruct the full `[covariates, outcome]` vector; only GANITE currently restricts its supervised loss to the outcome coordinate. CRN/CEPAE/CaTSG still allocate most loss mass to covariates.
- GANITE still uses joint training, alpha=beta=1 (not 2/5), and a discriminator that can see covariates; the static architecture is an adaptation.
- CRN uses fixed lambda=1, joint encoder/decoder training, and a static head not present in the paper.
- CEPAE uses ``entropy_weight=0.01`` (paper 0.09), an MLP rather than 1D convolutions, and no history encoder.
- CaTSG omits the swapped-prediction and orthogonality losses, CFG dropout and Sinkhorn; it uses a softmax mixture.
- Wu-IPTW uses a Gaussian propensity for binary treatments and 1/99 rather than 0.01/99.99 clipping.
- CausalGAN does not implement the anti-labeler correctly, and the controller does not re-propagate descendants under intervention.
- VGAN-CHG and CNF remain compact surrogates; their numbers are coverage only.

### P1 — adjacent and SDD

- LacaDM evaluates an action matrix as if it were a policy front; its hypervolume is not the paper metric.
- SCARF's improvement/fairness are computed on the jointly trained classifier, not on exact post-recourse states and a frozen base classifier; long-term loss is disabled.
- GBTPL uses an unordered path metric, a graph that can contain cycles, and no reverse-diffusion/candidate re-ranking.
- RCA-VAE injects the true regime label into the encoder and uses AUROC/path-F1 proxies, not the paper's mechanism-KL metrics.
- CausalConceptTS uses a deterministic MLP instead of a diffusion imputer and fixed classifier.
- Becker does not apply Ada-GVAE alignment to the predictor/attributions.
- SYNC is a compact surrogate, not the paper's MI/contrastive model.
- The SDD DGP does not provide a positive DANN test (domain classifier at chance) or VREx test (risk-variance optimum at the average spurious coefficient). The paper now reports this as "not testable", not a refutation.

### P1/P2 — data and protocol

- **SEM**: fixed pairing/radius; the 10-seed Linear SEM block must be rerun.
- **Voronoi/B**: alignment is fixed; the 10-seed Benchmark B block must be rerun.
- **CTEE temporal**: evaluation draw is fixed; the block must be rerun and the Wilcoxon tests regenerated.
- **CTEE static**: rewind is fixed; a full static rerun is needed after the static loss/architecture work.
- **Pool provenance**: `_existing_is_usable` still checks only finiteness; a stale finite pool can be reused. Add a generator revision hash before reruns.
- **LaLonde**: loader drops `educ/hisp/marr`; fix names and rerun.
- **Jobs**: loader uses train only; use the released test file or label as in-sample.
- **Real cohorts**: trained/evaluated on the same units; split or label.
- **Twins**: outcome is mortality but policy risk treats higher as better; fix polarity or label.
- **Batch helpers**: preserve `instantaneous_adjacency` and `target` in `_index_batch`, `_subset_batch`, `move_batch`, and the synthetic store.
- **SHD default**: set `aligned=False` by default or require explicit alignment; new-representative rows currently use the optimistic default.

## Local verification performed

- Full test suite: `235 passed, 1 skipped`.
- iCITRIS orientation and SEM pairing tests added and passing.
- CTEE temporal smoke (CRN/GANITE, harmonic, one seed): runs end-to-end with the shared evaluation batch.
- CTEE static smoke (IHDP, one replication, GANITE/CRN/CEPAE/CaTSG): runs end-to-end with per-method generator rewind.
- One-seed full-budget Linear SEM check is running in `outputs_sem_check`; the result will determine whether the paper's linear-SEM table changes.

## Protocol decision (2026-11-05)

Official modules will be used, but the shared comparison keeps the common
benchmark protocol (same seeds, data and training budget) so the ranking remains
controlled. Paper-specific fidelity checks are reported separately, not mixed
into the shared table. Large reruns will run on `gpu_212`; only corrected CTEE
blocks and one-seed pilots are run locally first.

## Next reruns, in order

1. Linear SEM, 10 seeds, after SEM fix.
2. Benchmark B / Voronoi, 10 seeds, after alignment fix.
3. CTEE temporal, 10 seeds, after shared-evaluation fix.
4. IHDP/static, 100 replications, after static loss scope and GANITE/CEPAE/CRN/CaTSG fixes.
5. Rebuild tables and regenerated statistics; update Table 23 and the affected result paragraphs.

## GPU runs launched (2026-11-05)

- `gpu_212`, GPU 1: IHDP 100 replications into `outputs_static_fixed`, pinned to CPUs 0--1.
- `gpu_212`, GPU 1: Voronoi/Causal Pinball 10 seeds into `outputs_shared_voronoi_fixed`, pinned to CPUs 2--3. Both use single-threaded BLAS/OpenMP, so the total CPU budget is four.
- Local: CTEE harmonic/tumour 10 seeds with shared evaluation draws into `outputs_ctee_fixed` (resumed after the local wall-clock timeout).

## Result checks in progress

### Linear SEM (one seed, full budget)

After the paired-counterfactual fix, the one-seed full-budget check in
`outputs_sem_check` gives:

| Method | PEHE |
|---|---:|
| CEPAE | 1.025 |
| CATSG | 1.068 |
| Wu-IPTW | 1.124 |
| GANITE | 1.159 |
| CRN | 1.561 |
| No effect | 1.081 |
| Oracle mean | 1.028 |

The previous CRN PEHE of 0.110 was an artefact of unpaired initial covariate
states; CRN is not the best estimator after the fix, and no method improves on
the oracle-mean reference. The paper's linear-SEM result is marked provisional
until the ten-seed block (`outputs_sem_fixed`) replaces it.

### Official-source integration

All method files now begin with paper/source provenance. Official TDRL, LEAP, MOSAIC, NCTRL and iVAE sources are vendored/adapted under `src/gcmts/vendor/` and the method package. The shared runner uses the official adapters for LEAP, TDRL, NCTRL and MOSAIC. The official iVAE adapter is retained for the fidelity diagnostic; the shared temporal table uses the image/temporal `IVAE` adapter because the official i.i.d. iVAE diverges on long flattened sequences. The TDRL nonparametric prior is chunked over batch slices; the single-seed `npvar` diagnostic gives MCC 0.775 (TDRL), 0.570 (LEAP), 0.625 (NCTRL), 0.806 (MOSAIC) and 0.397 (iVAE). Table 23 now reports these official-source values. A five-seed 200-epoch benchmark, using the Optuna-selected budget and early stopping, is running on `gpu_212` in two pinned workers.

### Pending result reruns

- CTEE temporal and linear SEM: completed in `outputs_ctee_fixed2`; Table 7/31 updated.
- Static IHDP/ACIC/Twins/News: must be rerun after the static loss-scope and
  model adaptations are resolved.
- Benchmark B / Voronoi: must be rerun after the instantaneous-graph alignment
  fix.
- New representatives: completed; Table 8 updated.
- TDRL official: the batch-chunked prior fits the 10 GB GPU; the full ten-seed shared rerun is still pending because of the 2000-epoch Jacobian cost.
