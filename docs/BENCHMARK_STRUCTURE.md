# Benchmark structure and main-text evidence plan

Status: 2026-11-03. This document fixes what the benchmark measures, which
representatives cover each branch, and the **minimum** result set the main
document needs. Everything else belongs in the appendix. It is the contract for
the GPU run and for the paper tables.

## 1. Principle

A method is compared only on a benchmark that exposes ground truth for the
causal object it claims to return. Each branch therefore has one protocol, one
headline soundness metric and one utility metric where utility is meaningful.
The main text reports the comparison that supports a conclusion; per-benchmark
values, seed dispersion, secondary metrics and wall-clock times go to the
appendix.

## 2. Coverage

Every first-level branch has at least two independent representatives and every
sub-branch at least one. The six additions marked `new` close the previous
one-per-branch gaps.

| Branch | Representative(s) | Benchmark | Headline metric |
|---|---|---|---|
| I.A.1 temporal (discrete) | Slow Flows, shared temporal family | NP/VAR, slow-feature process | MCC |
| I.A.2 temporal (continuous) | CEGEN | continuous-time SDE | one-step MSE, W2 |
| I.B.1 regimes/environments | LEAP, TDRL, NCTRL | regime-shifted NP/VAR | MCC, R2 |
| I.B.2 invariant vs specific | SYNC; DANN, VREx, ERM baselines | multi-domain SCM | OOD MSE |
| I.C.1 environment labels | iVAE | NP/VAR with regimes | MCC |
| I.C.2 partial labels/views | CHiLD | hierarchical process | MCC |
| I.D.1 known interventions | CITRIS | Voronoi / Causal Pinball | MCC, SHD |
| I.D.2 imperfect targets | iCITRIS | Voronoi / Causal Pinball | MCC, SHD |
| I.E structural constraints | MOSAIC; **CaDRe (new)** | module/missingness; latent SCM | MCC, SHD |
| II.A.1 conditioning | GANITE | harmonic, tumour | PEHE |
| II.A.2 re-weighting | Wu-IPTW | harmonic, tumour | PEHE |
| II.B.1 balanced representations | CRN | harmonic, tumour, linear SEM | PEHE |
| II.B.2 disentangled factors | CEPAE | harmonic, tumour, linear SEM | PEHE |
| II.C latent confounders | CaTSG; **VGAN-CHG (new)** | static panels, temporal CF | PEHE, policy risk |
| II.D given mechanisms | CausalGAN; **CNF (new)** | binary SCM; compositional images | CF MAE |
| III.A decision | LacaDM; **SCARF (new)** | multi-objective MDP; sequential recourse | hypervolume; improvement |
| III.B structure | GBTPL; **RCA-VAE (new)** | propagation; regime shift | path-F1; root-cause AUROC |
| III.C explanation | CausalConceptTS; **Becker (new)** | concept-annotated series | faithfulness |

Fidelity labels: CaDRe and CNF are reimplementations from the papers (CNF uses a
surrogate renderer and omits VGG/U2-Net/adversarial terms); VGAN-CHG, SCARF,
RCA-VAE and Becker are compact surrogates because no public code exists. The
labels must appear in the table caption and the audit.

## 3. Protocol

- Ten seeds per learned synthetic block; 100 replications for ACIC, Twins and
  IHDP, 50 for News; single runs for the real cohorts.
- Paired Wilcoxon with Holm correction within each table block; Friedman and
  Nemenyi over the configuration grid; all tests and critical-difference
  diagrams computed with StaTDS (Luna et al., Neurocomputing 2024).
- Pools, ACIC snapshot and seeds are pinned; the sampler rewind is part of the
  protocol. Two runs that differ only in these details are not comparable.
- Timing is a separate sequential pass on an idle machine and is reported only
  in the appendix.
- Attainable significance: ten seeds cannot reach Holm significance for a
  pairwise comparison with more than a few tests; the text says so where it
  matters. Static panels with 100 replications can.

## 4. Main-text evidence set

The main text keeps **six result blocks**. Each block has one compact table or
figure, one headline comparison and one conclusion. The full tables, all
metrics, all seeds and the statistics table are in the appendix.

| # | Block | Main-text content | Conclusion it supports |
|---|---|---|---|
| 1 | CRL representation | MCC matrix (methods x shared benchmarks) plus mean rank; no per-seed values | No method dominates every process; the ordering follows the mechanism |
| 2 | Grid contrast | MCC tie and one-step forecasting separation, with the CD figure | Identifiability and downstream utility are separate axes |
| 3 | Static CATE | PEHE, ATE error and policy risk for IHDP, ACIC, Twins, News, with the reference floors | No method wins more than one panel; the data-generating process selects the winner |
| 4 | Counterfactual validity | Harmonic, tumour, linear SEM: PEHE against reconstruction/CF MAE | Reconstruction does not imply counterfactual validity |
| 5 | Static--dynamic | OOD MSE for oracle, ERM, DANN, VREx, SYNC; MIG/CDS as proxies | A proxy can improve without delivering the guarantee it certifies |
| 6 | Adjacent + new representatives | One row per object: path-F1, hypervolume/improvement, faithfulness, CF MAE, root-cause AUROC, MCC/SHD | Coverage is per object; these are evidence, not a leaderboard |

### 4.1 Minimum numbers per block

1. **CRL**: mean MCC and mean rank per method; one sentence on the per-process
   winners. Full MCC/R2/DCI/SHD table in the appendix.
2. **Grid**: Friedman statistic, p, CD and average ranks for MCC and one-step
   MSE; grid-mean MSE per method. The statistics table and the full CD diagrams
   go to the appendix.
3. **Static**: one row per method per panel, PEHE plus the two reference floors
   (zero effect, oracle mean effect). Policy risk only where it is defined.
   Masking deltas and the ACIC snapshot caveat in a footnote.
4. **CTEE**: PEHE and CF MAE for the harmonic, tumour and linear-SEM blocks;
   the no-effect and mean-effect references. The tumour row is the key: the
   method with the best CF MAE is not the method with the best PEHE.
5. **SDD**: OOD MSE for oracle, ERM, DANN, VREx, SYNC and the min Holm p; MIG
   and CDS in the same table to show that they do not rank the methods.
6. **Adjacent/new**: one headline metric per object, with the benchmark and
   ground truth named. No aggregate rank across objects.

### 4.2 What stays in the appendix

- The full per-benchmark CRL tables (four parts), all metrics and seeds.
- The grid statistics table and the three CD diagrams.
- The full static and real-cohort tables.
- The full adjacent table and the new-representative rows.
- Wall-clock times and the hardware/software record.
- The audit and the fidelity notes for the new surrogates.

## 5. Current status and the run that completes it

| Item | Status |
|---|---|
| Shared CRL, 18 suites | final |
| Grid, 18 configurations | rerun in progress |
| IHDP, 100 replications | rerun in progress |
| Rendered-image Causal3DIdent (A.4b) | in progress |
| Static ACIC/Twins/News, real cohorts | final |
| CHiLD, CITRIS/iCITRIS, CEGEN | final |
| Six new representatives | implemented and smoke-tested; 10-seed run pending |
| Sequential timing | pending |
| Statistics, tables, audit | regenerate after the runs |

The GPU run executes, in order: the pending driver (grid, IHDP, image), the six
new representatives, the sequential timing pass, then the statistics and table
regeneration. The run is resumable; the screen session is `Benchmark` on
`gpu_212`, `JOBS=2`, `GPU=1`.

## 6. Copy-back

When the run finishes, copy these from `gpu_212:~/gcmts` to the same paths in the
local tree:

| Remote | Local |
|---|---|
| `outputs/` | `gcmts/outputs/` |
| `outputs_shared/` | `gcmts/outputs_shared/` |
| `outputs_timing/` and `outputs_shared_timing/` | `gcmts/` |
| `outputs/new_representatives.json` and `_agg.csv` | `gcmts/outputs/` |
| `docs/audit_pending.md` | `gcmts/docs/` |
| `docs/statistical_analysis.md` and `.json` | `gcmts/docs/` |
| `docs/reference_floors.md` | `gcmts/docs/` |
| `latex/pic/experiments/` | `latex/pic/experiments/` |
| `outputs/logs/` | `gcmts/outputs/logs/` |

The manifests (`data/synthetic/manifest.json`, `data/acic/manifest.json`) are
identical if the same tree was rsynced; copy them only if the remote regenerated
pools.
