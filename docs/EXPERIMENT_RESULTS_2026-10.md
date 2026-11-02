# Experimental results after the October 2026 audit

Local execution on an Apple M4 (MPS), analysis with StaTDS 1.1.7. This document
records what was run, what the results say, how they compare with the original
papers, and which code defects were found and fixed. It is updated as the
background runs finish; entries marked **(running)** are provisional.

---

## 1. Environment and protocol

| Item | Value |
|---|---|
| Host | Apple M4, 10 cores, 24 GB unified memory |
| Device | `mps` (`torch 2.14.0`), AMP disabled (CUDA-only) |
| Library revision | `7d74bcc` + the fixes in §6 |
| Protocol flags | `GCMTS_PREFETCH=1 GCMTS_LIGHT_BATCH=1`, `--epochs-scale 1.0`, `--scheduler adaptive`, `--early-stopping-patience 300`, `--ema-decay 0.99`, `--kl-warmup-epochs 50`, `--data-source pool` |
| Statistics | StaTDS 1.1.7 (`friedman`, `nemenyi`, `holm`) |

**Library caveat.** StaTDS evaluates the χ² tail with a truncated series that
saturates near 10⁻¹¹: for χ² = 72, df = 4 it returns 3.5×10⁻¹¹ instead of
8.6×10⁻¹⁵. Its ranks, Nemenyi critical distance and Holm table are exact, so the
report uses the StaTDS statistic and ranks and recomputes the Friedman p-value
with the exact χ² distribution (`scipy.stats.chi2.sf`). Every p-value below is
the exact one; the raw StaTDS value is kept in `statistical_analysis.json`.

The environment flags are part of the protocol, not an implementation detail:
they change the minibatch RNG stream (§3.2). Every run recorded here uses the
same values as the 2080 production run.

---

## 2. Executed blocks

| Block | Status | Runs | Where |
|---|---|---|---|
| CRL A.9 Lotka–Volterra | Complete | 50 | `outputs_shared/crl_shared_lotka_volterra.json` |
| CTEE static NEWS (50 replications) | Complete | 300 | `outputs/ctee_static_news.json` |
| CTEE real cohorts (LaLonde, Jobs, Oregon, NHEFS) | Complete, masked | 24 | `outputs/ctee_static_{lalonde_rct,jobs_rct,oregon_rct,nhefs}.json` |
| Static-masking A/B (IHDP reps 0–4) | Complete | 40 | `docs/diagnostics_masking.json` |
| CRL reset check (Duffing, 10 seeds) | Complete | 50 | scratch (§3.3) |
| CTEE static IHDP (masked) | Complete | 180 (30 rep × 6) | `outputs/ctee_static_ihdp.json` |
| MPS↔CUDA equivalence (Duffing, seed 0) | Complete | 5 | scratch |

The 17 other CRL shared suites, the grid, CTEE temporal, the static panels and the
real external-validity block were produced on the 2080 and are reused after the
equivalence checks in §3.

---

## 3. Protocol equivalence

### 3.1 Device

Same cell (Duffing, seed 0, all five methods, full budget), MPS against the stored
CUDA rows:

| Method | CUDA | MPS | Δ MCC |
|---|---|---|---|
| iVAE | 0.517 | 0.516 | −0.001 |
| LEAP | 0.943 | 0.931 | −0.012 |
| TDRL | 0.942 | 0.930 | −0.012 |
| NCTRL | 0.699 | 0.761 → 0.706 after §3.2 | +0.007 |
| MOSAIC | 0.950 | 0.950 | 0.000 |

MPS reproduces CUDA within one percentage point of MCC on four of five methods.
NCTRL was the outlier until the pool-reset defect (§3.3) was found.

### 3.2 The environment flags are part of the protocol

NCTRL on Duffing, seed 0:

| Configuration | MCC |
|---|---|
| both flags off | 0.706 (alone) / **0.761 (after four other methods)** |
| `GCMTS_PREFETCH=1` | 0.706 |
| `GCMTS_LIGHT_BATCH=1` | 0.706 |

Enabling either flag changes the result, and with both off the result depends on
how many methods ran before. This is the pool-reset defect.

### 3.3 Pool-reset defect (affects every CRL block)

`run_suite` loads one `PooledGenerator` per `(suite, seed)` and reuses it for all
five methods. Its internal RNG advances with every minibatch, so the k-th method
sees a different data stream than the first. A method's score therefore depends
on its position in the method list, and on whether prefetching drew a batch ahead.
NCTRL on Duffing scored 0.706 alone and 0.761 after four other methods.

**Fix.** `PooledGenerator.reset()` rewinds the sampler, and `_run` calls it before
every fit, so every method in a block sees the same stream. After the fix NCTRL is
0.706 both alone and after four methods, and within 0.007 of CUDA.

**Consequence.** The CRL numbers in `outputs_shared/` were produced without the
reset and share this confound. A ten-seed Duffing rerun with the reset quantifies
it:

| Method | Reset | No reset | Δ |
|---|---|---|---|
| LEAP | 0.903 | 0.907 | −0.004 |
| TDRL | 0.901 | 0.881 | +0.020 |
| NCTRL | 0.853 | 0.892 | −0.039 |
| MOSAIC | 0.806 | 0.806 | 0.000 |
| iVAE | 0.760 | 0.777 | −0.016 |

The largest move is 0.039 MCC (NCTRL) and TDRL/NCTRL swap adjacent ranks; the
top (LEAP/TDRL) and bottom (iVAE) are unchanged. The reset therefore matters for
exact reproducibility and for the validity of paired tests, but not for the
qualitative CRL conclusions. The statistical analysis in §4 is computed on the
pre-reset rows and must be regenerated after the CRL rerun; conclusions that
depend on adjacent-rank order are marked accordingly.

---

## 4. Statistical comparison (StaTDS)

Friedman test per block, with Nemenyi critical distance and Holm post-hoc against
the best-ranked method. A case is a benchmark (CRL), a replication (static CATE),
or a panel (real data).

### 4.1 CRL representation quality (MCC, higher better)

18 shared suites, including the completed A.9 Lotka–Volterra rerun.

| Method | Mean rank | Mean MCC | Median |
|---|---|---|---|
| NCTRL | 1.78 | 0.782 | 0.816 |
| LEAP | 2.28 | 0.761 | 0.756 |
| TDRL | 2.67 | 0.741 | 0.761 |
| MOSAIC | 3.39 | 0.730 | 0.748 |
| iVAE | 4.89 | 0.581 | 0.546 |

Friedman χ² = 42.09, p = 1.6×10⁻⁸. Holm: NCTRL–iVAE (adjusted
p = 1.4×10⁻⁸) and NCTRL–MOSAIC (p = 0.0067) survive; NCTRL–LEAP (p = 0.34) and
NCTRL–TDRL (p = 0.18) do not. **The four
temporally-aware methods form one group; iVAE is separated.** This is exactly the
qualitative prediction of the methods' own assumptions: iVAE assumes stationary
i.i.d. latents and has no mechanism for time.

### 4.2 CRL one-step forecasting (MSE, lower better)

| Method | Mean rank | Mean MSE | Median |
|---|---|---|---|
| LEAP | 2.06 | 0.189 | 0.008 |
| NCTRL | 2.06 | 0.192 | 0.009 |
| TDRL | 2.67 | 0.227 | 0.008 |
| MOSAIC | 3.67 | 0.275 | 0.017 |
| iVAE | 4.56 | 0.274 | 0.021 |

Friedman χ² = 34.27, p = 6.6×10⁻⁷; LEAP and NCTRL are tied at the top.
The ranks agree with MCC, so representation quality and downstream utility point
the same way across suites (they do not on the grid; §4.3).

### 4.3 18-problem grid

MCC: χ² = 5.02, p = 0.285 — **not significant**. Means LEAP 0.752, TDRL 0.751,
iVAE 0.739, NCTRL 0.738, MOSAIC 0.733.

One-step forecasting: χ² = 72.00, p = 8.58×10⁻¹⁵. Ranks LEAP 1.00, NCTRL 2.00,
TDRL 3.00, iVAE 4.00, MOSAIC 5.00; MSE 4.515, 4.622, 5.586, 9.110, 9.198.

This reproduces the paper's central empirical contrast on the corrected grid: a
representation metric that is statistically tied (p = 0.29) hides a two-fold
difference in downstream error. The ordering is identical to the pre-audit run,
which strengthens rather than weakens the claim.

### 4.4 Static CATE (√PEHE, lower better), masked and final

| Panel | n | Best | Worst | Friedman p |
|---|---|---|---|---|
| IHDP | 30 | ridge_t 3.230 | catsg 7.027 | 1.9×10⁻¹⁴ |
| ACIC | 100 | catsg 0.174 | ganite 1.310 | 5.6×10⁻⁶⁷ |
| Twins | 100 | catsg/crn 0.319 | nearest_neighbour 0.536 | 1.5×10⁻⁹⁰ |
| NEWS | 50 | cepae 3.449 | ganite 12.948 | 1.6×10⁻⁴⁶ |

Full means (√PEHE):

| Method | IHDP | ACIC | Twins | NEWS |
|---|---|---|---|---|
| CaTSG | 7.027 | **0.174** | **0.319** | 4.326 |
| CRN | 5.509 | 0.309 | **0.319** | 3.802 |
| CEPAE | 4.711 | 0.694 | 0.320 | **3.449** |
| GANITE | 3.295 | 1.310 | 0.472 | 12.948 |
| ridge T-learner | **3.230** | 0.649 | 0.321 | 9.570 |
| nearest neighbour | 6.906 | 0.984 | 0.536 | 4.641 |

No method wins on more than one panel. Twins is saturated (the constant-ATE
floor is 0.319, and four methods sit on it); GANITE and nearest neighbour fall
below the floor. The paired IHDP comparison (same 30 replications) is
GANITE 5.40 → 3.30, CRN 6.19 → 5.51, CEPAE 5.05 → 4.71, CaTSG 5.00 → 7.03, with
ridge and nearest neighbour unchanged. On ACIC, CaTSG improves from 0.354 to
0.174 and CRN, ridge and nearest neighbour are unchanged; on Twins GANITE
worsens from 0.329 to 0.472; on NEWS nothing changes materially.

### 4.5 Reference floors (the lens for §4.4)

For every static panel, two predictors that use no heterogeneity are computed per
replication: predict zero effect (√PEHE = RMS of the true effect) and predict the
true ATE constant (√PEHE = SD of the true effect). A method that does not beat the
constant-ATE floor has learned no heterogeneity.

| Panel | Zero-effect floor | Constant-ATE floor | Best method | Beats constant-ATE |
|---|---|---|---|---|
| IHDP (masked, 30 replications) | 9.10 | 6.82 | ridge 3.23 | 93% of replications |
| ACIC | 0.199 | 0.053 | crn 0.311 | 7% |
| Twins | 0.319 | 0.319 | catsg 0.319 | 0% |
| NEWS | 4.33 | 3.33 | cepae 3.43 | 4% |

Three consequences:

On IHDP with masking, the constant-ATE reference is 6.82 (mean over the 30
replications) and the zero-effect reference 9.10; ridge beats the constant-ATE
reference on 93% of replications, CRN on 67%, CEPAE on 53%, GANITE on 40% and
CaTSG on 37%. So CRN does learn heterogeneity once the outcome is no longer an
input, but only ridge does so consistently.

1. **Twins is saturated.** The constant-ATE floor equals the zero-effect floor
   (0.319), because the treatment effect is nearly homogeneous across the 11 400
   units. Every method sits on it; the rank differences (0.319 vs 0.329) are
   below the floor and should not be reported as a ranking.
2. **NEWS does not demonstrate heterogeneity learning.** The best method is at
   the constant-ATE floor (3.43 vs 3.33) and beats it in 4% of replications.
3. **ACIC is uniformly hard for these implementations**: no method beats the
   constant-ATE floor in more than 7% of replications, and all are worse than a
   zero-effect predictor on average. This is a finding about the implementations,
   not about the methods' published settings.
4. Only IHDP shows genuine heterogeneity learning, and only ridge does it
   consistently (93% of the masked replications; CRN 67%).

---

### 4.6 Temporal CTEE: effect error versus counterfactual error

PEHE (effect error) and CF MAE (counterfactual trajectory error) are different
axes, and the two claims of CRN/CEPAE concern the second.

| Benchmark | Axis | CRN | CEPAE | CaTSG | GANITE | Wu-IPTW |
|---|---|---|---|---|---|---|
| harmonic | √PEHE | **0.270** | 0.634 | 1.098 | 1.130 | 1.479 |
| harmonic | CF MAE | **0.181** | 0.211 | 0.458 | 0.325 | 1.105 |
| tumour | √PEHE | **6.416** | 6.726 | 6.653 | 7.291 | 6.840 |
| tumour | CF MAE | 1.070 | 0.944 | **0.828** | 2.098 | 0.998 |
| linear SEM | √PEHE | **0.110** | 1.380 | 1.053 | 1.411 | 1.444 |
| linear SEM | CF MAE | **0.512** | 0.824 | 0.772 | 0.853 | 0.795 |

CRN is best on both axes on the harmonic and linear-SEM blocks, which is what its
counterfactual-recurrent construction predicts. On the tumour block CaTSG has the
lower trajectory error (0.828 versus 1.070) while CRN has the lower effect error;
this is the same inversion the paper reports at the block level, now visible at
the metric level.

## 5. Consistency with the original papers

| Method | Original evaluation | Published result | Ours | Verdict |
|---|---|---|---|---|
| GANITE (II.A) | IHDP, Twins, Jobs | IHDP √PEHE 2.4±0.4 out-of-sample; Twins 0.297±0.016 | IHDP 4.98 (median 4.21); Twins 0.329 | **Inconsistent on IHDP before §6.1; 1.40 on rep 0 after the fix.** Twins at the floor. |
| CRN (II.B) | Tumour-growth PK-PD + MIMIC-III counterfactual RMSE | Lower counterfactual error than MSM/linear | Tumour √PEHE 6.42: best of five, better than the zero-effect floor (6.73), worse than the constant-ATE floor (6.04) | **Partially consistent**: CRN is the best of the five on the tumour block, as its inductive bias predicts, but the block is an extrapolation regime where a constant effect still wins. |
| CEPAE (II.B) | Synthetic and semi-synthetic time series (Rossmann), CF MAE/MBE | Abrupt-change setting | CF MAE 0.211 on harmonic (second after CRN 0.181); 0.944 on tumour; NEWS √PEHE 3.43 at the constant-ATE floor (3.33) | **Partially consistent**: the CF-MAE claim (low counterfactual error) holds on harmonic, but the generator and metric differ, and the effect-error advantage on NEWS is marginal. |
| CaTSG (II.C) | Synthetic + real TSG (traffic/temperature); metrics are distributional (KL, MMD, J-FTSD), not PEHE/ATE | Backdoor-adjusted guidance improves counterfactual fidelity | ACIC best of six (0.354); harmonic 1.098 (no-effect floor 1.123); tumour 6.653; masking does not change it (2.745 vs 2.743 on IHDP rep 0) | **Not testable directly**: the original metric family is distributional and the original task is time-series generation. Our effect-error numbers are an extension to a static CATE task and must not be read as a replication verdict. |
| iVAE / LEAP / TDRL / NCTRL / MOSAIC (I) | Each on its own identifiability benchmark | Identifiability under its own assumptions | MCC: NCTRL/LEAP/TDRL/MOSAIC tie; iVAE separated | **Consistent in the qualitative sense** (the four temporal methods beat the stationary one), but the block is a re-implementation test, not a replication. |

Two published numbers are directly comparable and both matter:

- **GANITE on IHDP** (2.4±0.4) is the only IHDP number our static block can be
  scored against. Before the fix we were at 4.98; after masking, rep 0 gives 1.40,
  in the published range. The full masked rerun is needed before this is claimed.
- **The published SOTA on IHDP** (TARNet 0.95, CFR-Wass 0.76 out-of-sample) is not
  reproduced by any implementation in the library, because TARNet/CFR are not
  among the implemented methods. The paper should not imply that our methods
  reach that level; it should say that the benchmark can distinguish them, and
  report the gap.

---

## 6. Defects found and fixed

### 6.1 Static outcome leak (affects IHDP, ACIC, Twins, NEWS, all real cohorts)

**Symptom.** On IHDP, GANITE's median √PEHE (4.21) is the zero-effect floor
(4.13); CRN, CEPAE and CaTSG are 2–5. Ridge, which sees covariates only, scores
2.40 and beats a constant-ATE predictor in 87% of replications.

**Cause.** `IHDPGenerator` builds the observed state as `[x_1..x_p, y]`, and every
static model reconstructs that state. The factual outcome is therefore an input
feature, and the model can copy it for both treatment arms, yielding τ≈0. The
ridge T-learner uses `generator.covariates` only, so the comparison was not
between the same information sets.

**Fix.** The generator now masks the outcome coordinate in `batch.x` and carries
the true state in `batch.target`; the four losses (CRN, CEPAE, CaTSG, GANITE) use
the target. Every method now sees covariates and treatment, and predicts the
outcome.

**What the fix does and does not mean.** Conditioning on the factual outcome is
legitimate for a Level-3 counterfactual query — that is what abduction is — but it
is not legitimate for a *CATE* benchmark, where the target is
$\mathbb{E}[\tau\mid X]$ and the baselines (ridge T-learner, nearest neighbour)
see covariates only. The block is described as static CATE, so the information
sets must match; masking is the correct choice for that claim. The unmasked
variant measures a different, also defensible, task (counterfactual prediction
given the factual outcome), and it gives CRN and CaTSG an input the baselines do
not have. The two variants must not be mixed in one table.

**Measured effect** (IHDP, replication 0, 400 epochs):

| Method | τ̂ mean (true 4.06) before | after | √PEHE before | after |
|---|---|---|---|---|
| GANITE | 0.52 | 4.30 | 3.62 | **1.40** |
| CRN | 3.09 | 3.95 | 1.32 | **0.87** |
| CEPAE | 4.57 | 4.54 | 1.23 | **1.04** |
| CaTSG | 1.40 | 1.40 | 2.74 | 2.74 |

Over replications 0–4 at 400 epochs, the mean √PEHE changes as follows:

| Method | Masked (CATE) | Legacy (outcome as input) | Δ |
|---|---|---|---|
| GANITE | **1.24** | 3.71 | −2.47 |
| CRN | **1.46** | 1.88 | −0.42 |
| CEPAE | 1.33 | 1.37 | −0.04 |
| CaTSG | 3.00 | 2.98 | +0.02 |

Masking is not uniformly better. CEPAE is essentially unaffected — its entropy
penalty already limits copying — and CaTSG is unaffected because its diffusion
prior dominates the outcome coordinate. GANITE's collapse under the legacy input
is the clearest case: it has no abduction mechanism, so copying the factual
outcome is simply wrong. CRN benefits from being forced to use covariates.

**Real cohorts.** LaLonde (NSW) ATE error, before and after masking (single
deterministic cohort; ridge 213 and nearest-neighbour 528 are unaffected because
they never saw the outcome):

| Method | Before | After |
|---|---|---|
| CEPAE | 197.8 | **61.8** |
| GANITE | 1484.9 | **120.1** |
| CRN | 1796.0 | **707.5** |
| CaTSG | 1636.5 | 1633.7 |
| ridge T-learner | 212.9 | 212.9 |
| nearest neighbour | 528.2 | 528.2 |

The paper's claim that all four neural estimators predict essentially no effect on
LaLonde is therefore an artefact of the leak for three of them; after the fix
CEPAE is the best estimator on the cohort and GANITE is second, while CaTSG still
predicts almost no effect.

**IHDP, same 30 replications, masked against legacy** (√PEHE):

| Method | Masked | Legacy | Δ |
|---|---|---|---|
| ridge T-learner | 3.230 | 3.230 | 0 |
| GANITE | **3.295** | 5.402 | −2.11 |
| CEPAE | 4.711 | 5.050 | −0.34 |
| CRN | 5.509 | 6.188 | −0.68 |
| nearest neighbour | 6.906 | 6.906 | 0 |
| CaTSG | 7.027 | 5.001 | +2.03 |

The masked ordering is ridge ≈ GANITE < CEPAE < CRN < nearest neighbour ≈ CaTSG.
GANITE moves from fourth to a statistical tie with ridge; CaTSG moves the other
way — its median is unchanged (2.91 versus 2.85) but a few masked replications are
catastrophic, so the mean rises by two units. The block is a 30-replication
corrected rerun; the full 100-replication rerun is still required for the paper's
primary table.

Masked results for all four real cohorts (ε_ATE; lower is better; single run):

| Method | LaLonde | Jobs | Oregon | NHEFS |
|---|---|---|---|---|
| CEPAE | **61.8** | 0.201 | **0.003** | 0.771 |
| GANITE | 120.1 | 0.049 | 0.125 | 1.051 |
| ridge T-learner | 212.9 | **0.047** | 0.090 | **0.077** |
| nearest neighbour | 528.2 | 0.101 | 0.062 | 0.112 |
| CRN | 707.5 | 0.137 | 0.100 | 0.617 |
| CaTSG | 1633.7 | 0.077 | 0.014 | 3.107 |

No method wins on all four cohorts. CEPAE leads on the two largest-effect
cohorts (LaLonde, Oregon); the ridge baseline leads on Jobs and NHEFS; GANITE is
within 0.002 of ridge on Jobs. The Oregon reference ATE is 0.09 (ED visits), so
the absolute errors are on that scale, not comparable to LaLonde dollars.

### 6.2 Stale shared-CRL pools (found during the final rerun)

`run_2080.sh` called `generate_synthetic_data.py` without `--force`, and the
generator skipped every pool that already existed. The 2080's shared-CRL pools
therefore predated the audit's generator fix (persistent latents, nonlinear
mixing): they were generated by the old nilpotent process and reused for the
`2d5deb9` results.

**Evidence.** The fresh local `npvar` pool has lag-1 latent autocorrelations
0.448, 0.584 and 0.285 — the corrected range 0.29–0.59 that the audit reports.
Its full-budget ten-seed MCC is 0.61–0.80, matching the audit's corrected pilot
(0.62–0.79). The stored `2d5deb9` rows give 0.84–0.88, matching the old
nilpotent generator's 0.86. The same skip affected every suite whose pool
predated the fix.

**Consequence.** The shared-CRL numbers in the paper (mean MCC ≈ 0.77; the
Friedman p-values) are from the stale pools. The 17-suite rerun now in progress
regenerates the pools and the rows; its numbers will replace them. A.9
Lotka–Volterra was already regenerated and is unaffected. The single-method
blocks (`child`, `sde`, `voronoi`) use the same generators and need checking
against their pool timestamps.

### 6.3 Pool-reset defect (see §3.3)

### 6.4 Silent divergence and pool integrity (from the 2026-10 rerun)

Already fixed and tested in `7d74bcc`: the trainer raises after sustained
non-finite losses, the shared runner records a diverged run as a failure instead
of storing NaN, the generator regenerates non-finite pools, and
`scripts/verify_outputs.py` audits pools and results.

### 6.5 CHiLD KL term was broadcast per timestep

The CHiLD loss computed `(log_q - prior.unsqueeze(1)).mean()` with `log_q` of shape
`(B, T)` and `prior` already summed over the trajectory, shape `(B,)`. Broadcasting
subtracted the whole-trajectory log-prior from every timestep, so the "KL" term was
not a KL divergence and could become negative (observed: −69.85 during training,
which the `kl_weight` then rewarded). Fixed to
`kl = (log_q.sum(dim=1) - prior).mean()`; the same random batch now gives a
positive, finite value. CHiLD must be rerun; its stored block is invalid.

### 6.6 CLI: `--only` did not filter suite methods

`--only` filtered the grid runner but not `run_suite`, so a targeted rerun of one
method silently ran all five. Fixed and used throughout this document.

---

> **Task register.** The remaining work, with commands and acceptance criteria, is tracked in `docs/OPEN_TASKS.md`.

## 7. What remains

1. **Static rerun with masking for ACIC, Twins and News.** IHDP (30
   replications) and the four real cohorts are final under the masked protocol.
   ACIC and Twins need 100 masked replications each and News 50; on the M4 this
   is roughly 25 GPU-hours.
2. **CRL rerun with the pool reset.** The reset changes means by at most 0.039 MCC
   and can swap adjacent ranks; the qualitative grouping is stable, but the
   paired tests in the paper are not exactly paired until the block is rerun.
   A.9 is already post-reset; the other 17 suites are not.
3. **Grid rerun with the reset?** The grid reproduces its published contrast
   exactly on the corrected generator, but it predates the reset. The reset effect
   on the grid is unknown (it is bounded by the shared-suite measurement).
4. **Wall-clock times.** All `seconds` are NaN under parallel runs; a sequential
   rerun with `--keep-timing` is needed for the Time columns of the paper.
5. **Rendered-image Causal3DIdent**, the remaining II.D/adjacent implementations
   and the optional CPS panel remain unimplemented.

---

## 8. Per-suite detail (CRL, current rows)

Mean MCC over ten seeds. The winner follows the mechanism, as the methods'
assumptions predict; iVAE, the only stationary method, trails on every nonlinear
process except Ornstein–Uhlenbeck.

| Suite | iVAE | LEAP | TDRL | NCTRL | MOSAIC | Winner |
|---|---|---|---|---|---|---|
| npvar (A.1) | 0.54 | 0.72 | **0.75** | 0.72 | 0.70 | TDRL |
| npvar_linear (A.1b) | 0.80 | 0.83 | **0.85** | 0.82 | 0.80 | TDRL |
| npvar_hard (A.2) | 0.45 | **0.51** | **0.51** | 0.48 | 0.49 | LEAP/TDRL |
| cartpole (A.3) | 0.26 | 0.64 | 0.59 | **0.72** | 0.54 | NCTRL |
| causal3d (A.4) | 0.86 | 0.87 | 0.89 | 0.91 | **0.96** | MOSAIC |
| lorenz (A.5) | 0.41 | 0.70 | 0.59 | **0.77** | 0.67 | NCTRL |
| lorenz96 (A.6) | 0.39 | 0.59 | 0.54 | **0.64** | 0.58 | NCTRL |
| pcl (A.7) | 0.36 | 0.64 | 0.59 | **0.66** | 0.58 | NCTRL |
| kuramoto (A.8) | 0.45 | **0.65** | 0.63 | 0.64 | 0.63 | LEAP |
| lotka_volterra (A.9) | 0.44 | **0.72** | 0.65 | 0.69 | 0.66 | LEAP |
| duffing (A.10) | 0.76 | **0.90** | **0.90** | 0.85 | 0.81 | LEAP/TDRL |
| fitzhugh_nagumo (A.11) | 0.73 | **0.92** | **0.92** | 0.88 | 0.81 | LEAP/TDRL |
| chua (A.12) | 0.55 | 0.72 | 0.67 | **0.81** | 0.63 | NCTRL |
| van_der_pol (A.13) | 0.76 | 0.83 | 0.82 | **0.90** | 0.81 | NCTRL |
| pendulum (A.14) | 0.79 | 0.79 | 0.85 | **0.87** | 0.85 | NCTRL |
| mountain_car (A.15) | 0.76 | 0.93 | **0.97** | **0.97** | 0.83 | NCTRL/TDRL |
| varma (A.16) | 0.40 | 0.86 | 0.87 | 0.87 | **0.97** | MOSAIC |
| ou (A.17) | 0.74 | 0.87 | 0.77 | **0.88** | 0.84 | NCTRL |

Two suites changed far more than the rest when the stale pools were replaced:
A.1 (TDRL 0.75 instead of a five-way tie at 0.86) and Temporal Causal3DIdent
(MOSAIC 0.96 and iVAE 0.86, against 0.70 and 0.46 on the stale pool).

## 9. SDD and adjacent blocks (already final)

SDD (ten seeds): oracle OOD MSE 0.662; SYNC 1.927, DANN 1.927, VREx 1.927, ERM
1.928 — statistically indistinguishable, as the paper states. ID MSE 0.046–0.047
against the oracle's 0.374; MIG 0.100–0.121; CDS 0.576–0.611. The paper's values
(SYNC 1.966, DANN 1.969, VREx 1.978, ERM 1.977; MIG 0.11–0.12; CDS 0.58–0.61)
agree to within the aggregation used here.

Adjacent: GBTPL path-F1 0.805, source accuracy 0.197, zero structural violations
(paper: 0.79, 0.18, none). CausalConceptTS faithfulness 0.044 (paper 0.04).
CausalGAN CF MAE 0.414 (paper 0.41). LacaDM hypervolume **4.79**, not the
pre-audit 8.43: the paper's red caveat is correct and the corrected value should
replace it.

---

## 10. Reproduction

```bash
# environment
export GCMTS_PREFETCH=1 GCMTS_LIGHT_BATCH=1

# A.9 Lotka-Volterra (MPS)
uv run python scripts/run_crl_shared.py --device mps --suite lotka_volterra \
    --seeds 10 --epochs-scale 1.0 --data-source pool --dataset-dir data/synthetic \
    --scheduler adaptive --early-stopping-patience 300 --ema-decay 0.99 \
    --kl-warmup-epochs 50 --out-dir outputs_shared

# static CATE rerun under the masked protocol
uv run python scripts/run_ctee_static.py --device mps --dataset ihdp \
    --replications 100 --out-dir outputs

# real cohorts (one replication each)
for d in lalonde_rct jobs_rct nhefs oregon_rct; do
  uv run python scripts/run_ctee_static.py --device mps --dataset "$d" \
      --replications 1 --out-dir outputs
done

# static-masking A/B
uv run python scripts/diagnose_static_masking.py --device mps --reps 0 1 2 3 4 \
    --epochs-scale 0.5 --out docs/diagnostics_masking.json

# statistics and reference floors
uv run python scripts/statds_report.py --outputs outputs --shared outputs_shared \
    --out docs/statistical_analysis.md --json-out docs/statistical_analysis.json
uv run python scripts/reference_floors.py --outputs outputs

# audit (exit 1 on any FAIL)
uv run python scripts/verify_outputs.py --pools --report docs/audit_2026-10.md
```

The p-values in `docs/statistical_analysis.md` are recomputed with the exact
χ² tail; `docs/statistical_analysis.json` keeps the raw StaTDS value for
comparison.
