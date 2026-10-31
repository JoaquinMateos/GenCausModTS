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
| CRL A.9 Lotka–Volterra | **(running)** | 50 | `outputs_shared/crl_shared_lotka_volterra.json` |
| CTEE static NEWS (50 replications) | Complete | 300 | `outputs/ctee_static_news.json` |
| CTEE real Oregon | Complete, **needs rerun** after the masking fix | 6 | `outputs/ctee_static_oregon_rct.json` |
| Static-masking A/B (IHDP reps 0–4) | Complete | 40 | `docs/diagnostics_masking.json` |
| CRL reset check (Duffing, 10 seeds) | **(running)** | 50 | scratch |
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
reset and share this confound. The reset changes individual cells by up to 0.05
MCC (Duffing, seed 0, NCTRL). The statistical analysis in §4 is computed on those
pre-reset rows and must be regenerated after the CRL rerun; the conclusions that
depend on it are marked accordingly.

---

## 4. Statistical comparison (StaTDS)

Friedman test per block, with Nemenyi critical distance and Holm post-hoc against
the best-ranked method. A case is a benchmark (CRL), a replication (static CATE),
or a panel (real data).

### 4.1 CRL representation quality (MCC, higher better)

17 shared suites (A.9 excluded until its rerun finishes).

| Method | Mean rank | Mean MCC | Median |
|---|---|---|---|
| NCTRL | 2.06 | 0.775 | 0.791 |
| LEAP | 2.24 | 0.759 | 0.788 |
| TDRL | 2.82 | 0.735 | 0.773 |
| MOSAIC | 3.06 | 0.732 | 0.802 |
| iVAE | 4.82 | 0.580 | 0.539 |

Friedman χ² = 32.85, p = 1.28×10⁻⁶. Holm: NCTRL beats only iVAE
(adjusted p = 1.4×10⁻⁶); NCTRL–LEAP (p = 0.75), NCTRL–TDRL (p = 0.32) and
NCTRL–MOSAIC (p = 0.20) are not distinguishable. **The four
temporally-aware methods form one group; iVAE is separated.** This is exactly the
qualitative prediction of the methods' own assumptions: iVAE assumes stationary
i.i.d. latents and has no mechanism for time.

### 4.2 CRL one-step forecasting (MSE, lower better)

| Method | Mean rank | Mean MSE | Median |
|---|---|---|---|
| NCTRL | 1.88 | 0.328 | 0.010 |
| LEAP | 1.94 | 0.325 | 0.009 |
| TDRL | 2.76 | 0.463 | 0.009 |
| MOSAIC | 3.82 | 0.570 | 0.017 |
| iVAE | 4.59 | 0.567 | 0.021 |

Friedman χ² = 38.26, p = 9.91×10⁻⁸; NCTRL–iVAE and NCTRL–MOSAIC survive Holm.
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

### 4.4 Static CATE (√PEHE, lower better)

| Panel | Best | Worst | Friedman p |
|---|---|---|---|
| IHDP (legacy input, being rerun) | ridge_t 2.397 | nearest_neighbour 5.475 | 1.2×10⁻¹⁰ |
| ACIC (legacy input) | crn 0.311 | ganite 1.066 | 1.2×10⁻¹⁰ |
| Twins (legacy input) | catsg 0.319 | nearest_neighbour 0.536 | 1.6×10⁻¹⁰ |
| NEWS (legacy input) | cepae 3.432 | ganite 12.728 | 9.7×10⁻¹¹ |

The ACIC, Twins and NEWS rows predate the masking fix; the IHDP rerun under the
masked protocol is in progress (30 replications). The first replications show the
masked ordering ridge < CRN ≈ CEPAE < nearest-neighbour < GANITE < CaTSG, i.e.
CRN moves from last to second, but the block is not final.

On IHDP, ridge beats every neural method (Holm p < 10⁻⁸ for all five). This is the
opposite of the published IHDP ordering (§5) and is the signature of the outcome
leak identified in §6.1. These rows will be replaced by the masked rerun.

### 4.5 Reference floors (the lens for §4.4)

For every static panel, two predictors that use no heterogeneity are computed per
replication: predict zero effect (√PEHE = RMS of the true effect) and predict the
true ATE constant (√PEHE = SD of the true effect). A method that does not beat the
constant-ATE floor has learned no heterogeneity.

| Panel | Zero-effect floor | Constant-ATE floor | Best method | Beats constant-ATE |
|---|---|---|---|---|
| IHDP | 7.96 | 5.64 | ridge 2.40 | 87% of replications |
| ACIC | 0.199 | 0.053 | crn 0.311 | 7% |
| Twins | 0.319 | 0.319 | catsg 0.319 | 0% |
| NEWS | 4.33 | 3.33 | cepae 3.43 | 4% |

Three consequences:

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
   consistently (87% of replications).

---

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

### 6.2 Pool-reset defect (see §3.3)

### 6.3 Silent divergence and pool integrity (from the 2026-10 rerun)

Already fixed and tested in `7d74bcc`: the trainer raises after sustained
non-finite losses, the shared runner records a diverged run as a failure instead
of storing NaN, the generator regenerates non-finite pools, and
`scripts/verify_outputs.py` audits pools and results.

### 6.4 CLI: `--only` did not filter suite methods

`--only` filtered the grid runner but not `run_suite`, so a targeted rerun of one
method silently ran all five. Fixed and used throughout this document.

---

## 7. What remains

1. **Static rerun with masking** (IHDP, ACIC, Twins, NEWS, LaLonde/NHEFS/Oregon).
   This is required before any static-CATE number enters the paper.
2. **CRL rerun with the pool reset.** The current CRL table has an order
   confound; the reset changes cells by up to 0.05 MCC.
3. **A.9 Lotka–Volterra** (running) and its inclusion in the 18-suite statistics.
4. **Grid rerun?** The grid reproduces its published contrast exactly, but it was
   produced without the pool reset. The reset effect on the grid is unknown.
5. **Wall-clock times.** All `seconds` are NaN under parallel runs; a sequential
   rerun with `--keep-timing` is needed for the Time columns.
6. **Rendered-image Causal3DIdent** and the remaining II.D/adjacent
   implementations are still unimplemented.

---

## 8. Per-suite detail (CRL, current rows)

Mean MCC over ten seeds. The winner follows the mechanism, as the methods'
assumptions predict; iVAE, the only stationary method, trails on every nonlinear
process except Ornstein–Uhlenbeck.

| Suite | iVAE | LEAP | TDRL | NCTRL | MOSAIC | Winner |
|---|---|---|---|---|---|---|
| npvar (A.1) | 0.86 | 0.88 | 0.87 | 0.84 | 0.87 | tie |
| npvar_linear (A.1b) | 0.80 | 0.83 | 0.86 | 0.79 | 0.80 | TDRL |
| npvar_hard (A.2) | 0.38 | 0.50 | 0.49 | 0.47 | 0.49 | LEAP |
| cartpole (A.3) | 0.26 | 0.61 | 0.57 | 0.67 | 0.55 | NCTRL |
| causal3d (A.4) | 0.46 | 0.68 | 0.67 | 0.69 | 0.70 | MOSAIC |
| lorenz (A.5) | 0.41 | 0.70 | 0.58 | 0.75 | 0.67 | NCTRL |
| lorenz96 (A.6) | 0.40 | 0.60 | 0.54 | 0.65 | 0.57 | NCTRL |
| pcl (A.7) | 0.36 | 0.65 | 0.59 | 0.66 | 0.58 | NCTRL |
| kuramoto (A.8) | 0.45 | 0.65 | 0.63 | 0.62 | 0.62 | LEAP |
| duffing (A.10) | 0.78 | 0.91 | 0.88 | 0.89 | 0.81 | LEAP |
| fitzhugh_nagumo (A.11) | 0.73 | 0.90 | 0.89 | 0.90 | 0.81 | LEAP/NCTRL |
| chua (A.12) | 0.54 | 0.72 | 0.64 | 0.77 | 0.64 | NCTRL |
| van_der_pol (A.13) | 0.78 | 0.85 | 0.82 | 0.91 | 0.81 | NCTRL |
| pendulum (A.14) | 0.79 | 0.79 | 0.85 | 0.87 | 0.85 | NCTRL |
| mountain_car (A.15) | 0.76 | 0.93 | 0.96 | 0.97 | 0.84 | NCTRL |
| varma (A.16) | 0.36 | 0.86 | 0.89 | 0.87 | 0.98 | MOSAIC |
| ou (A.17) | 0.75 | 0.87 | 0.77 | 0.87 | 0.88 | MOSAIC |

Two deviations from the paper's pre-audit text: physics-coupled Lorenz (A.7) is
now led by NCTRL (0.66) rather than LEAP (0.65), a difference of one point, and
iVAE is no longer competitive on Ornstein–Uhlenbeck (0.75 versus 0.88).

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
