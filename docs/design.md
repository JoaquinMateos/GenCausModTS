# gcmts — Implementation Plan

This document describes the `gcmts` library and how it grows into a full
benchmark suite matching the review paper.

## 0. Status

Core is implemented and all five **Identifiable CRL sub-taxonomies** are covered:

| Sub-taxonomy | Method | Module | Notes |
| --- | --- | --- | --- |
| Foundations | iVAE | `methods/ivae.py` | conditional prior `p(z\|u)` |
| Temporal & Sequential | LEAP | `methods/leap.py` | temporal iVAE, causal-process prior |
| Temporal & Sequential | TDRL | `methods/tdrl.py` | fix/changing/observation latent blocks |
| Temporal & Sequential | NCTRL | `methods/nctrl.py` | HMM regimes, exact forward-algorithm marginalisation |
| Temporal & Sequential | Slow Flows | `methods/slow_flows.py` | volume-preserving RealNVP + SFA prior |
| Intervention-Conditioned | CITRIS | `methods/citris.py` | invertible AE + intervention prior + adversarial shared group |
| Latent Hierarchical | MOSAIC | `methods/mosaic.py` | sparse additive decoder, ANOVA module supports |
| Continuous-Time SDEs | CEGEN | `methods/cegen.py` | Deep Euler + conditional W2 (Bures) |

Run `uv run python scripts/taxonomy_coverage.py` to check coverage against the
LaTeX taxonomy table (CRL: 5/5).

Core building blocks: `core/backbones.py` (MLP, GaussianHead, ExpFamilyPrior,
RealNVP, GradientReversal, context helpers), `core/graph_ops.py` (DAG sampling,
NOTEARS acyclicity, SHD), `core/solvers.py` (Euler–Maruyama, Bures/W2),
`core/trainer.py` (`SimpleTrainer` with logging).

## 0b. Simple benchmark results

`scripts/run_crl_benchmarks.py` trains each method on a matching synthetic
benchmark and evaluates `MCC` / `R²_diag` (matched) with `CrlBenchmarkRunner`,
logging to `outputs/crl_benchmarks.log` and saving JSON/CSV.

Batch 256, RTX 3050 (`uv run python scripts/run_crl_benchmarks.py --batch-size 256 --device {cuda,cpu}`):

| Case | Method | MCC | R²_diag | CPU s | GPU s |
| --- | --- | ---: | ---: | ---: | ---: |
| `ivae_linear` | IVAE | 0.646 | 0.427 | 55.7 | 61.9 |
| `ivae_nonlinear` | IVAE | 0.416 | 0.221 | 42.0 | 50.4 |
| `leap_temporal` | LEAP | 0.286 | 0.125 | 109.3 | 38.7 |
| `slow_flows` | Slow Flows | 0.996 | 0.992 | 178.5 | 55.5 |
| `citris_interventional` | CITRIS | 0.711 | 0.526 | 144.8 | 68.4 |
| `tdrl_modular` | TDRL | 0.197 | 0.061 | 190.6 | 73.0 |
| `nctrl_regimes` | NCTRL | 0.402 | 0.189 | 220.1 | 118.0 |
| `mosaic_modules` | MOSAIC | 0.409 | 0.220 | 182.7 | 138.0 |
| `cegen_sde` | CEGEN | W2 0.0014 | — | 125.2 | 114.7 |

All input/output shape and finiteness checks pass; `R²_diag ≈ MCC²` confirms the
two metrics are mutually consistent. CITRIS's transition loop is vectorised over
time (189→68 s GPU). At batch 256 the GPU is ~1.7× faster overall (≈720 s vs
≈1250 s), with the largest gains on the deep/sequential methods; the small
feed-forward iVAE is roughly break-even.

## 1. Guiding principles

1. **One abstract base per paradigm.** Every concrete method subclasses
   `BaseCausalRepresentationLearner`, `BaseEffectEstimator`, or
   `BaseStaticDynamicDisentangler`. This keeps the public interface uniform.
2. **Causal-level honesty.** Each base exposes a `causal_level()` and every
   benchmark runner only reports metrics that match that level (L1/L2/L3).
3. **Identifiability documentation.** Every method implements
   `identifiability_statement()` stating assumptions and ambiguity class.
4. **Metric-to-ground-truth rule.** A metric is only computed when the
   synthetic benchmark provides the corresponding ground truth.
5. **Reusable core.** `TSCM`/`LTSCM`, adjacency tensor ops, Euler–Maruyama,
   diffusion schedules, and flow bijectors live in `core` and are reused by
   multiple paradigms.

## 2. Layered architecture

```
gcmts/
├── core/
│   ├── base.py          # BaseModel, BaseTrainer
│   ├── tscm.py          # TSCM, LTSCM (forward simulators)
│   ├── backbones.py     # MLP, GaussianHead, ExpFamilyPrior, RealNVP, GradientReversal
│   ├── graph_ops.py     # DAG constraints, NOTEARS-style acyclicity, SHD
│   ├── solvers.py       # Euler–Maruyama, Bures/W2
│   ├── trainer.py       # SimpleTrainer (logging, batching, device)
│   └── utils.py         # MCC, Hungarian matching, MLP factory
├── causal_representation_learning/
│   ├── base.py
│   ├── methods/         # implemented: iVAE, LEAP, TDRL, NCTRL, SlowFlows,
│   │                    #   CITRIS, MOSAIC, CEGEN
│   │                    # planned: CtrlNS, IDOL, DMM, SNICA, iCITRIS,
│   │                    #   hierarchical discovery, APPEX
│   └── losses/          # (planned) ELBO, sparsity, contrastive
├── effect_estimation/
│   ├── base.py
│   ├── methods/         # CaTSG, Wu-IPTW, LCD/CLIPR, PFD-BDCM, CaPaint,
│   │                    #   LacaDM, CRN, CausalTransformer, GANITE, CEPAE
│   └── losses/          # diffusion score losses, IPW reweighting, axioms
├── static_dynamic_disentanglement/
│   ├── base.py
│   ├── methods/         # SYNC, DAG-VAE, GCIM, UDA, CaDRe
│   └── losses/          # MI penalties, domain-confusion, sparsity
├── data/
│   ├── synthetic.py     # VAR/NP, Voronoi/Causal3DIdent-like, harmonic,
│   │                    #   market/SDE simulators
│   ├── real.py          # MIMIC, Climate, scRNA-seq loaders (thin wrappers)
│   └── transforms.py    # normalisation, missingness, windowing
└── evaluation/
    ├── metrics.py       # MCC, R², SHD, DCI, triplet, PEHE, ATE, CF MAE,
    │                    #   MMD, Wasserstein, J-FTSD, axioms, OOD MSE, MIG
    ├── benchmark.py     # Generic + specialised runners
    ├── report.py        # LaTeX/markdown tables, PRISMA-style summary
    └── significance.py  # Paired tests, confidence intervals
```

## 3. Per-method contract

Every method file (e.g. `crl/methods/leap.py`) must contain:

- A docstring with the BibTeX key and a one-sentence summary.
- A class inheriting the paradigm base.
- `forward(batch) -> Output` returning all tensors needed for loss and
  evaluation.
- `loss(outputs, batch) -> dict[str, Tensor]` with a `'loss'` total.
- `identifiability_statement() -> str`.
- A unit test (see `tests/test_methods.py`, `tests/test_methods_extended.py`).

## 4. Evaluation roadmap

Phase A — Representation recovery:
- `MCC`, `R²_diag`, `R²_sep`, `DCI`, `SHD` on `LTSCMGenerator` and
  Causal3DIdent-like generators.

Phase B — Counterfactual accuracy:
- `PEHE`, `ε_ATE`, `CF MAE/MBE`, `MMD²`, `Wasserstein`, `J-FTSD`, and axiom
  checks (reconstruction, effectiveness, reversibility) on harmonic oscillator
  and simulated clinical trajectories.

Phase C — Cross-domain generalisation:
- `OOD MSE/MAE`, domain-identification accuracy, `MIG`/`CDS` on regime-switching
  and spatio-temporal graph benchmarks.

Phase D — Statistical reporting:
- Bootstrap confidence intervals, paired Wilcoxon tests, and aggregated
  markdown/LaTeX tables generated by `evaluation/report.py`.

## 5. Next concrete steps

Done:
- Core: `backbones.py`, `graph_ops.py`, `solvers.py`, `trainer.py`.
- Methods iVAE, LEAP, TDRL, NCTRL, Slow Flows, CITRIS, MOSAIC and CEGEN, all
  wired to `CrlBenchmarkRunner` (all five CRL sub-taxonomies covered).
- `data/synthetic.py` — nonlinear ICA, temporal nonlinear, interventional,
  TDRL/NCTRL models, slow-feature and linear-SDE generators.
- Benchmark and taxonomy-coverage scripts; tests; mypy-strict and ruff clean.

Next:
1. Second representatives per sub-taxonomy: iCITRIS (instantaneous DAG), APPEX
   (linear-SDE identification from snapshots), and a hierarchical method.
2. Counterfactual and treatment-effect methods (paradigm II).
3. Static-dynamic disentanglement methods (paradigm III).
4. Real-data loaders and CI-style regression tests.

## 6. Python 3.14 opportunities

- Use `typing.TypeAliasType` and modern generic syntax where it improves
  readability.
- Keep runtime compatibility with the versions supported by PyTorch; do not
  use experimental syntax that tooling cannot parse.
- The `pyproject.toml` pins `requires-python = ">=3.14"` so the library can
  rely on `tomllib`, improved `typing`, and `asyncio` improvements.
