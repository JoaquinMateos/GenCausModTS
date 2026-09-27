# gcmts — Generative Causal Modelling for Time Series

`gcmts` is a research-grade Python library for **generative causal modelling of
time series**. It implements the most important algorithms for (i) learning
*identifiable* causal representations from sequential data, (ii) estimating
counterfactual and treatment effects under time-varying treatments, and (iii)
disentangling static from dynamic causal factors for robust generalisation.

The library accompanies the manuscript *"Generative Causal Modelling in Time
Series: An Experimental Review and Benchmark"* and follows a single organising
principle:

> **The choice of dataset is dictated by the causal quantity the model claims to
> recover, and the choice of metric is dictated by the level of Pearl's causal
> hierarchy the model claims to answer.**

Every method therefore declares the causal level it targets and the ambiguity
class under which its solution is identifiable, and every evaluation uses only
metrics whose ground truth actually exists.

---

## Table of contents

1. [Background](#background)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Architecture](#architecture)
6. [Implemented methods](#implemented-methods)
7. [Evaluation](#evaluation)
8. [Benchmarks](#benchmarks)
9. [Taxonomy coverage](#taxonomy-coverage)
10. [Extending the library](#extending-the-library)
11. [Reproducibility](#reproducibility)
12. [Limitations and roadmap](#limitations-and-roadmap)
13. [Citation](#citation)
14. [License and third-party notices](#license-and-third-party-notices)

---

## Background

### The three paradigms

`gcmts` mirrors the three paradigms of the companion review:

| Paradigm | Goal | Causal level | Module |
| --- | --- | --- | --- |
| **Identifiable CRL** | Recover latent causal variables/mechanisms from `x = g(z)` up to an ambiguity class | L1 / L2 | `causal_representation_learning` |
| **Effect estimation** | Estimate `P(X \| do(A=a))` and counterfactual trajectories | L2 / L3 | `effect_estimation` |
| **Static–dynamic disentanglement** | Separate invariant from varying causal factors | L1 (generalisation) | `static_dynamic_disentanglement` |

### Latent temporal structural causal model

The shared generative model is the *latent temporal SCM* (L-TSCM):

```
z_{i,t} = f_i( Pa(z_{i,t}), u_t, ε_{i,t} ),      ε_{i,t} ~ p_ε
x_t     = g( z_t, η_t ),                            η_t     ~ p_η
```

where `Pa(·)` are the causal parents within a maximum lag `p`, `u_t` is an
exogenous context (regime, domain, or intervention), `g` is an unknown mixing
map, and the contemporaneous subgraph is a DAG. `TSCM` and `LTSCM` in
`gcmts.core.tscm` implement forward simulation, noise re-sampling, and
interventions for this model.

### Pearl's hierarchy

1. **L1 — Association:** `P(X_{1:T} | A_{1:T}=a)`.
2. **L2 — Intervention:** `P(X_{1:T} | do(A_{1:T}=a))` via the truncated
   factorisation (g-formula).
3. **L3 — Counterfactual:** `P(X'_{1:T} | X_{1:T}=x, A=a, A'=a')` via
   abduction → action → prediction.

A method is evaluated only at the level it claims. `BaseModel.causal_level()`
reports this, and `CrlBenchmarkRunner` / `EffectBenchmarkRunner` /
`DisentanglementBenchmarkRunner` enforce it.

---

## Features

- **One abstract base per paradigm** — `BaseCausalRepresentationLearner`,
  `BaseEffectEstimator`, `BaseStaticDynamicDisentangler` — with a uniform
  `forward` / `loss` / `identifiability_statement` / `ambiguity_class` contract.
- **Reusable core** — MLPs and Gaussian heads, an exponential-family conditional
  prior, invertible affine-coupling flows (RealNVP), gradient reversal, DAG
  sampling and the NOTEARS acyclicity constraint, Euler–Maruyama integration, and
  the closed-form Bures/Wasserstein-2 distance.
- **Reproducible synthetic benchmarks** — generators that reproduce each
  method's generative assumptions, with ground-truth latents, graphs,
  intervention targets, and regimes.
- **Evaluation at the right level** — a metric registry (`MCC`, `R²_diag`,
  `SHD`, `PEHE`, `ATE`, `CF MAE`, `OOD MSE`, …) plus benchmark runners.
- **Logging trainer** — `SimpleTrainer` with deterministic batching, device
  selection (CPU/CUDA/MPS), gradient clipping, linear warmup + cosine decay
  (`cosine`/`plateau`/`step`/`none`), held-out validation monitoring with early
  stopping, and per-epoch history.
- **Training curves** — `save_training_report` writes `history.json`/`.csv` and
  a loss/terms/learning-rate figure to `outputs/<experiment>/`.
- **Typed and tested** — `mypy --strict` and `ruff` clean; a `pytest` suite
  covering flows' invertibility, graph operators, solvers, and every method's
  forward/loss/gradients plus latent-recovery sanity checks.

---

## Installation

Python **3.14** is required.

```bash
# with uv (recommended)
uv add "git+https://github.com/JoaquinMateos/GenCausModTS"

# with pip
pip install "git+https://github.com/JoaquinMateos/GenCausModTS"
```

For development:

```bash
git clone https://github.com/JoaquinMateos/GenCausModTS
cd gcmts
uv sync --extra dev
```

---

## Quick start

Train iVAE on a nonlinear-ICA benchmark and score latent recovery:

```python
from gcmts.causal_representation_learning.methods import IVAE
from gcmts.core import SimpleTrainer
from gcmts.data.synthetic import NonlinearICAGenerator
from gcmts.evaluation import BenchmarkTask, CrlBenchmarkRunner

# regime-conditioned nonlinear ICA data with ground-truth latents
data = NonlinearICAGenerator(
    observed_dim=6, latent_dim=3, horizon=1, n_regimes=3, mixing="linear", seed=0
)
model = IVAE(observed_dim=6, latent_dim=3, u_dim=3, obs_noise=0.1)

# warmup + cosine schedule, validation monitoring, per-epoch history
history = SimpleTrainer(
    max_epochs=2000, batch_size=256, val_size=512, scheduler="cosine", seed=0
).fit(model, data)

task = BenchmarkTask(
    name="ivae", model=model, data=data.sample(2048),
    metrics=["mcc", "r2_diag"], causal_level="L1_association",
)
print(CrlBenchmarkRunner().run(task).to_dict())
```

---

## Architecture

```
src/gcmts/
├── core/                              # shared, paradigm-agnostic building blocks
│   ├── base.py                        # BaseModel, BaseTrainer, device resolution
│   ├── tscm.py                        # TSCM, LTSCM forward simulators
│   ├── backbones.py                   # MLP, GaussianHead, ExpFamilyPrior, RealNVP, GRL
│   ├── graph_ops.py                   # sample_dag, acyclicity (NOTEARS), SHD, topo-sort
│   ├── solvers.py                     # euler_maruyama, bures_w2 / gaussian_w2
│   ├── trainer.py                     # SimpleTrainer (logging, batching, move_batch)
│   └── utils.py                       # MCC, Hungarian matching
├── causal_representation_learning/    # base.py + methods/
├── effect_estimation/                 # base.py + methods/ (interface + stubs)
├── static_dynamic_disentanglement/    # base.py + methods/ (SYNC, DANN, ERM)
├── data/                              # synthetic generators and loaders
├── evaluation/                        # metrics registry + benchmark runners
└── typing.py                          # Batch, outputs, ambiguity classes
```

All methods inherit from the base of their paradigm and are trainable through
`SimpleTrainer`; all data generators return a `Batch(x, z, context, mask)` that
carries ground-truth latents and context (regimes `u`, intervention targets `I`).

---

## Implemented methods

The five **Identifiable-CRL sub-taxonomies** of the review are covered.

| Sub-taxonomy | Method | Ambiguity class | Reference |
| --- | --- | --- | --- |
| Foundations | iVAE | permutation + component-wise invertible | Khemakhem et al., 2020 |
| Temporal & sequential | LEAP | permutation + component-wise invertible | Yao et al., 2021 |
| Temporal & sequential | TDRL | permutation + component-wise invertible | Yao et al., 2022 |
| Temporal & sequential | NCTRL | permutation + component-wise invertible | Song et al., 2023 |
| Temporal & sequential | Slow Flows | linear demixing | Pineau et al., 2020 |
| Intervention-conditioned | CITRIS | element-wise invertible | Lippe et al., 2022 |
| Latent hierarchical & module | MOSAIC | permutation of module supports | Fan et al., 2026 |
| Continuous-time SDEs | CEGEN | exact drift/volatility (no mixing) | Remlinger et al., 2021 |

### iVAE — identifiable VAE
Learns `x = g(z) + η` with an auxiliary-conditioned, factorised exponential-family
prior `p(z|u) = ∏_i p(z_i|u)`. Trained by maximising the ELBO. Identifiable up to
permutation and component-wise invertible transforms when `u` has sufficient
variability.

### LEAP — temporal iVAE
Replaces the iVAE prior with a non-stationary causal-process prior
`p(z_t | z_{t-p:t-1}, u_t)`. Identifiability follows from linear independence of
the log-density gradients across regimes.

### TDRL — factorised temporal representation
Splits the latent space into a fixed-dynamics block, a transition-shift block
(conditioned on a domain factor), and an observation-shift block generated from a
domain factor independently of the past (`x = g(z)`). Recovers each block up to
permutation and component-wise transforms.

### NCTRL — unknown regimes
Latent regimes `c_t` follow a first-order hidden Markov model and the transition
is regime-dependent, `p(z_t | z_{t-1}, c_t)`. The discrete regimes are
marginalised exactly with the forward algorithm, giving component-wise
identifiability from observations alone.

### Slow Flows — time-series source separation
A volume-preserving normalizing flow maps observations to sources, with a slow
feature prior on the temporal increments `Δz_t = z_t − z_{t-1} ~ N(0, I)`. The
sources are identified up to a linear demixing, resolvable by linear ICA.

### CITRIS — intervention-conditioned CRL
An invertible autoencoder maps observations to `M` latents assigned to `K` causal
variables plus a shared group; the transition prior is conditioned on binary
intervention targets `I`. Minimising the information in the shared group recovers
minimal causal variables up to element-wise invertible transforms.

### MOSAIC — module discovery
A sparse additive decoder `x_i = Σ_j g_{ij}(z_j) + b_i` discovers the ANOVA
main-effect supports `A_{ij} = 1{x_i depends on z_j}` up to permutation of the
latent modules, using an L1 support penalty and a regime-conditioned encoder.

### CEGEN — Deep Euler scheme
Models the process directly in observation space with the Euler step
`z_{t+1} = z_t + f_θ(z_t)Δt + G_θ(z_t)√Δt·ε_t` and trains it by matching
step-wise conditional transitions in state-space regions with the closed-form
Wasserstein-2 (Bures) distance.

Every method exposes its assumptions and ambiguity through
`identifiability_statement()` and `ambiguity_class()`.

---

## Evaluation

Metrics live in `gcmts.evaluation.metrics` and are indexed by causal level:

| Family | Metrics | Ground truth required |
| --- | --- | --- |
| Representation recovery | `MCC`, `R²_diag` | latent factors |
| Structural recovery | `SHD`, `WSHD` | latent graph |
| Counterfactual/treatment | `PEHE`, `ε_ATE`, `CF MAE`, `MMD²`, `J-FTSD` | counterfactuals |
| Generalisation | `OOD MSE/MAE`, domain-ID accuracy, `MIG`, `CDS` | held-out regimes |

`MCC` is invariant to permutation and monotone rescaling (the empirical
signature of identifiability); `R²_diag` matches components by correlation before
scoring. The benchmark runners move data to the model's device and compute only
the requested metrics.

---

## Benchmarks

```bash
uv run python scripts/run_crl_benchmarks.py                 # CPU
uv run python scripts/run_crl_benchmarks.py --device cuda   # GPU
uv run python scripts/run_crl_benchmarks.py --quick         # smoke run
uv run python scripts/run_crl_benchmarks.py --only leap_temporal
```

Each method is trained on the synthetic benchmark matching its assumptions;
input/output shapes are validated and the aggregate results are written to
`outputs/crl_benchmarks.{log,json,csv}`. Per-experiment training histories,
model checkpoints, and training-curve figures (`training_curves.png/pdf`) are
written to `outputs/<case>/`.

Preliminary results (`--batch-size 256`, single seed; RTX 3050):

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

Metrics are internally consistent (`R²_diag ≈ MCC²`). Slow Flows and CITRIS
recover strongly; iVAE, NCTRL and MOSAIC are moderate; TDRL is weaker but above
chance. CEGEN operates in observation space and is scored by its conditional W²
loss. On GPU the suite is ≈1.7× faster at batch 256.

---

## Taxonomy coverage

```bash
uv run python scripts/taxonomy_coverage.py            # report
uv run python scripts/taxonomy_coverage.py --strict   # non-zero if any row uncovered
```

The script parses the bundled taxonomy table (`docs/taxonomy_table.tex`) and
maps each sub-taxonomy to its implemented method(s). Identifiable CRL is fully
covered (5/5); the effect-estimation and disentanglement sub-taxonomies are
declared but not yet implemented.

---

## Extending the library

To add a method:

1. Subclass the paradigm base, e.g. `BaseCausalRepresentationLearner`.
2. Implement `forward(batch) -> CausalRepresentationOutput`,
   `loss(outputs, batch)`, `encode`, `decode`, `transition`,
   `identifiability_statement`, and `ambiguity_class`.
3. Add a synthetic generator that reproduces the method's generative
   assumptions, returning ground-truth `z` and any context (`u` / `I`).
4. Add a unit test (forward shapes, finite loss, gradient flow) and, where
   possible, a latent-recovery sanity check.
5. Register any new metric in `gcmts.evaluation.metrics` at the causal level it
   measures.

The design and roadmap are documented in [`docs/design.md`](docs/design.md).

---

## Reproducibility

- Fully synthetic benchmarks with fixed seed schedules; the latent variables,
  mechanisms, mixing map and graph are regenerated identically across methods.
- `uv.lock` is committed so environments can be reconstructed.
- Benchmark scripts log every run to `outputs/`.
- The evaluation computes only metrics whose ground truth exists (rules R1–R3).

---

## Limitations and roadmap

- **Under-tuned methods.** TDRL and MOSAIC are above chance but not yet tuned to
  their best attainable performance; MOSAIC's result varies with training length.
- **Single-seed preliminary results.** The reported table is a single-seed sanity
  check; the multi-seed protocol with confidence intervals and significance tests
  is planned.
- **Effect estimation and disentanglement** are specified through their abstract
  bases but not yet implemented.
- **Real-data loaders** (clinical, climate, single-cell) are on the roadmap.

---

## Citation

If you use this library, please cite the companion review:

```bibtex
@article{mateos2026gcmts,
  title   = {Generative Causal Modelling in Time Series:
             An Experimental Review and Benchmark},
  author  = {Mateos, Joaqu{\'i}n and Moya, Antonio R. and Ventura, Sebasti{\'a}n},
  year    = {2026},
  note    = {Code: \url{https://github.com/JoaquinMateos/GenCausModTS}}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

---

## License and third-party notices

`gcmts` is released under the **MIT License** (see [`LICENSE`](LICENSE)). MIT is
compatible with every license of the reference implementations and dependencies
used by the project:

- **Algorithms** are implemented from the published papers, not copied from the
  authors' repositories; the code in this library is original. The companion
  paper's reference implementations are cited and, where available, released under
  MIT (LEAP, TDRL, NCTRL), BSD-3-Clause-Clear (CITRIS), or are unlicensed
  (MOSAIC, consulted only for the algorithm description).
- **Direct dependencies** are permissively licensed: PyTorch (BSD-3-Clause),
  Lightning (Apache-2.0), scikit-learn/networkx/pandas/numpy/scipy
  (BSD-3-Clause), and pydantic (MIT).

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full list and
attribution.
