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
| **Static–dynamic** (CRL sub-branch I.B.2) | Separate invariant from varying causal factors | L1 (generalisation) | `static_dynamic_disentanglement` |

### Paradigm taxonomies

The five branches of **Causal Representation Learning**, by the primary source of
the identification constraint (I.A–I.E):

<img src="docs/figures/crl_taxonomy.png" alt="Taxonomy of Causal Representation Learning" width="620">

The four identification strategies of **Counterfactual & Treatment-Effect
Estimation** (II.A–II.D):

<img src="docs/figures/ctee_taxonomy.png" alt="Identification strategies for counterfactual and treatment-effect estimation" width="700">

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

<img src="docs/figures/ltscm.png" alt="Latent Temporal Structural Causal Model" width="760">

*Context `u_t`, latent `z_{i,t}` and observed `x_{i,t}` states unrolled over time;
`f` the latent mechanism, `g` the mixing (decoder) map, `ε` process noise and `η`
observation noise, with parents bounded by lag `p`.*

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

All five **CRL branches** of the review are covered.

| Branch | Method | Ambiguity class | Reference |
| --- | --- | --- | --- |
| I.A Temporal structure | Slow Flows | linear demixing | Pineau et al., 2020 |
| I.A Temporal structure | CEGEN | exact drift/volatility (no mixing) | Remlinger et al., 2021 |
| I.B Non-stationarity & environment shift | LEAP | permutation + component-wise invertible | Yao et al., 2021 |
| I.B Non-stationarity & environment shift | TDRL | permutation + component-wise invertible | Yao et al., 2022 |
| I.B Non-stationarity & environment shift | NCTRL | permutation + component-wise invertible | Song et al., 2023 |
| I.C Observed auxiliary variables | iVAE | permutation + component-wise invertible | Khemakhem et al., 2020 |
| I.D Recorded interventions | CITRIS | element-wise invertible | Lippe et al., 2022 |
| I.E Structural & functional constraints | MOSAIC | permutation of module supports | Fan et al., 2026 |

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

The **effect-estimation** paradigm implements CaTSG (score-guided diffusion),
CRN (counterfactual recurrent network), GANITE (adversarial ITE) and CEPAE
(conditional-entropy-penalised autoencoder); the **static–dynamic** sub-branch
(CRL I.B.2) implements SYNC (four-way factor split), DANN and VREx (invariant prediction),
with ERM and oracle baselines. See `effect_estimation/methods/` and
`static_dynamic_disentanglement/methods/`.

---

## Evaluation

Metrics live in `gcmts.evaluation.metrics` and are indexed by causal level:

| Family | Metrics | Ground truth required |
| --- | --- | --- |
| Representation recovery | `MCC`, `R²_diag`, `R²_sep`, `DCI` | latent factors |
| Structural recovery | `SHD`, `WSHD` | latent graph |
| Intervention recovery | `triplet`, `two_sample` | intervention targets |
| Counterfactual/treatment | `PEHE`, `ε_ATE`, `CF MAE/MBE`, `MMD²`, `J-FTSD` | counterfactuals |
| Generalisation | `OOD MSE/MAE`, domain-ID accuracy, `MIG`, `CDS` | held-out regimes |

`MCC` is invariant to permutation and monotone rescaling (the empirical
signature of identifiability); `R²_diag` matches components by correlation before
scoring. The benchmark runners move data to the model's device and compute only
the requested metrics.

Statistical comparison lives in `gcmts.evaluation.stats`: Holm-corrected paired
Wilcoxon signed-rank tests paired by `(problem, seed)`, the Friedman omnibus
test, and the Nemenyi post-hoc test with the critical difference
`CD = q_α √(k(k+1)/6N)` and Demsar-style CD diagrams.

Loaders with ground-truth causal quantities (rule R1): the tumour-growth
simulator (`data.semi_synthetic.TumourGrowthSimulator`), IHDP (100
replications), ACIC (hand-download layout), and the Voronoi/Causal Pinball SCM
(`data.voronoi.VoronoiGenerator`). A differentiable Causal3DIdent image
surrogate with known factors and interventions lives in
`data.causal3dident.py`, with CNN encoder/decoder backbones in
`core.backbones`.

---

## Benchmarks

```bash
uv run python scripts/download_datasets.py                  # real + IHDP archives
uv run python scripts/download_datasets.py --generate       # materialise Voronoi tensors
uv run python scripts/run_crl_benchmarks.py --device cuda   # per-method sanity check
uv run python scripts/run_crl_shared.py --suite npvar --device cuda --seeds 10
uv run python scripts/run_crl_shared.py --grid --device cuda --seeds 10
uv run python scripts/run_ctee_benchmarks.py --device cuda --seeds 10
uv run python scripts/run_sdd_benchmarks.py --device cuda --seeds 10
uv run python scripts/make_paper_tables.py                  # LaTeX tables from JSON
```

The whole suite — every benchmark family plus the LaTeX tables — is driven by
one orchestrator:

```bash
uv run python scripts/run_full_benchmark.py --device cuda --seeds 10          # everything
uv run python scripts/run_full_benchmark.py --soft --device cuda             # smoke run
```

These runs train small models whose cost is dominated by CPU-side data
generation, so a single process leaves the GPU almost entirely idle. The
orchestrator therefore dispatches the independent benchmark steps to a bounded
pool of worker processes that share the GPU (`--jobs N`; default auto from the
CPU count, `--jobs 1` restores the original strictly-sequential behaviour):

```bash
uv run python scripts/run_full_benchmark.py --device cuda --seeds 10 --jobs 8
```

On multi-GPU hosts `--gpus 0,1` pins steps round-robin via
`CUDA_VISIBLE_DEVICES`; the 18-problem grid is split into two balanced shards by
default (`--grid-shards 1` disables this). Every step writes exactly the same
files as a sequential run, and each step's stdout/stderr is captured under
`outputs/logs/`.

Because concurrent steps contend for CPU, their wall-clock `seconds` are no
longer comparable to a sequential run: the orchestrator therefore records
per-run timings as unavailable (the `Time (s)` column renders `--`) unless you
pass `--keep-timing`. To produce the `Time (s)` column for the paper, run once
with `--jobs 1` on an idle machine; thread counts are left untouched by default
(`--threads 0`) so the *metrics* stay numerically identical to a sequential run.

### Progress and ClearML tracking

#### Terminal progress

Every benchmark process writes a `<stem>_progress.json` file (`done/total`,
elapsed, ETA) next to its results. When `rich` is installed the orchestrator
renders a live dashboard from those files — one bar per step plus an overall
count and approximate remaining time — and falls back to a periodic plain log
otherwise (`--no-dashboard`, `--dashboard-interval`).

#### ClearML (optional web UI)

Install the extra once:

```bash
uv sync --extra tracking        # adds rich + clearml
```

**1. Have a server.** Use an existing ClearML instance, or self-host one on the
same box as the benchmark (the official Docker stack):

```bash
git clone https://github.com/allegroai/clearml-server.git
cd clearml-server
docker compose up -d            # API :8008, Web :8080, Files :8081
```

**2. Point the code at it.** The easiest way is the interactive wizard, which
writes `~/clearml.conf` for you:

```bash
uv run --extra tracking clearml-init
# Prompts for the API server, Web server and Files server, then for credentials.
```

Get credentials from the web UI: open the Web server, sign in, then
**Settings → Workspace → Create new credentials** (copy the generated snippet).
For non-interactive/remote runs you can export the standard variables instead
of using `clearml-init`:

```bash
export CLEARML_API_HOST=http://localhost:8008
export CLEARML_WEB_HOST=http://localhost:8080
export CLEARML_FILES_HOST=http://localhost:8081
export CLEARML_API_ACCESS_KEY=...
export CLEARML_API_SECRET_KEY=...
# optional
export GCMTS_CLEARML_PROJECT=GenCausModTS
```

**3. Run** (the `--extra` flag keeps the tracking packages available to the
subprocess workers):

```bash
uv run --extra tracking python scripts/run_full_benchmark.py \
    --device cuda --seeds 10 --jobs 8
```

You then get **one task per benchmark step** (e.g. `Benchmark A.1: …`,
`CTEE static IHDP`, `CRL grid crl_shared_grid_a`) under the project, with each
run's metrics as scalars (`series = method`, `iteration = seed`), every task's
progress percentage, and seed-0 training curves.

#### Can `your-server` be `localhost`?

Yes — but note *which machine* resolves each host, because the three URLs are
used by different clients:

| Variable | Used by | `localhost` means |
| --- | --- | --- |
| `CLEARML_API_HOST` / `CLEARML_FILES_HOST` | the Python process (benchmark box) | the box running the benchmark |
| `CLEARML_WEB_HOST` | your **browser** | the browser's machine |

So `localhost` is correct **when the ClearML server runs on the same machine as
the benchmark process**. Common cases:

- **Server and benchmark on the same CUDA box.** Use `localhost` for all three.
  If you open the UI from your laptop, tunnel the web port over SSH:
  ```bash
  ssh -N -L 8080:localhost:8080 -L 8008:localhost:8008 you@cuda-box
  # then browse http://localhost:8080 locally
  ```
- **Server on your laptop, benchmark on the CUDA box.** `localhost` won't work
  for the benchmark process: set `CLEARML_API_HOST`/`CLEARML_FILES_HOST` to the
  laptop's LAN address (e.g. `http://192.168.1.20:8008`) and make ports 8008/8081
  reachable from the box.
- **Server on a third host.** Use its hostname for all three variables.

#### If ClearML is unavailable

Tracking is a no-op when the `clearml` package is missing, when the API host is
unreachable (a 2 s TCP probe runs first), or when you set `GCMTS_CLEARML=0`;
the benchmark keeps running. To run detached from any server and upload later,
set `GCMTS_CLEARML_OFFLINE=1` and use `clearml-task`/`clearml-agent` to sync.

### Data path and performance tuning

Most suites are **CPU-bound**: each step regenerates its batch on the CPU
(Python loops over `horizon × latent_dim`), which is far costlier than the tiny
model step, so the GPU idles. Measure the split first, per suite:

```bash
uv run python scripts/profile_training.py --suite npvar --method iVAE --device cuda
uv run python scripts/profile_training.py --suite voronoi --method CITRIS --device cuda --json p.json
```

It prints the mean/median time and share of each phase (`data`, `forward`,
`loss`, `backward`, `optim`, `sync`). Three results-neutral options can be
enabled through the environment (honoured by every script via `SimpleTrainer`):

```bash
export GCMTS_PREFETCH=1     # generate the next batch on a background thread, overlapping compute
export GCMTS_LIGHT_BATCH=1  # copy only x/context to the GPU during training (skip z/adjacency/...)
export GCMTS_PROFILE=1      # print a phase profile at the end of each fit
```

These are **result-neutral**: the batches and the RNG sequence are unchanged
(verified in `tests/test_data_path.py`). Gains are modest for the tiny-MLP
suites (a few percent) because generation, not the host-to-device copy, is the
bottleneck.

For a genuinely large throughput win there is an explicitly **protocol-changing**
mode: pre-generate a fixed pool of trajectories once and resample minibatches
from it (this removes ~all per-step generation — e.g. Voronoi drops from ~21 ms
to ~0.3 ms per step — but the data seen each step is no longer freshly drawn):

```bash
uv run python scripts/run_crl_shared.py --suite voronoi --device cuda --seeds 10 --pool-size 65536
```

Results from a pooled run are **not comparable** to a standard run and must be
reported separately (see `gcmts/data/pool.py`).

### Materialising the synthetic data once (persistent pools)

The CRL benchmarks are CPU-bound because the runners generate trajectories on
the fly. You can instead materialise the **fixed synthetic process** once — a
train pool plus a *disjoint* held-out eval pool per `(suite, seed)` — and train
from it. This preserves the protocol's invariant ("fixed seed schedules so that
the latent variables, mechanisms, mixing map and graph are identical across
methods") and additionally fixes the exact trajectories across methods.

Generate everything (this deletes any previous generation first):

```bash
cd gcmts
uv run python scripts/generate_synthetic_data.py --clean --seeds 10
# include the 18-problem grid too (larger; more disk)
uv run python scripts/generate_synthetic_data.py --clean --seeds 10 --with-grid
```

- Writes `data/synthetic/<suite>/seed_NN.pt` (train + eval pools) and
  `data/synthetic/manifest.json` (sizes, generator, dims, `sha256`) for audit.
- `--clean` removes `data/synthetic` first, so stale generations never linger.
- Defaults: `--train-size 65536`, `--eval-size 4096`. A pool is **finite**: over
  one run each trajectory is seen about `steps × batch_size / train_size` times,
  unlike the standard fresh-sampling protocol. Raise `--train-size` to move
  closer to fresh sampling, at the cost of disk/time.
- Progress is written to `data/synthetic/generate_progress.json`.

Now run the whole benchmark from those pools:

```bash
uv run python scripts/run_full_benchmark.py \
    --device cuda --jobs 8 --seeds 10 --with-grid \
    --data-source pool --dataset-dir data/synthetic
```

`--data-source pool` makes each CRL step load its `(suite, seed)` pool, train by
resampling it, and evaluate on the disjoint eval pool; a **missing pool aborts
that step** with a message telling you to run the generator. Coverage today is
the **CRL shared suites and the grid**; CTEE temporal, SDD and the adjacent
branches still sample fresh (they are a small share of the runtime).

> **Protocol note.** `--data-source pool` is a *different protocol* from the
> default `fresh` runs — training data is a finite fixed set shared across
> methods rather than freshly drawn each step — so pool runs must be reported
> separately, or adopted as the protocol and **every** number regenerated. The
> default is `--data-source fresh`.

Per-step cost drops from ~`data` + GPU to `pool-index` + GPU (e.g. Voronoi from
~2.9 h to ~11 min per run), so the run becomes GPU-bound and finishes far sooner.

### Benchmark results

Each method is trained on the synthetic benchmark matching its assumptions;
input/output shapes are validated and the aggregate results are written to
`outputs/crl_benchmarks.{log,json,csv}`. The shared suites (NP/VAR, CartPole,
Causal3DIdent, Voronoi) and the 18-problem grid are written to
`outputs_shared/`, together with Holm-corrected Wilcoxon tests,
Friedman/Nemenyi summaries and CD diagrams. Per-experiment training histories,
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

These single-seed numbers are the implementation sanity check. The comparative
protocol uses ten seeds: on NP/VAR no pairwise representation difference
survives Holm correction, the forecast columns separate LEAP/NCTRL from
iVAE/TDRL/MOSAIC, and the CRL methods are ranked over an eighteen-problem
configuration grid with Friedman/Nemenyi tests (see `outputs_shared/` and
`scripts/make_paper_tables.py`).

---

## Taxonomy coverage

```bash
uv run python scripts/taxonomy_coverage.py            # report
uv run python scripts/taxonomy_coverage.py --strict   # non-zero if any row uncovered
```

The script parses the bundled taxonomy table (`docs/taxonomy_table.tex`) and
maps each branch to its implemented method(s). All five CRL branches are
covered (5/5); the effect-estimation and adjacent branches are declared but not
yet tracked by this CRL-focused script.

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

- **Under-tuned methods.** TDRL and NCTRL are above chance but below their
  papers' reported ranges; further architecture/tuning work is needed.
- **Statistical protocol.** The shared suites and the configuration grid run
  with ten seeds and report Holm-corrected Wilcoxon plus Friedman/Nemenyi tests;
  bootstrap confidence intervals and a tagged release with a DOI are pending.
- **Pending methods.** iCITRIS, APPEX, and the adjacent-branch methods
  (structure discovery, decision-making, explanation) are not implemented.
- **Pending benchmarks.** The SDE suite in the shared comparison and the
  rendered image variant of Causal3DIdent remain to be run; the ACIC archives
  require a hand download.

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
