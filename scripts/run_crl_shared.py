"""Shared head-to-head CRL suites (Table~\\ref{tab:exp_crl}) and grid sweep.

Trains the temporal CRL methods on a *common* benchmark so that methods from
different branches are directly comparable within a block, and reports
representation- and structural-recovery metrics (mean/std over seeds). The
``--grid`` mode expands the problem set to 18 commensurable NP/VAR
configurations (latent dimension, observed dimension, graph sparsity and
regime count), which gives Friedman/Nemenyi the sample size they need; the
per-suite mode runs the three shared suites.

Usage::

    uv run python scripts/run_crl_shared.py --device cuda --suite npvar --seeds 10
    uv run python scripts/run_crl_shared.py --device cuda --grid --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.causal_representation_learning.methods import (
    CEGEN,
    CITRIS,
    IVAE,
    LEAP,
    MOSAIC,
    NCTRL,
    TDRL,
    CHiLD,
    iCITRIS,
)
from gcmts.core import SimpleTrainer, move_batch, save_training_report
from gcmts.core.graph_ops import edge_auprc, estimate_latent_adjacency, latent_permutation
from gcmts.data.control import MountainCarGenerator, PendulumGenerator
from gcmts.data.hierarchical import HierarchicalTemporalGenerator
from gcmts.data.linear_processes import OUProcessGenerator, VARMAProcessGenerator
from gcmts.data.lorenz import LorenzGenerator
from gcmts.data.physics import PhysicsSCMGenerator
from gcmts.data.pool import PooledGenerator
from gcmts.data.synthetic import (
    CartPoleGenerator,
    Causal3DIdentGenerator,
    LinearSDEGenerator,
    TemporalNonlinearGenerator,
)
from gcmts.data.synthetic_store import dataset_path, load_dataset
from gcmts.data.voronoi import VoronoiGenerator
from gcmts.evaluation import BenchmarkTask, CrlBenchmarkRunner
from gcmts.evaluation.benchmarks import crl_label
from gcmts.evaluation.metrics import SHDMetric, TripletDistanceMetric
from gcmts.evaluation.progress import ProgressTracker
from gcmts.evaluation.report import aggregate_rows, save_rows, wall_seconds
from gcmts.evaluation.stats import (
    friedman_test,
    metric_matrix,
    nemenyi,
    paired_wilcoxon_holm,
    plot_cd_diagram,
)
from gcmts.evaluation.tracking import StepTracker
from gcmts.typing import Batch

logger = logging.getLogger("gcmts.crl_shared")

MethodSpec = tuple[str, Callable[[], BaseCausalRepresentationLearner], int]


@dataclass
class Suite:
    name: str
    observed_dim: int
    latent_dim: int
    horizon: int
    n_regimes: int
    make_generator: Callable[[int], Any]


SUITES: dict[str, Suite] = {
    "npvar": Suite(
        name="npvar",
        observed_dim=6,
        latent_dim=3,
        horizon=16,
        n_regimes=4,
        make_generator=lambda seed: TemporalNonlinearGenerator(
            observed_dim=6,
            latent_dim=3,
            horizon=16,
            max_lag=1,
            n_regimes=4,
            regime_mode="time",
            mixing="linear",
            seed=seed,
        ),
    ),
    "causal3d": Suite(
        name="causal3d",
        observed_dim=4,
        latent_dim=4,
        horizon=16,
        n_regimes=1,
        make_generator=lambda seed: Causal3DIdentGenerator(
            observed_dim=4,
            latent_dim=4,
            horizon=16,
            max_lag=1,
            mixing="invertible",
            seed=seed,
        ),
    ),
    "cartpole": Suite(
        name="cartpole",
        observed_dim=4,
        latent_dim=4,
        horizon=16,
        n_regimes=3,
        make_generator=lambda seed: CartPoleGenerator(
            horizon=16,
            observed_dim=4,
            n_regimes=3,
            seed=seed,
        ),
    ),
    "voronoi": Suite(
        name="voronoi",
        observed_dim=6,
        latent_dim=6,
        horizon=16,
        n_regimes=1,
        make_generator=lambda seed: VoronoiGenerator(n_vars=5, horizon=16, seed=seed),
    ),
    "sde": Suite(
        name="sde",
        observed_dim=4,
        latent_dim=4,
        horizon=20,
        n_regimes=1,
        make_generator=lambda seed: LinearSDEGenerator(
            observed_dim=4, horizon=20, seed=seed
        ),
    ),
    "lorenz": Suite(
        name="lorenz",
        observed_dim=6,
        latent_dim=3,
        horizon=16,
        n_regimes=3,
        make_generator=lambda seed: LorenzGenerator(
            observed_dim=6, horizon=16, n_regimes=3, mixing="mlp", seed=seed
        ),
    ),
    "npvar_hard": Suite(
        name="npvar_hard",
        observed_dim=8,
        latent_dim=6,
        horizon=16,
        n_regimes=4,
        make_generator=lambda seed: TemporalNonlinearGenerator(
            observed_dim=8,
            latent_dim=6,
            horizon=16,
            max_lag=1,
            n_regimes=4,
            regime_mode="time",
            mixing="mlp",
            noise_std=0.2,
            seed=seed,
        ),
    ),
    "child": Suite(
        name="child",
        observed_dim=4,
        latent_dim=8,
        horizon=12,
        n_regimes=1,
        make_generator=lambda seed: HierarchicalTemporalGenerator(
            width=4, n_layers=2, horizon=12, observed_dim=4, seed=seed
        ),
    ),
}

# Physics, control and linear process families (Benchmarks A.6--A.17): each is a
# temporal-family benchmark, so it uses the default method list.
_PHYSICS_DIMS = {
    "lorenz96": 5,
    "pcl": 6,
    "kuramoto": 6,
    "lotka_volterra": 4,
    "duffing": 2,
    "fitzhugh_nagumo": 2,
    "chua": 3,
    "van_der_pol": 2,
}
for _system, _dim in _PHYSICS_DIMS.items():
    SUITES[_system] = Suite(
        name=_system,
        observed_dim=2 * _dim,
        latent_dim=_dim,
        horizon=16,
        n_regimes=3,
        make_generator=lambda seed, s=_system, d=_dim: PhysicsSCMGenerator(
            system=s, observed_dim=2 * d, horizon=16, n_regimes=3, seed=seed
        ),
    )
SUITES["pendulum"] = Suite(
    name="pendulum", observed_dim=4, latent_dim=2, horizon=16, n_regimes=3,
    make_generator=lambda seed: PendulumGenerator(
        observed_dim=4, horizon=16, n_regimes=3, seed=seed
    ),
)
SUITES["mountain_car"] = Suite(
    name="mountain_car", observed_dim=4, latent_dim=2, horizon=16, n_regimes=3,
    make_generator=lambda seed: MountainCarGenerator(
        observed_dim=4, horizon=16, n_regimes=3, seed=seed
    ),
)
SUITES["varma"] = Suite(
    name="varma", observed_dim=4, latent_dim=4, horizon=16, n_regimes=3,
    make_generator=lambda seed: VARMAProcessGenerator(
        latent_dim=4, observed_dim=4, horizon=16, n_regimes=3, seed=seed
    ),
)
SUITES["ou"] = Suite(
    name="ou", observed_dim=4, latent_dim=4, horizon=16, n_regimes=3,
    make_generator=lambda seed: OUProcessGenerator(
        latent_dim=4, observed_dim=4, horizon=16, n_regimes=3, seed=seed
    ),
)

# Suite B is intervention-conditioned, so the applicable representative is
# CITRIS (I.D); the non-stationarity family is reported on Suites A.
SUITE_METHODS = {
    "voronoi": ["CITRIS", "iCITRIS"],
    "sde": ["CEGEN"],
    "child": ["CHiLD"],
}


@dataclass
class GridConfig:
    name: str
    latent_dim: int
    observed_dim: int
    sparsity: float
    n_regimes: int
    noise_std: float

    def as_suite(self) -> Suite:
        return Suite(
            name=f"grid_{self.name}",
            observed_dim=self.observed_dim,
            latent_dim=self.latent_dim,
            horizon=16,
            n_regimes=self.n_regimes,
            make_generator=lambda seed: TemporalNonlinearGenerator(
                observed_dim=self.observed_dim,
                latent_dim=self.latent_dim,
                horizon=16,
                max_lag=1,
                n_regimes=self.n_regimes,
                regime_mode="time",
                mixing="linear",
                adjacency_sparsity=self.sparsity,
                noise_std=self.noise_std,
                seed=seed,
            ),
        )


def _grid_configs() -> list[GridConfig]:
    """18 commensurable NP/VAR problems: d x D x sparsity, regimes/noise cycling."""
    regimes_cycle = (2, 4, 8)
    noise_cycle = (0.05, 0.1, 0.2)
    configs: list[GridConfig] = []
    for latent_index, latent_dim in enumerate((3, 5, 8)):
        for multiplier_index, multiplier in enumerate((2, 3)):
            for sparsity_index, sparsity in enumerate((0.2, 0.4, 0.6)):
                n_regimes = regimes_cycle[(latent_index + sparsity_index) % 3]
                noise_std = noise_cycle[(multiplier_index + sparsity_index) % 3]
                configs.append(
                    GridConfig(
                        name=(
                            f"d{latent_dim}_D{multiplier * latent_dim}"
                            f"_s{int(sparsity * 10)}_r{n_regimes}"
                        ),
                        latent_dim=latent_dim,
                        observed_dim=multiplier * latent_dim,
                        sparsity=sparsity,
                        n_regimes=n_regimes,
                        noise_std=noise_std,
                    )
                )
    return configs


GRID_CONFIGS = _grid_configs()


def build_methods(suite: Suite, *, epochs_scale: float = 1.0) -> list[MethodSpec]:
    d, r, D = suite.latent_dim, suite.n_regimes, suite.observed_dim
    u = max(r, 1)  # stationary suites expose a constant, one-dimensional context
    fix = max(d - 2, 1)
    obs = 1
    dyn = max(d - fix - obs, 1)

    def scale(value: int) -> int:
        return max(int(value * epochs_scale), 200)

    return [
        (
            "iVAE",
            lambda: IVAE(observed_dim=D, latent_dim=d, u_dim=u, obs_noise=0.1),
            scale(1500),
        ),
        (
            "LEAP",
            lambda: LEAP(
                observed_dim=D,
                latent_dim=d,
                max_lag=1,
                u_dim=u,
                kl_weight=0.01,
                obs_noise=0.1,
            ),
            scale(2000),
        ),
        (
            "TDRL",
            lambda: TDRL(
                observed_dim=D,
                latent_fix_dim=fix,
                latent_dyn_dim=dyn,
                latent_obs_dim=obs,
                u_dim=u,
                kl_weight=0.01,
                obs_noise=0.1,
            ),
            scale(2000),
        ),
        (
            "NCTRL",
            lambda: NCTRL(
                observed_dim=D, latent_dim=d, n_regimes=u, kl_weight=0.01, obs_noise=0.1
            ),
            scale(2000),
        ),
        (
            "MOSAIC",
            lambda: MOSAIC(observed_dim=D, latent_dim=d, u_dim=u, kl_weight=0.1, obs_noise=0.1),
            scale(1200),
        ),
    ]


def build_voronoi_methods(suite: Suite, epochs_scale: float = 1.0) -> list[MethodSpec]:
    n_vars = suite.observed_dim - 1
    epochs = max(int(2000 * epochs_scale), 200)
    return [
        (
            "CITRIS",
            lambda: CITRIS(
                observed_dim=suite.observed_dim,
                n_vars=n_vars,
                var_dim=1,
                n_shared=1,
            ),
            epochs,
        ),
        (
            "iCITRIS",
            lambda: iCITRIS(
                observed_dim=suite.observed_dim,
                n_vars=n_vars,
                var_dim=1,
                n_shared=1,
                n_flow_layers=4,
            ),
            epochs,
        ),
    ]


def build_child_methods(suite: Suite, epochs_scale: float = 1.0) -> list[MethodSpec]:
    """CHiLD is the only representative of the hierarchical temporal suite."""
    return [
        (
            "CHiLD",
            lambda: CHiLD(
                observed_dim=suite.observed_dim,
                latent_dim=suite.observed_dim,
                n_layers=2,
            ),
            max(int(1500 * epochs_scale), 200),
        )
    ]


def build_sde_methods(suite: Suite, epochs_scale: float = 1.0) -> list[MethodSpec]:
    """CEGEN operates in observation space (no mixing model), so it is the only
    representative of the continuous-time SDE suite."""
    return [
        (
            "CEGEN",
            lambda: CEGEN(observed_dim=suite.observed_dim, n_regions=4),
            max(int(800 * epochs_scale), 200),
        )
    ]


@dataclass
class Row:
    case: str
    dataset: str
    suite: str
    method: str
    seed: int
    mcc: float
    r2_diag: float
    r2_sep: float
    dci: float
    shd: float | None
    triplet: float | None
    forecast1: float
    forecast4: float
    seconds: float
    auprc: float | None = None
    w2: float | None = None


def _slice_context(context: Any, time: int, new_time: int) -> dict[str, Any]:
    """Truncate per-timestep context tensors to match a shortened input window."""
    out: dict[str, Any] = {}
    for key, value in (context or {}).items():
        out[key] = (
            value[:, :new_time]
            if (value.ndim in (2, 3) and value.shape[1] == time)
            else value
        )
    return out


def _forecast_mse(model: BaseCausalRepresentationLearner, batch: Any, steps: int) -> float:
    """Forecast ``steps`` future observations from the preceding window."""
    time = batch.x.shape[1]
    if time <= steps:
        return float("nan")
    inp = batch.x[:, : time - steps]
    context = _slice_context(batch.context, time, time - steps)
    with torch.no_grad():
        pred = model.predict_next_observations(inp, steps, context)
    target = batch.x[:, time - steps :]
    return float(((pred - target) ** 2).mean())


def _persistence_mse(batch: Any, steps: int) -> float:
    time = batch.x.shape[1]
    last = batch.x[:, time - steps - 1 : time - steps]
    return float(((last.expand(-1, steps, -1) - batch.x[:, time - steps :]) ** 2).mean())


def _triplet_score(
    model: BaseCausalRepresentationLearner, batch: Any, max_lag: int = 1
) -> float | None:
    """Intervention isolation from consecutive latent pairs with known targets."""
    if batch.context is None or "I" not in batch.context or batch.z is None:
        return None
    targets = batch.context["I"]
    if targets.shape[-1] < 1:
        return None
    with torch.no_grad():
        z_pred = model.encode(batch.x, batch.context)
    before, after, target_index, truth = [], [], [], []
    for step in range(max_lag, targets.shape[1]):
        active = targets[:, step].sum(-1) == 1
        if not bool(active.any()):
            continue
        index = targets[active, step].argmax(dim=-1)
        before.append(z_pred[active, step - max_lag])
        after.append(z_pred[active, step])
        target_index.append(index)
        truth.append(batch.z[active, step - max_lag])
    if not before:
        return None
    prediction = {
        "before": torch.cat(before, dim=0),
        "after": torch.cat(after, dim=0),
        "targets": torch.cat(target_index, dim=0),
    }
    return TripletDistanceMetric()(prediction, torch.cat(truth, dim=0)).value


def _load_pool(dataset_dir: Path, suite_name: str, seed: int) -> tuple[Any, Batch, int]:
    """Load a persisted ``(train, eval)`` pool for ``(suite, seed)``.

    Raises with an actionable message when the file is missing, so
    ``--data-source pool`` never silently falls back to fresh sampling.
    """
    path = dataset_path(dataset_dir, suite_name, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing synthetic dataset {path}. Generate it first with "
            "`uv run python scripts/generate_synthetic_data.py --seeds ... "
            f"--suites {suite_name}` (or run with --data-source fresh)."
        )
    train, eval_batch, meta = load_dataset(path)
    return (
        PooledGenerator.from_batch(train, resample_seed=seed),
        eval_batch,
        int(meta.get("max_lag", 1)),
    )


def _run(
    suite: Suite,
    method: str,
    factory: Callable[[], BaseCausalRepresentationLearner],
    epochs: int,
    seed: int,
    device: str,
    batch_size: int,
    out_dir: Path,
    save_curves: bool,
    tracker: StepTracker | None = None,
    pool_size: int = 0,
    train_data: Any = None,
    eval_batch: Batch | None = None,
    max_lag_override: int | None = None,
) -> Row:
    if train_data is None:
        torch.manual_seed(seed)
        source: Any = suite.make_generator(seed)
        if pool_size:
            source = PooledGenerator(source, pool_size, resample_seed=seed)
    else:
        # Pre-materialised fixed pool (see scripts/generate_synthetic_data.py).
        source = train_data
    model = factory()
    trainer = SimpleTrainer(
        max_epochs=epochs,
        lr=1e-3,
        batch_size=batch_size,
        steps_per_epoch=15,
        log_every=max(epochs // 10, 1),
        device=device,
        seed=seed,
        grad_clip=1.0,
        scheduler="cosine",
        val_size=512,
        val_every=20,
    )
    start = time.perf_counter()
    history = trainer.fit(model, source)
    seconds = wall_seconds(start)
    if save_curves:
        exp = out_dir / f"shared_{suite.name}_{method}"
        save_training_report(history, exp, title=f"{suite.name}/{method}")
        if tracker is not None:
            tracker.log_curves(method, history)

    resolved = trainer.resolve_device(model)
    raw_eval = eval_batch if eval_batch is not None else source.sample(2048)
    batch = move_batch(raw_eval, resolved)
    if suite.name == "sde":
        with torch.no_grad():
            outputs = model.forward(batch)
            extras = outputs.extras or {}
            w2 = float(extras.get("w2", torch.tensor(float("nan"))))
        return Row(
            case=method,
            dataset=suite.name,
            suite=suite.name,
            method=method,
            seed=seed,
            mcc=float("nan"),
            r2_diag=float("nan"),
            r2_sep=float("nan"),
            dci=float("nan"),
            shd=None,
            triplet=None,
            forecast1=_forecast_mse(model, batch, 1),
            forecast4=_forecast_mse(model, batch, 4),
            seconds=seconds,
            w2=w2,
        )
    task = BenchmarkTask(
        name=f"{suite.name}_{method}",
        model=model,
        data=batch,
        metrics=["mcc", "r2_diag", "r2_sep", "dci_disentanglement"],
        causal_level="L1_association",
    )
    result = CrlBenchmarkRunner().run(task)
    scores = {m.name: m.value for m in result.metrics}
    shd: float | None = None
    auprc: float | None = None
    if batch.adjacency is not None:
        with torch.no_grad():
            z_pred = model.encode(batch.x, batch.context)
        max_lag = max_lag_override if max_lag_override is not None else int(
            getattr(source, "max_lag", 1)
        )
        adj_est = estimate_latent_adjacency(z_pred, max_lag=max_lag)
        adjacency = batch.adjacency[..., -max_lag:].to(adj_est.device)
        if batch.z is not None:
            perm = latent_permutation(z_pred, batch.z)
            adj_est = adj_est[perm][:, perm]
        shd = SHDMetric()(adj_est, adjacency).value
        auprc = edge_auprc(adj_est, adjacency)
    triplet = _triplet_score(model, batch)
    forecast1 = _forecast_mse(model, batch, 1)
    forecast4 = _forecast_mse(model, batch, 4)
    return Row(
        case=method,
        dataset=suite.name,
        suite=suite.name,
        method=method,
        seed=seed,
        mcc=scores["mcc"],
        r2_diag=scores["r2_diag"],
        r2_sep=scores["r2_sep"],
        dci=scores["dci_disentanglement"],
        shd=shd,
        triplet=triplet,
        forecast1=forecast1,
        forecast4=forecast4,
        seconds=seconds,
        auprc=auprc,
    )


def _report_statistics(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    stem: str,
    metrics: list[tuple[str, bool]],
) -> dict[str, Any]:
    """Pairwise Holm-corrected Wilcoxon + Friedman/Nemenyi with CD diagrams."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for metric, higher_is_better in metrics:
        matrix, cases, problems = metric_matrix(rows, metric)
        comparisons = paired_wilcoxon_holm(rows, metric)
        entry: dict[str, Any] = {
            "comparisons": [asdict(c) for c in comparisons],
        }
        logger.info(
            "--- %s | %d cases | %d problems | pairwise Holm-corrected Wilcoxon ---",
            metric,
            len(cases),
            len(problems),
        )
        for comparison in comparisons:
            logger.info(
                "%-10s %-8s vs %-8s n=%d p=%.4g p_holm=%.4g%s",
                metric,
                comparison.case_a,
                comparison.case_b,
                comparison.n_pairs,
                comparison.p_value,
                comparison.p_holm,
                " *" if comparison.significant else "",
            )
        complete = matrix[~np.isnan(matrix).any(axis=1)]
        if complete.shape[0] >= 2 and complete.shape[1] >= 3:
            friedman = friedman_test(matrix)
            posthoc = nemenyi(matrix, higher_is_better=higher_is_better)
            entry["friedman"] = friedman
            entry["nemenyi"] = {
                "average_ranks": {
                    case: float(rank)
                    for case, rank in zip(cases, posthoc["average_ranks"].tolist(), strict=True)
                },
                "critical_difference": float(posthoc["critical_difference"]),
                "n_problems": int(posthoc["n_problems"]),
                "alpha": float(posthoc["alpha"]),
            }
            plot_cd_diagram(
                {
                    case: float(rank)
                    for case, rank in zip(cases, posthoc["average_ranks"].tolist(), strict=True)
                },
                float(posthoc["critical_difference"]),
                path=out_dir / f"cd_{stem}_{metric}.png",
                title=f"{metric} ({'higher' if higher_is_better else 'lower'} is better)",
            )
            logger.info(
                "Friedman %s: chi2=%.3f p=%.4g | Nemenyi CD=%.3f over N=%d",
                metric,
                friedman["statistic"],
                friedman["p_value"],
                posthoc["critical_difference"],
                posthoc["n_problems"],
            )
        summary[metric] = entry
    (out_dir / f"{stem}_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _methods_for(suite: Suite, epochs_scale: float) -> list[MethodSpec]:
    if suite.name == "voronoi":
        return build_voronoi_methods(suite, epochs_scale)
    if suite.name == "sde":
        return build_sde_methods(suite, epochs_scale)
    if suite.name == "child":
        return build_child_methods(suite, epochs_scale)
    return build_methods(suite, epochs_scale=epochs_scale)


def _log_rows(suite_name: str, method: str, seed: int, row: Row) -> None:
    logger.info(
        "RESULT %s/%s seed=%d mcc=%.3f r2d=%.3f r2s=%.3f dci=%.3f shd=%s triplet=%s f1=%.3f",
        suite_name,
        method,
        seed,
        row.mcc,
        row.r2_diag,
        row.r2_sep,
        row.dci,
        "n/a" if row.shd is None else f"{row.shd:.1f}",
        "n/a" if row.triplet is None else f"{row.triplet:.3f}",
        row.forecast1,
    )


def _log_aggregate(rows: list[dict[str, Any]], methods: list[str], seeds: int) -> None:
    aggregated = aggregate_rows(rows)

    def stat(case: str, metric: str) -> str:
        if case not in aggregated or metric not in aggregated[case]:
            return "--"
        entry = aggregated[case][metric]
        return f"{entry['mean']:.3f}±{entry['std']:.3f}"

    logger.info("=== mean±std over %d seeds ===", seeds)
    for name in methods:
        logger.info(
            "%-8s mcc=%s r2d=%s r2s=%s shd=%s triplet=%s f1=%s f4=%s",
            name,
            stat(name, "mcc"),
            stat(name, "r2_diag"),
            stat(name, "r2_sep"),
            stat(name, "shd"),
            stat(name, "triplet"),
            stat(name, "forecast1"),
            stat(name, "forecast4"),
        )


def _baselines(suite: Suite) -> None:
    from sklearn.linear_model import Ridge

    eval_batch = move_batch(suite.make_generator(0).sample(4096), torch.device("cpu"))
    train_batch = suite.make_generator(12345).sample(8192)
    flat_x = train_batch.x[:, :-1].reshape(-1, train_batch.x.shape[-1]).numpy()
    flat_y = train_batch.x[:, 1:].reshape(-1, train_batch.x.shape[-1]).numpy()
    ar = Ridge(alpha=1e-3).fit(flat_x, flat_y)
    test_x = eval_batch.x[:, :-1].reshape(-1, eval_batch.x.shape[-1]).numpy()
    test_y = eval_batch.x[:, 1:].reshape(-1, eval_batch.x.shape[-1]).numpy()
    ar_mse = float(((ar.predict(test_x) - test_y) ** 2).mean())
    logger.info(
        "BASELINE %s persistence f1=%.4f f4=%.4f | AR(1) f1=%.4f",
        suite.name,
        _persistence_mse(eval_batch, 1),
        _persistence_mse(eval_batch, 4),
        ar_mse,
    )


def configure_logging(out_dir: Path, name: str) -> None:
    """Configure logging to ``<out_dir>/<name>.log``.

    ``name`` is a full stem (e.g. ``crl_shared_npvar`` or
    ``crl_shared_grid_a``) so that the concurrently-run benchmarks never share
    a log file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [
        logging.FileHandler(out_dir / f"{name}.log", mode="w"),
        logging.StreamHandler(),
    ]


def run_suite(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    methods = _methods_for(suite, args.epochs_scale)
    stem = f"crl_shared_{suite.name}"
    logger.info(
        "=== %s | methods: %s | seeds: %d | epochs_scale: %.2f ===",
        crl_label(suite.name),
        ", ".join(name for name, _, _ in methods),
        args.seeds,
        args.epochs_scale,
    )
    rows: list[dict[str, Any]] = []
    total = args.seeds * len(methods)
    progress = ProgressTracker(
        args.out_dir / f"{stem}_progress.json", crl_label(suite.name), total
    ).start()
    step = StepTracker.start(
        name=crl_label(suite.name),
        params={
            "suite": suite.name,
            "seeds": args.seeds,
            "epochs_scale": args.epochs_scale,
            "methods": [name for name, _, _ in methods],
        },
        tags=["crl", suite.name],
        progress=progress,
    )
    for seed in range(args.seeds):
        if args.data_source == "pool":
            train_data, eval_batch, ds_max_lag = _load_pool(args.dataset_dir, suite.name, seed)
        else:
            train_data = eval_batch = ds_max_lag = None
        for name, factory, epochs in methods:
            row = _run(
                suite,
                name,
                factory,
                epochs,
                seed,
                args.device,
                args.batch_size,
                args.out_dir,
                save_curves=(seed == 0),
                tracker=step,
                pool_size=args.pool_size,
                train_data=train_data,
                eval_batch=eval_batch,
                max_lag_override=ds_max_lag,
            )
            rows.append(asdict(row))
            _log_rows(suite.name, name, seed, row)
            save_rows(rows, args.out_dir, stem=stem)
            progress.advance()
            step.log_run(
                method=name,
                iteration=seed,
                metrics={
                    "mcc": row.mcc,
                    "r2_diag": row.r2_diag,
                    "r2_sep": row.r2_sep,
                    "dci": row.dci,
                    "shd": row.shd,
                    "triplet": row.triplet,
                    "forecast1": row.forecast1,
                    "forecast4": row.forecast4,
                    "auprc": row.auprc,
                    "w2": row.w2,
                },
            )
    progress.finish()
    step.close()
    save_rows(rows, args.out_dir, stem=stem)
    _log_aggregate(rows, [name for name, _, _ in methods], args.seeds)
    if args.baselines:
        _baselines(suite)
    _report_statistics(
        rows,
        args.out_dir,
        stem=stem,
        metrics=[
            ("mcc", True),
            ("r2_diag", True),
            ("r2_sep", False),
            ("shd", False),
            ("auprc", True),
            ("triplet", False),
            ("forecast1", False),
            ("forecast4", False),
        ],
    )
    return 0


def run_grid(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    configs = GRID_CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [config for config in configs if config.name in wanted]
    if args.shard_total:
        if not 0 <= args.shard_index < args.shard_total:
            logger.error(
                "Invalid shard: --shard-index %d out of range for --shard-total %d.",
                args.shard_index,
                args.shard_total,
            )
            return 1
        configs = configs[args.shard_index :: args.shard_total]
    if not configs:
        logger.error("No grid configs selected; check --configs/--shard-index.")
        return 1
    stem = args.stem or "crl_shared_grid"
    representative = build_methods(configs[0].as_suite(), epochs_scale=args.epochs_scale)
    if args.only:
        representative = [m for m in representative if m[0] in set(args.only)]
    total = args.seeds * len(configs) * len(representative)
    progress = ProgressTracker(
        args.out_dir / f"{stem}_progress.json", f"Grid {stem} ({len(configs)} configs)", total
    ).start()
    step = StepTracker.start(
        name=f"CRL grid {stem}",
        params={
            "configs": [c.name for c in configs],
            "seeds": args.seeds,
            "epochs_scale": args.epochs_scale,
            "methods": [name for name, _, _ in representative],
        },
        tags=["crl", "grid", stem],
        progress=progress,
    )
    for seed in range(args.seeds):
        for config in configs:
            suite = config.as_suite()
            methods = build_methods(suite, epochs_scale=args.epochs_scale)
            if args.only:
                methods = [m for m in methods if m[0] in set(args.only)]
            if args.data_source == "pool":
                train_data, eval_batch, ds_max_lag = _load_pool(args.dataset_dir, suite.name, seed)
            else:
                train_data = eval_batch = ds_max_lag = None
            for name, factory, epochs in methods:
                row = _run(
                    suite,
                    name,
                    factory,
                    epochs,
                    seed,
                    args.device,
                    args.batch_size,
                    args.out_dir,
                    save_curves=False,
                    pool_size=args.pool_size,
                    train_data=train_data,
                    eval_batch=eval_batch,
                    max_lag_override=ds_max_lag,
                )
                row.dataset = config.name
                rows.append(asdict(row))
                _log_rows(config.name, name, seed, row)
                save_rows(rows, args.out_dir, stem=stem)
                progress.advance()
                step.log_run(
                    method=name,
                    iteration=seed,
                    params={"config": config.name},
                    metrics={
                        "mcc": row.mcc,
                        "r2_diag": row.r2_diag,
                        "r2_sep": row.r2_sep,
                        "dci": row.dci,
                        "forecast1": row.forecast1,
                        "forecast4": row.forecast4,
                    },
                )
    progress.finish()
    step.close()
    save_rows(rows, args.out_dir, stem=stem)
    method_names = [name for name, _, _ in build_methods(configs[0].as_suite())]
    _log_aggregate(rows, method_names, args.seeds)
    _report_statistics(
        rows,
        args.out_dir,
        stem=stem,
        metrics=[
            ("mcc", True),
            ("r2_diag", True),
            ("r2_sep", False),
            ("dci", True),
            ("forecast1", False),
            ("forecast4", False),
        ],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run shared CRL suites / grid.")
    parser.add_argument("--suite", default="npvar", choices=sorted(SUITES))
    parser.add_argument("--grid", action="store_true", help="run the 18-problem config sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--configs", nargs="*", default=None, help="grid config names to run")
    parser.add_argument("--stem", default=None, help="output stem for grid splits")
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="index of this grid shard (paired with --shard-total) to run configs round-robin",
    )
    parser.add_argument(
        "--shard-total", type=int, default=0,
        help="split the grid configs into this many round-robin shards (0 = run all)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=0,
        help=(
            "PROTOCOL-CHANGING throughput mode: pre-generate this many trajectories "
            "and resample minibatches instead of drawing fresh data each step "
            "(results are not comparable to a standard run)"
        ),
    )
    parser.add_argument(
        "--data-source", choices=("fresh", "pool"), default="fresh",
        help=(
            "training data source: 'fresh' draws new SCM trajectories each step "
            "(standard protocol); 'pool' loads the persisted fixed dataset for "
            "(suite, seed) from --dataset-dir (finite-pool protocol; see "
            "scripts/generate_synthetic_data.py)"
        ),
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/synthetic"),
        help="root of the persisted synthetic pools used with --data-source pool",
    )
    parser.add_argument(
        "--epochs-scale",
        type=float,
        default=1.0,
        help="multiply the per-method epoch budgets (grid runs use < 1 to fit the budget)",
    )
    parser.add_argument("--no-baselines", dest="baselines", action="store_false")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.grid:
        configure_logging(args.out_dir, args.stem or "crl_shared_grid")
        return run_grid(args)
    configure_logging(args.out_dir, f"crl_shared_{args.suite}")
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
