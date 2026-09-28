"""Shared head-to-head CRL suites (Table~\\ref{tab:exp_crl}).

Trains the temporal CRL methods on a *common* benchmark so that methods from
different sub-taxonomies are directly comparable within a block, and reports
representation- and structural-recovery metrics (mean/std over seeds).

Usage::

    uv run python scripts/run_crl_shared.py --device cuda --suite npvar --seeds 5
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.causal_representation_learning.methods import (
    IVAE,
    LEAP,
    MOSAIC,
    NCTRL,
    TDRL,
)
from gcmts.core import SimpleTrainer, move_batch, save_training_report
from gcmts.core.graph_ops import estimate_latent_adjacency
from gcmts.data.synthetic import (
    CartPoleGenerator,
    Causal3DIdentGenerator,
    TemporalNonlinearGenerator,
)
from gcmts.evaluation import BenchmarkTask, CrlBenchmarkRunner
from gcmts.evaluation.metrics import SHDMetric, WSHDMetric
from gcmts.evaluation.report import aggregate_rows, save_rows

logger = logging.getLogger("gcmts.crl_shared")


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
            observed_dim=6, latent_dim=3, horizon=16, max_lag=1, n_regimes=4,
            regime_mode="time", mixing="linear", seed=seed,
        ),
    ),
    "causal3d": Suite(
        name="causal3d",
        observed_dim=4,
        latent_dim=4,
        horizon=16,
        n_regimes=1,
        make_generator=lambda seed: Causal3DIdentGenerator(
            observed_dim=4, latent_dim=4, horizon=16, max_lag=1,
            mixing="invertible", seed=seed,
        ),
    ),
    "cartpole": Suite(
        name="cartpole",
        observed_dim=4,
        latent_dim=4,
        horizon=16,
        n_regimes=3,
        make_generator=lambda seed: CartPoleGenerator(
            horizon=16, observed_dim=4, n_regimes=3, seed=seed,
        ),
    ),
}


def build_methods(
    suite: Suite,
) -> list[tuple[str, Callable[[], BaseCausalRepresentationLearner], int]]:
    d, r, D = suite.latent_dim, suite.n_regimes, suite.observed_dim
    u = max(r, 2)  # unconditional suites still expose a small regime context
    fix = max(d - 2, 1)
    obs = 1
    dyn = max(d - fix - obs, 1)
    return [
        ("iVAE", lambda: IVAE(observed_dim=D, latent_dim=d, u_dim=u, obs_noise=0.1), 1500),
        (
            "LEAP",
            lambda: LEAP(
                observed_dim=D, latent_dim=d, max_lag=1, u_dim=u,
                kl_weight=5.0, obs_noise=0.1,
            ),
            800,
        ),
        (
            "TDRL",
            lambda: TDRL(
                observed_dim=D, latent_fix_dim=fix, latent_dyn_dim=dyn, latent_obs_dim=obs,
                u_dim=u, obs_noise=0.1,
            ),
            800,
        ),
        ("NCTRL", lambda: NCTRL(observed_dim=D, latent_dim=d, n_regimes=u, obs_noise=0.1), 1000),
        ("MOSAIC", lambda: MOSAIC(observed_dim=D, latent_dim=d, u_dim=u, obs_noise=0.1), 800),
    ]


@dataclass
class Row:
    case: str
    suite: str
    method: str
    mcc: float
    r2_diag: float
    r2_sep: float
    dci: float
    shd: float | None
    seconds: float


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
) -> Row:
    torch.manual_seed(seed)
    generator = suite.make_generator(seed)
    model = factory()
    trainer = SimpleTrainer(
        max_epochs=epochs, lr=1e-3, batch_size=batch_size, steps_per_epoch=15,
        log_every=max(epochs // 10, 1), device=device, seed=seed, grad_clip=1.0,
        scheduler="cosine", val_size=512, val_every=20,
    )
    start = time.perf_counter()
    history = trainer.fit(model, generator)
    seconds = time.perf_counter() - start
    if save_curves:
        exp = out_dir / f"shared_{suite.name}_{method}"
        save_training_report(history, exp, title=f"{suite.name}/{method}")

    resolved = trainer.resolve_device(model)
    batch = move_batch(generator.sample(2048), resolved)
    task = BenchmarkTask(
        name=f"{suite.name}_{method}", model=model, data=batch,
        metrics=["mcc", "r2_diag", "r2_sep", "dci_disentanglement"],
        causal_level="L1_association",
    )
    result = CrlBenchmarkRunner().run(task)
    scores = {m.name: m.value for m in result.metrics}
    shd: float | None = None
    if batch.adjacency is not None:
        with torch.no_grad():
            z_pred = model.encode(batch.x, batch.context)
        max_lag = int(getattr(generator, "max_lag", 1))
        adj_est = estimate_latent_adjacency(z_pred, max_lag=max_lag)
        shd = SHDMetric()(adj_est, batch.adjacency).value
        _ = WSHDMetric()  # available for future reporting
    return Row(
        case=method,
        suite=suite.name,
        method=method,
        mcc=scores["mcc"],
        r2_diag=scores["r2_diag"],
        r2_sep=scores["r2_sep"],
        dci=scores["dci_disentanglement"],
        shd=shd,
        seconds=seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run shared CRL suites.")
    parser.add_argument("--suite", default="npvar")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [
        logging.FileHandler(args.out_dir / "crl_shared.log", mode="w"),
        logging.StreamHandler(),
    ]

    suite = SUITES[args.suite]
    methods = build_methods(suite)
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        for name, factory, epochs in methods:
            row = _run(suite, name, factory, epochs, seed, args.device, args.batch_size,
                       args.out_dir, save_curves=(seed == 0))
            rows.append(asdict(row))
            logger.info(
                "RESULT %s/%s seed=%d mcc=%.3f r2d=%.3f r2s=%.3f dci=%.3f shd=%s",
                suite.name, name, seed, row.mcc, row.r2_diag, row.r2_sep, row.dci,
                "n/a" if row.shd is None else f"{row.shd:.1f}",
            )
    save_rows(rows, args.out_dir, stem=f"crl_shared_{suite.name}")

    aggregated = aggregate_rows(rows)

    def stat(case: str, metric: str) -> str:
        if case not in aggregated or metric not in aggregated[case]:
            return "--"
        entry = aggregated[case][metric]
        return f"{entry['mean']:.3f}±{entry['std']:.3f}"

    logger.info("=== %s mean±std over %d seeds ===", suite.name, args.seeds)
    for name, _, _ in methods:
        logger.info(
            "%-8s mcc=%s r2d=%s r2s=%s dci=%s shd=%s",
            name,
            stat(name, "mcc"),
            stat(name, "r2_diag"),
            stat(name, "r2_sep"),
            stat(name, "dci"),
            stat(name, "shd"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
