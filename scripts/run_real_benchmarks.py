"""External-validity benchmarks on real time series.

Real series carry no latent ground truth, so representation metrics (MCC, R2,
SHD) are undefined; per rule R1 we report only *external-validity* proxies:
reconstruction and forecasting MSE on a held-out, chronologically later split.
A method that learns a useful causal representation should forecast the future
of a real series better than a trivial baseline.

Usage::

    uv run python scripts/download_datasets.py
    uv run python scripts/run_real_benchmarks.py --device cuda --seeds 3
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.causal_representation_learning.methods import IVAE, LEAP, MOSAIC, NCTRL, TDRL
from gcmts.core import SimpleTrainer, save_training_report
from gcmts.data.real import RealTimeSeries, RealTSGenerator
from gcmts.evaluation.report import aggregate_rows, save_rows

logger = logging.getLogger("gcmts.real")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real"


@dataclass
class DatasetSpec:
    name: str
    file: str
    horizon: int
    subsample: int = 1
    channels: list[str] | None = None


DATASETS = {
    "etth1": DatasetSpec("etth1", "ETTh1.csv", horizon=24),
    "etth2": DatasetSpec("etth2", "ETTh2.csv", horizon=24),
    "jena": DatasetSpec("jena", "jena_climate_2009_2016.csv", horizon=24, subsample=6),
}


def build_methods(
    observed_dim: int, latent_dim: int
) -> list[tuple[str, Callable[[], BaseCausalRepresentationLearner], int]]:
    fix = max(latent_dim - 2, 1)
    obs = 1
    dyn = max(latent_dim - fix - obs, 1)
    return [
        (
            "iVAE",
            lambda: IVAE(observed_dim=observed_dim, latent_dim=latent_dim, u_dim=0, obs_noise=0.5),
            1000,
        ),
        (
            "LEAP",
            lambda: LEAP(
                observed_dim=observed_dim,
                latent_dim=latent_dim,
                max_lag=1,
                u_dim=0,
                kl_weight=0.01,
                obs_noise=0.5,
            ),
            1000,
        ),
        (
            "TDRL",
            lambda: TDRL(
                observed_dim=observed_dim,
                latent_fix_dim=fix,
                latent_dyn_dim=dyn,
                latent_obs_dim=obs,
                u_dim=0,
                kl_weight=0.01,
                obs_noise=0.5,
            ),
            1000,
        ),
        (
            "NCTRL",
            lambda: NCTRL(
                observed_dim=observed_dim,
                latent_dim=latent_dim,
                n_regimes=2,
                kl_weight=0.01,
                obs_noise=0.5,
            ),
            1000,
        ),
        (
            "MOSAIC",
            lambda: MOSAIC(
                observed_dim=observed_dim,
                latent_dim=latent_dim,
                u_dim=0,
                kl_weight=0.1,
                obs_noise=0.5,
            ),
            1000,
        ),
    ]


def _forecast_mse(
    model: BaseCausalRepresentationLearner, windows: Tensor, steps: int, device: torch.device
) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, windows.shape[0], 256):
            x = windows[start : start + 256].to(device)
            inp = x[:, : x.shape[1] - steps]
            pred = model.predict_next_observations(
                inp, steps, {"split": torch.zeros(x.shape[0], device=device)}
            )
            target = x[:, x.shape[1] - steps :]
            total += float(((pred - target) ** 2).sum())
            count += target.numel()
    return total / max(count, 1)


def _recon_mse(
    model: BaseCausalRepresentationLearner, windows: Tensor, device: torch.device
) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, windows.shape[0], 256):
            x = windows[start : start + 256].to(device)
            z = model.encode(x, {"split": torch.zeros(x.shape[0], device=device)})
            xr = model.decode(z)
            total += float(((xr - x) ** 2).sum())
            count += x.numel()
    return total / max(count, 1)


def run_case(
    spec: DatasetSpec,
    method: str,
    factory: Callable[[], BaseCausalRepresentationLearner],
    epochs: int,
    seed: int,
    device: str,
    out_dir: Path,
    save_curves: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    series = RealTimeSeries(
        DATA_DIR / spec.file,
        horizon=spec.horizon,
        subsample=spec.subsample,
        channels=spec.channels,
        seed=seed,
    )
    model = factory()
    trainer = SimpleTrainer(
        max_epochs=epochs,
        lr=1e-3,
        batch_size=256,
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
    history = trainer.fit(model, RealTSGenerator(series))
    seconds = time.perf_counter() - start
    if save_curves:
        save_training_report(
            history, out_dir / f"real_{spec.name}_{method}", title=f"{spec.name}/{method}"
        )

    resolved = trainer.resolve_device(model)
    test = series.windows["test"]
    recon = _recon_mse(model, test, resolved)
    forecast1 = _forecast_mse(model, test, 1, resolved)
    forecast4 = _forecast_mse(model, test, 4, resolved)
    logger.info(
        "RESULT %s/%s seed=%d recon=%.3f f1=%.3f f4=%.3f (%.1fs)",
        spec.name,
        method,
        seed,
        recon,
        forecast1,
        forecast4,
        seconds,
    )
    return {
        "case": f"{spec.name}/{method}",
        "dataset": spec.name,
        "method": method,
        "recon_mse": recon,
        "forecast1_mse": forecast1,
        "forecast4_mse": forecast4,
        "seconds": seconds,
    }


def baseline_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    """Persistence and ridge AR(1) baselines on the held-out test split."""
    from sklearn.linear_model import Ridge

    series = RealTimeSeries(
        DATA_DIR / spec.file,
        horizon=spec.horizon,
        subsample=spec.subsample,
        channels=spec.channels,
        seed=0,
    )
    train = series.windows["train"]
    test = series.windows["test"]

    def persistence(windows: Tensor, steps: int) -> float:
        last = windows[:, windows.shape[1] - steps - 1 : windows.shape[1] - steps]
        pred = last.expand(-1, steps, -1)
        target = windows[:, windows.shape[1] - steps :]
        return float(((pred - target) ** 2).mean())

    flat_x = train[:, :-1].reshape(-1, train.shape[-1]).numpy()
    flat_y = train[:, 1:].reshape(-1, train.shape[-1]).numpy()
    ar = Ridge(alpha=1e-3).fit(flat_x, flat_y)
    test_x = test[:, :-1].reshape(-1, test.shape[-1]).numpy()
    test_y = test[:, 1:].reshape(-1, test.shape[-1]).numpy()
    ar_mse = float(((ar.predict(test_x) - test_y) ** 2).mean())

    return [
        {
            "case": f"{spec.name}/Persistence",
            "dataset": spec.name,
            "method": "Persistence",
            "recon_mse": float("nan"),
            "forecast1_mse": persistence(test, 1),
            "forecast4_mse": persistence(test, 4),
            "seconds": 0.0,
        },
        {
            "case": f"{spec.name}/AR(1)",
            "dataset": spec.name,
            "method": "AR(1)",
            "recon_mse": float("nan"),
            "forecast1_mse": ar_mse,
            "forecast4_mse": float("nan"),
            "seconds": 0.0,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-data external-validity benchmarks.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--datasets", nargs="*", default=["etth1", "jena"])
    parser.add_argument("--only", nargs="*", default=None, help="subset of method names")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [
        logging.FileHandler(args.out_dir / "real_benchmarks.log", mode="w"),
        logging.StreamHandler(),
    ]

    rows: list[dict[str, Any]] = []
    for name in args.datasets:
        spec = DATASETS[name]
        probe = RealTimeSeries(
            DATA_DIR / spec.file,
            horizon=spec.horizon,
            subsample=spec.subsample,
            channels=spec.channels,
        )
        latent_dim = max(min(probe.observed_dim, 8), 3)
        rows.extend(baseline_rows(spec))
        methods = build_methods(probe.observed_dim, latent_dim)
        if args.only:
            methods = [m for m in methods if m[0] in set(args.only)]
        for seed in range(args.seeds):
            for method, factory, epochs in methods:
                rows.append(
                    run_case(
                        spec,
                        method,
                        factory,
                        epochs,
                        seed,
                        args.device,
                        args.out_dir,
                        save_curves=(seed == 0),
                    )
                )

    save_rows(rows, args.out_dir, stem="real_benchmarks")
    aggregated = aggregate_rows(rows)

    def stat(case: str, metric: str) -> str:
        if case not in aggregated or metric not in aggregated[case]:
            return "--"
        entry = aggregated[case][metric]
        return f"{entry['mean']:.4f}\u00b1{entry['std']:.4f}"

    logger.info("=== real-data external validity (mean\u00b1std over %d seeds) ===", args.seeds)
    for case in aggregated:
        logger.info(
            "%-16s recon=%s f1=%s f4=%s",
            case,
            stat(case, "recon_mse"),
            stat(case, "forecast1_mse"),
            stat(case, "forecast4_mse"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
