"""Static counterfactual / treatment-effect benchmarks (IHDP, ACIC).

Implements the canonical IHDP protocol: each of the 100 replications is a
separate benchmark instance; models train on the standard train split and are
scored on the test split against the noiseless potential means ``mu0``/``mu1``
with PEHE and epsilon_ATE, plus the policy risk of the induced rule. Because
the loaders expose ground-truth counterfactuals (rule R1), the static
benchmark is sound at Level 3; it complements the temporal harmonic and
tumour-growth benchmarks where treatments are time-varying.

Baselines: a ridge T-learner and a nearest-neighbour matching estimator, the
two standard non-parametric references for IHDP.

Usage::

    uv run python scripts/run_ctee_static.py --device cpu --replications 100
    uv run python scripts/run_ctee_static.py --dataset acic --acic-dir data/acic
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
from scipy.spatial.distance import cdist

from gcmts.core import SimpleTrainer, move_batch, save_training_report
from gcmts.data.real_panels import load_lalonde, load_nhefs, load_oregon
from gcmts.data.semi_synthetic import (
    ACICLoader,
    IHDPGenerator,
    IHDPLoader,
    IHDPReplication,
)
from gcmts.data.static_effect import STATIC_CATE_BENCHMARKS, StaticCateLoader
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.effect_estimation.methods import CEPAE, CRN, GANITE, CaTSG
from gcmts.evaluation.benchmarks import CTEE_REAL_BENCHMARKS
from gcmts.evaluation.metrics import PolicyRiskMetric
from gcmts.evaluation.progress import ProgressTracker
from gcmts.evaluation.report import aggregate_rows, save_rows, wall_seconds
from gcmts.evaluation.stats import paired_wilcoxon_holm
from gcmts.evaluation.tracking import StepTracker

logger = logging.getLogger("gcmts.ctee_static")

Replication = IHDPReplication

# Real effect cohorts: only the marginal (ATE) effect is exposed, so individual
# metrics (PEHE, policy risk) are undefined and reported as n/a.
REAL_COHORTS: dict[str, tuple[str, Callable[[Path], object]]] = {
    "lalonde_rct": ("nsw_mixtape.dta", load_lalonde),
    "oregon_rct": ("dataverse_files/Oregon/oregon_puf.zip", load_oregon),
    "nhefs": ("NHEFS.csv", load_nhefs),
}
_REFERENCE_ATE: dict[str, float] = {}

METHODS: dict[str, tuple[Callable[[IHDPGenerator], BaseEffectEstimator], int, str]] = {
    "catsg": (
        lambda g: CaTSG(
            observed_dim=g.observed_dim, treatment_dim=1, horizon=1, n_steps=20
        ),
        800,
        "score-guided conditional diffusion (II.C)",
    ),
    "cepae": (
        lambda g: CEPAE(observed_dim=g.observed_dim, treatment_dim=1, latent_dim=8),
        500,
        "conditional entropy-penalised autoencoder (II.B)",
    ),
    "crn": (
        lambda g: CRN(
            observed_dim=g.observed_dim, treatment_dim=1, latent_dim=8, horizon=1
        ),
        500,
        "balanced sequential representation (II.B)",
    ),
    "ganite": (
        lambda g: GANITE(observed_dim=g.observed_dim, treatment_dim=1, noise_dim=4),
        600,
        "adversarial counterfactual generator (II.A)",
    ),
}


@dataclass
class StaticResult:
    case: str
    dataset: str
    method: str
    seed: int
    n_parameters: int
    final_loss: float
    pehe: float
    ate_error: float
    policy_risk: float
    seconds: float


def _load_replication(dataset: str, index: int, directory: Path) -> Replication:
    if dataset == "ihdp":
        train_file = directory / "ihdp_npci_1-100.train.npz"
        test_file = directory / "ihdp_npci_1-100.test.npz"
        loader = IHDPLoader(train_file, test_file)
        return loader.replication(index, split="test")
    if dataset in STATIC_CATE_BENCHMARKS:
        loader = StaticCateLoader(dataset, replications=index + 1, seed=0)
        return loader.load()[index]
    if dataset in REAL_COHORTS:
        filename, cohort_loader = REAL_COHORTS[dataset]
        cohort = cohort_loader(directory / filename)
        _REFERENCE_ATE[dataset] = cohort.reference_ate
        n = cohort.y.shape[0]
        nan = np.full(n, np.nan, dtype=np.float64)
        return Replication(
            x=cohort.x.astype(np.float32),
            t=cohort.t.astype(np.int64),
            yf=cohort.y.astype(np.float32),
            ycf=nan,
            mu0=nan,
            mu1=nan,
        )
    loader = ACICLoader(directory)
    names = sorted(
        p for p in directory.iterdir() if p.is_dir() and (p / "x.csv").exists()
    )
    if not names:
        raise FileNotFoundError(
            f"No ACIC replication directories under {directory}. "
            "Download 2018 from Zenodo (record 19175621) and convert first."
        )
    name = names[index % len(names)].name
    item = loader.replication(name)
    yf = item["yf"]
    t = item["t"]
    mu0 = item.get("mu0", np.zeros_like(yf))
    mu1 = item.get("mu1", np.zeros_like(yf))
    return Replication(
        x=item["x"],
        t=t,
        yf=yf,
        ycf=np.where(t > 0, mu0, mu1),
        mu0=mu0,
        mu1=mu1,
    )


def _tau_from_model(
    model: BaseEffectEstimator, generator: IHDPGenerator, device: torch.device
) -> tuple[np.ndarray, float]:
    batch = move_batch(generator.sample_all(), device)
    action = batch.context["A"]
    with torch.no_grad():
        y1 = model.counterfactual(batch, {"A": torch.ones_like(action)}).counterfactual
        y0 = model.counterfactual(batch, {"A": torch.zeros_like(action)}).counterfactual
    y1 = generator.rescale_outcome(y1[:, -1, -1].detach().cpu())
    y0 = generator.rescale_outcome(y0[:, -1, -1].detach().cpu())
    return (y1 - y0).numpy(), float(((y1 - y0).pow(2)).mean().item())


def _metrics(
    tau_hat: np.ndarray, replication: Replication, *, reference_ate: float | None = None
) -> dict[str, float]:
    if reference_ate is not None:
        return {
            "pehe": float("nan"),
            "ate_error": float(abs(np.mean(tau_hat) - reference_ate)),
            "policy_risk": float("nan"),
        }
    tau_true = replication.mu1 - replication.mu0
    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
    ate_error = float(abs(np.mean(tau_hat) - np.mean(tau_true)))
    policy = PolicyRiskMetric()(
        torch.from_numpy(tau_hat).float(),
        {
            "yf": torch.from_numpy(replication.yf).float(),
            "ycf": torch.from_numpy(replication.ycf).float(),
            "t": torch.from_numpy(replication.t).float(),
        },
    ).value
    return {"pehe": pehe, "ate_error": ate_error, "policy_risk": policy}


def _ridge_t_learner(
    train: IHDPGenerator, test: IHDPGenerator
) -> np.ndarray:
    from sklearn.linear_model import Ridge

    x = train.covariates.numpy()
    y = train.outcomes.numpy()
    t = train.treatment.numpy()
    model_1 = Ridge(alpha=1.0).fit(x[t > 0], y[t > 0])
    model_0 = Ridge(alpha=1.0).fit(x[t <= 0], y[t <= 0])
    x_test = test.covariates.numpy()
    return (model_1.predict(x_test) - model_0.predict(x_test)) * float(train.y_std)


def _nearest_neighbour(
    train: IHDPGenerator, test: IHDPGenerator
) -> np.ndarray:
    x = train.covariates.numpy()
    y = train.outcomes.numpy()
    t = train.treatment.numpy()
    x_test = test.covariates.numpy()
    treated, control = x[t > 0], x[t <= 0]
    y_treated, y_control = y[t > 0], y[t <= 0]
    nearest_treated = cdist(x_test, treated).argmin(axis=1)
    nearest_control = cdist(x_test, control).argmin(axis=1)
    return (y_treated[nearest_treated] - y_control[nearest_control]) * float(train.y_std)


def run_replication(
    dataset: str,
    index: int,
    methods: list[str],
    device: str,
    batch_size: int,
    out_dir: Path,
    data_dir: Path,
    *,
    save_curves: bool = False,
) -> list[StaticResult]:
    replication = _load_replication(dataset, index, data_dir)
    results: list[StaticResult] = []
    for method_name in methods:
        factory, epochs, note = METHODS[method_name]
        logger.info("=" * 72)
        logger.info(
            "CASE %s_%s replication=%d | %s", method_name, dataset, index, note
        )
        torch.manual_seed(index)
        train_generator = IHDPGenerator(replication, seed=index)
        test_generator = IHDPGenerator(replication, seed=index + 10_000)
        model = factory(train_generator)
        n_params = sum(p.numel() for p in model.parameters())
        exp_dir = out_dir / f"{method_name}_{dataset}_{index}"
        trainer = SimpleTrainer(
            max_epochs=epochs,
            lr=1e-3,
            batch_size=batch_size,
            steps_per_epoch=15,
            log_every=max(epochs // 5, 1),
            device=device,
            seed=index,
            grad_clip=1.0,
            scheduler="cosine",
            val_size=512,
            val_every=20,
        )
        start = time.perf_counter()
        history = trainer.fit(model, train_generator)
        seconds = wall_seconds(start)
        if save_curves:
            save_training_report(history, exp_dir, title=f"{method_name}_{dataset}")
        resolved = trainer.resolve_device(model)
        tau_hat, _ = _tau_from_model(model, test_generator, resolved)
        metrics = _metrics(tau_hat, replication, reference_ate=_REFERENCE_ATE.get(dataset))
        logger.info(
            "RESULT %s_%s rep=%d pehe=%.3f ate=%.3f risk=%.3f (%.1fs)",
            method_name,
            dataset,
            index,
            metrics["pehe"],
            metrics["ate_error"],
            metrics["policy_risk"],
            seconds,
        )
        results.append(
            StaticResult(
                case=f"{method_name}_{dataset}",
                dataset=dataset,
                method=method_name,
                seed=index,
                n_parameters=n_params,
                final_loss=history["loss"][-1] if history.get("loss") else float("nan"),
                seconds=seconds,
                **metrics,
            )
        )
    for baseline_name, estimator in (
        ("ridge_t", _ridge_t_learner),
        ("nearest_neighbour", _nearest_neighbour),
    ):
        tau_hat = estimator(train_generator, test_generator)
        metrics = _metrics(tau_hat, replication, reference_ate=_REFERENCE_ATE.get(dataset))
        logger.info(
            "RESULT %s_%s rep=%d pehe=%.3f ate=%.3f risk=%.3f (baseline)",
            baseline_name,
            dataset,
            index,
            metrics["pehe"],
            metrics["ate_error"],
            metrics["policy_risk"],
        )
        results.append(
            StaticResult(
                case=f"{baseline_name}_{dataset}",
                dataset=dataset,
                method=baseline_name,
                seed=index,
                n_parameters=0,
                final_loss=float("nan"),
                seconds=0.0,
                **metrics,
            )
        )
    return results


def configure_logging(out_dir: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [
        logging.FileHandler(out_dir / f"ctee_static_{dataset}.log", mode="w"),
        logging.StreamHandler(),
    ]


def _report_stats(rows: list[dict[str, Any]], out_dir: Path, dataset: str) -> None:
    summary: dict[str, Any] = {}
    logger.info("=== Holm-corrected paired Wilcoxon (paired by replication) ===")
    for metric in ("pehe", "ate_error", "policy_risk"):
        comparisons = paired_wilcoxon_holm(rows, metric, seed_key="seed")
        summary[metric] = [asdict(c) for c in comparisons]
        for comparison in comparisons:
            logger.info(
                "%-13s %-16s vs %-16s n=%d p=%.4g p_holm=%.4g%s",
                metric,
                comparison.case_a,
                comparison.case_b,
                comparison.n_pairs,
                comparison.p_value,
                comparison.p_holm,
                " *" if comparison.significant else "",
            )
    (out_dir / f"ctee_static_{dataset}_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run static CTEE benchmarks.")
    parser.add_argument(
        "--dataset",
        choices=["ihdp", "acic", *STATIC_CATE_BENCHMARKS, *CTEE_REAL_BENCHMARKS],
        default="ihdp",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.data_dir is not None:
        data_dir = args.data_dir
    elif args.dataset == "ihdp":
        data_dir = Path("data/semi_synthetic")
    elif args.dataset in REAL_COHORTS:
        data_dir = Path("data/real")
    else:
        data_dir = Path("data/acic")
    configure_logging(args.out_dir, args.dataset)
    methods = args.methods or list(METHODS)
    rows: list[dict[str, Any]] = []
    total = args.replications * len(methods)
    progress = ProgressTracker(
        args.out_dir / f"ctee_static_{args.dataset}_progress.json",
        f"CTEE static {args.dataset}",
        total,
    ).start()
    step = StepTracker.start(
        name=f"CTEE static {args.dataset.upper()}",
        params={
            "dataset": args.dataset,
            "replications": args.replications,
            "methods": list(methods),
        },
        tags=["ctee", "static", args.dataset],
        progress=progress,
    )
    for index in range(args.replications):
        results = run_replication(
            args.dataset,
            index,
            methods,
            args.device,
            args.batch_size,
            args.out_dir,
            data_dir,
            save_curves=False,
        )
        rows.extend(asdict(result) for result in results)
        save_rows(rows, args.out_dir, stem=f"ctee_static_{args.dataset}")
        progress.advance(len(results))
        for result in results:
            step.log_run(
                method=result.method,
                iteration=index,
                metrics={
                    "pehe": result.pehe,
                    "ate_error": result.ate_error,
                    "policy_risk": result.policy_risk,
                },
            )
    progress.finish()
    step.close()
    aggregated = aggregate_rows(rows)
    logger.info("%-24s %12s %12s %12s", "case", "PEHE", "eATE", "Risk")
    for case, metrics in aggregated.items():
        logger.info(
            "%-24s %12.3f %12.3f %12.3f",
            case,
            metrics.get("pehe", {}).get("mean", float("nan")),
            metrics.get("ate_error", {}).get("mean", float("nan")),
            metrics.get("policy_risk", {}).get("mean", float("nan")),
        )
    _report_stats(rows, args.out_dir, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
