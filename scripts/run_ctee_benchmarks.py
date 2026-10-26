"""Counterfactual / treatment-effect benchmarks (harmonic + tumour-growth).

Each representative estimator is trained on factual trajectories and evaluated
against the generator's exact counterfactuals with PEHE, epsilon_ATE, CF MAE,
MBE, MMD^2, and the three structural axioms (reconstruction, effectiveness,
reversibility). Two benchmarks with ground-truth counterfactuals are used: the
damped harmonic oscillator and the tumour-growth simulator of Bica et al.
(2020). Statistical tests are Holm-corrected paired Wilcoxon signed-rank tests
on the per-seed rows.

Usage::

    uv run python scripts/run_ctee_benchmarks.py --device cuda --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from gcmts.core import SimpleTrainer, move_batch, save_training_report
from gcmts.data.sem_effect import TimeVaryingSEMGenerator
from gcmts.data.semi_synthetic import TumourGrowthSimulator
from gcmts.data.synthetic_effect import HarmonicOscillatorGenerator
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.effect_estimation.methods import CEPAE, CRN, GANITE, CaTSG, WuIPTW
from gcmts.evaluation.benchmarks import EMA_DECAY
from gcmts.evaluation.metrics import (
    CFMAEMetric,
    MBEMetric,
    MMD2Metric,
    PEHEMetric,
)
from gcmts.evaluation.progress import ProgressTracker
from gcmts.evaluation.report import (
    aggregate_rows,
    guard_resume,
    load_rows,
    save_rows,
    wall_seconds,
)
from gcmts.evaluation.stats import paired_wilcoxon_holm
from gcmts.evaluation.tracking import StepTracker

logger = logging.getLogger("gcmts.ctee")

Generator = HarmonicOscillatorGenerator | TumourGrowthSimulator


@dataclass
class DatasetConfig:
    name: str
    observed_dim: int
    treatment_dim: int
    horizon: int
    outcome_dim: int
    make_generator: Callable[[int], Generator]


DATASETS: dict[str, DatasetConfig] = {
    "harmonic": DatasetConfig(
        name="harmonic",
        observed_dim=6,
        treatment_dim=1,
        horizon=24,
        outcome_dim=0,
        make_generator=lambda seed: HarmonicOscillatorGenerator(
            n_units=3, observed_dim=6, horizon=24, seed=seed
        ),
    ),
    "tumour": DatasetConfig(
        name="tumour",
        observed_dim=3,
        treatment_dim=2,
        horizon=15,
        outcome_dim=0,
        make_generator=lambda seed: TumourGrowthSimulator(horizon=15, seed=seed),
    ),
    "linear_sem": DatasetConfig(
        name="linear_sem",
        observed_dim=6,
        treatment_dim=1,
        horizon=20,
        outcome_dim=0,
        make_generator=lambda seed: TimeVaryingSEMGenerator(
            n_covariates=4, horizon=20, seed=seed
        ),
    ),
}

# Representative methods per branch, matching the review's taxonomy:
# GANITE (II.A observed-data adjustment), CRN (II.B balanced representation),
# CEPAE (II.B disentangled entropy-penalised), CaTSG (II.C latent environments).
METHODS: dict[str, tuple[Callable[[DatasetConfig], BaseEffectEstimator], int, str]] = {
    "catsg": (
        lambda d: CaTSG(
            observed_dim=d.observed_dim,
            treatment_dim=d.treatment_dim,
            horizon=d.horizon,
            n_steps=20,
        ),
        800,
        "score-guided conditional diffusion (abduction via DDIM inversion)",
    ),
    "cepae": (
        lambda d: CEPAE(
            observed_dim=d.observed_dim,
            treatment_dim=d.treatment_dim,
            latent_dim=max(d.observed_dim, 4),
        ),
        500,
        "variational autoencoder with adversarial conditional-entropy penalty",
    ),
    "crn": (
        lambda d: CRN(
            observed_dim=d.observed_dim,
            treatment_dim=d.treatment_dim,
            latent_dim=max(d.observed_dim, 4),
            horizon=d.horizon,
        ),
        500,
        "sequential latent-dynamics VAE (treatment enters the transition)",
    ),
    "ganite": (
        lambda d: GANITE(
            observed_dim=d.observed_dim,
            treatment_dim=d.treatment_dim,
            noise_dim=4,
        ),
        600,
        "adversarial generator/discriminator with abducted noise",
    ),
    "wu_iptw": (
        lambda d: WuIPTW(
            observed_dim=d.observed_dim,
            treatment_dim=d.treatment_dim,
            latent_dim=max(d.observed_dim, 4),
        ),
        500,
        "IPTW-reweighted conditional VAE (g-formula pseudo-population)",
    ),
}


@dataclass
class CaseConfig:
    name: str
    dataset: DatasetConfig
    model_factory: Callable[[], BaseEffectEstimator]
    epochs: int
    note: str = ""


@dataclass
class CaseResult:
    case: str
    dataset: str
    method: str
    seed: int
    epochs: int
    n_parameters: int
    final_loss: float
    pehe: float
    ate_error: float
    cf_mae: float
    cf_mbe: float
    mmd2: float
    recon_axiom: float
    effectiveness: float
    effectiveness_ok: float
    reversibility: float
    pehe_zero: float
    pehe_mean: float
    ate_zero: float
    seconds: float
    device: str = "auto"


def build_cases(
    datasets: list[str] | None = None, methods: list[str] | None = None
) -> list[CaseConfig]:
    cases: list[CaseConfig] = []
    for dataset_name in datasets or list(DATASETS):
        dataset = DATASETS[dataset_name]
        for method_name, (factory, epochs, note) in METHODS.items():
            if methods is not None and method_name not in methods:
                continue
            cases.append(
                CaseConfig(
                    name=f"{method_name}_{dataset_name}",
                    dataset=dataset,
                    model_factory=lambda d=dataset, f=factory: f(d),
                    epochs=epochs,
                    note=note,
                )
            )
    return cases


def _outcome(traj: torch.Tensor, dim: int) -> torch.Tensor:
    return traj[:, -1, dim]


def evaluate(
    model: BaseEffectEstimator, generator: Generator, device: torch.device
) -> dict[str, float]:
    batch = move_batch(generator.sample(1024), device)
    action = batch.context["A"].to(device)
    zeros = torch.zeros_like(action)

    cf_hat = model.counterfactual(batch, {"A": zeros}).counterfactual
    factual_hat = model.counterfactual(batch, {"A": action}).counterfactual
    true_cf = batch.counterfactual

    y_cf_hat = _outcome(cf_hat, generator.outcome_dim)
    y_f_hat = _outcome(factual_hat, generator.outcome_dim)
    y_cf_true = _outcome(true_cf, generator.outcome_dim)
    y_f_true = batch.outcome

    ite_hat = y_cf_hat - y_f_hat
    ite_true = y_cf_true - y_f_true
    pehe = PEHEMetric()(ite_hat, ite_true).value
    ate_error = abs(ite_hat.mean().item() - ite_true.mean().item())
    # Reference predictors that no model can beat without estimating the
    # individual effect: predicting no effect and predicting the oracle mean
    # effect.  Without these rows a PEHE of 1.1 on a benchmark whose ITE
    # standard deviation is 0.3 looks like a modelling result rather than a
    # failure to learn anything.
    pehe_zero = float(torch.sqrt((ite_true**2).mean()).item())
    pehe_mean = float(ite_true.std(unbiased=False).item())
    ate_zero = float(abs(ite_true.mean().item()))
    cf_mae = CFMAEMetric()(cf_hat, true_cf).value
    cf_mbe = MBEMetric()(cf_hat, true_cf).value
    mmd2 = MMD2Metric()(cf_hat[:256], true_cf[:256]).value
    recon_axiom = CFMAEMetric()(factual_hat, batch.x).value
    effectiveness = ite_hat.mean().item() - ite_true.mean().item()
    effectiveness_ok = float(ite_hat.mean().item() * ite_true.mean().item() > 0)

    # Reversibility: start from the estimated counterfactual (treatment off) and
    # apply the factual treatment; a reversible model recovers the factual state.
    cf_batch = replace(batch, x=cf_hat, context={"A": zeros})
    recovered = model.counterfactual(cf_batch, {"A": action}).counterfactual
    reversibility = CFMAEMetric()(recovered, batch.x).value
    return {
        "pehe": pehe,
        "ate_error": ate_error,
        "cf_mae": cf_mae,
        "cf_mbe": cf_mbe,
        "mmd2": mmd2,
        "recon_axiom": recon_axiom,
        "effectiveness": effectiveness,
        "effectiveness_ok": effectiveness_ok,
        "reversibility": reversibility,
        "pehe_zero": pehe_zero,
        "pehe_mean": pehe_mean,
        "ate_zero": ate_zero,
    }


def run_case(
    cfg: CaseConfig,
    device: str,
    batch_size: int,
    out_dir: Path,
    *,
    seed: int = 0,
    save_curves: bool = True,
) -> CaseResult:
    logger.info("=" * 72)
    logger.info("CASE %s seed=%d | %s", cfg.name, seed, cfg.note)
    torch.manual_seed(seed)
    generator = cfg.dataset.make_generator(seed)
    model = cfg.model_factory()
    n_params = sum(p.numel() for p in model.parameters())
    exp_dir = out_dir / cfg.name
    trainer = SimpleTrainer(
        max_epochs=cfg.epochs,
        lr=1e-3,
        batch_size=batch_size,
        steps_per_epoch=15,
        log_every=max(cfg.epochs // 10, 1),
        device=device,
        seed=seed,
        grad_clip=1.0,
        scheduler="cosine",
        val_size=512,
        val_every=20,
        ema_decay=EMA_DECAY,
    )
    start = time.perf_counter()
    history = trainer.fit(model, generator)
    seconds = wall_seconds(start)
    if save_curves:
        save_training_report(history, exp_dir, title=f"{cfg.name} ({type(model).__name__})")
        model.save(exp_dir / "model.pt")

    resolved = trainer.resolve_device(model)
    scores = evaluate(model, generator, resolved)
    logger.info(
        "RESULT %s | %s (%.1fs)",
        cfg.name,
        " ".join(f"{k}={v:.4f}" for k, v in scores.items()),
        seconds,
    )
    return CaseResult(
        case=cfg.name,
        dataset=cfg.dataset.name,
        method=type(model).__name__,
        seed=seed,
        epochs=cfg.epochs,
        n_parameters=n_params,
        final_loss=history["loss"][-1] if history.get("loss") else float("nan"),
        seconds=seconds,
        device=str(resolved),
        **scores,
    )


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [
        logging.FileHandler(out_dir / "ctee_benchmarks.log", mode="w"),
        logging.StreamHandler(),
    ]


def _report_stats(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Holm-corrected paired Wilcoxon tests per dataset for the key metrics."""
    if not rows:
        return
    summary: dict[str, Any] = {}
    logger.info("=== Holm-corrected paired Wilcoxon (paired by dataset, seed) ===")
    for metric in ("pehe", "ate_error", "cf_mae"):
        comparisons = paired_wilcoxon_holm(rows, metric, seed_key="seed")
        summary[metric] = [asdict(c) for c in comparisons]
        for comparison in comparisons:
            logger.info(
                "%-8s %-10s vs %-10s n=%d p=%.4g p_holm=%.4g%s",
                metric,
                comparison.case_a,
                comparison.case_b,
                comparison.n_pairs,
                comparison.p_value,
                comparison.p_holm,
                " *" if comparison.significant else "",
            )
    (out_dir / "ctee_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CTEE benchmarks.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds")
    parser.add_argument(
        "--epochs-scale", type=float, default=1.0, help="smoke-test budget multiplier"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="skip runs already present in the output JSON (default: resume)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="resume even when the stored configuration or code revision differs",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    configure_logging(args.out_dir)
    cases = build_cases(args.datasets, args.methods)
    for case in cases:
        case.epochs = max(2, int(round(case.epochs * args.epochs_scale)))
    if args.only:
        cases = [c for c in cases if c.name in set(args.only)]
    stem = "ctee_benchmarks"
    rows: list[dict[str, Any]] = load_rows(args.out_dir, stem) if args.resume else []
    ok, reason = guard_resume(
        args.out_dir, stem,
        {"cases": [c.name for c in cases], "seeds": args.seeds, "batch_size": args.batch_size},
        force=args.force,
    )
    if not ok:
        logger.error(
            "Refusing to resume %s: %s. Re-run with --force to append anyway, "
            "or delete %s to start over.", stem, reason, args.out_dir / f"{stem}.json",
        )
        return 2
    done = {(row.get("case"), row.get("seed")) for row in rows}
    progress = ProgressTracker(
        args.out_dir / f"{stem}_progress.json",
        "CTEE temporal",
        args.seeds * len(cases),
    )
    progress.done = len(rows)
    progress.start()
    step = StepTracker.start(
        name="CTEE temporal",
        params={"seeds": args.seeds, "cases": [c.name for c in cases]},
        tags=["ctee", "temporal"],
        progress=progress,
    )
    for seed in range(args.seeds):
        for cfg in cases:
            if (cfg.name, seed) in done:
                continue
            result = run_case(
                cfg, args.device, args.batch_size, args.out_dir,
                seed=seed, save_curves=(seed == 0),
            )
            rows.append(asdict(result))
            # Incremental save: long runs remain analysable if interrupted.
            save_rows(rows, args.out_dir, stem="ctee_benchmarks")
            progress.advance()
            step.log_run(
                method=result.method,
                iteration=seed,
                params={"dataset": result.dataset},
                metrics={
                    "pehe": result.pehe,
                    "ate_error": result.ate_error,
                    "cf_mae": result.cf_mae,
                    "cf_mbe": result.cf_mbe,
                    "mmd2": result.mmd2,
                    "recon_axiom": result.recon_axiom,
                    "effectiveness": result.effectiveness,
                    "reversibility": result.reversibility,
                },
            )
    progress.finish()
    step.close()
    save_rows(rows, args.out_dir, stem="ctee_benchmarks")
    logger.info(
        "%-20s %14s %14s %14s %14s", "case", "PEHE", "eATE", "CFMAE", "MBE"
    )
    aggregated = aggregate_rows(rows)

    def stat(metrics: dict[str, Any], name: str) -> str:
        if name not in metrics:
            return "     --     "
        return f"{metrics[name]['mean']:.3f}±{metrics[name]['std']:.3f}"

    for case, metrics in aggregated.items():
        logger.info(
            "%-20s %14s %14s %14s %14s",
            case,
            stat(metrics, "pehe"),
            stat(metrics, "ate_error"),
            stat(metrics, "cf_mae"),
            stat(metrics, "cf_mbe"),
        )
    _report_stats(rows, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
