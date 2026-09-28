"""Counterfactual / treatment-effect benchmarks on the harmonic oscillator.

Each representative estimator is trained on factual trajectories and evaluated
against the generator's exact counterfactuals with PEHE, epsilon_ATE, CF MAE,
MBE, MMD^2, and the three structural axioms (reconstruction, effectiveness,
reversibility).

Usage::

    uv run python scripts/run_ctee_benchmarks.py --device cuda
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

from gcmts.core import SimpleTrainer, move_batch, save_training_report
from gcmts.data.synthetic_effect import HarmonicOscillatorGenerator
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.effect_estimation.methods import CEPAE, CRN, GANITE, CaTSG
from gcmts.evaluation.metrics import (
    CFMAEMetric,
    MBEMetric,
    MMD2Metric,
    PEHEMetric,
)
from gcmts.evaluation.report import aggregate_rows, save_rows

logger = logging.getLogger("gcmts.ctee")


@dataclass
class CaseConfig:
    name: str
    model_factory: Callable[[], BaseEffectEstimator]
    epochs: int
    note: str = ""


@dataclass
class CaseResult:
    case: str
    method: str
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
    seconds: float
    device: str = "auto"


def build_cases() -> list[CaseConfig]:
    return [
        CaseConfig(
            name="catsg_harmonic",
            model_factory=lambda: CaTSG(observed_dim=6, treatment_dim=1, horizon=24, n_steps=20),
            epochs=800,
            note="score-guided conditional diffusion (abduction via DDIM inversion)",
        ),
        CaseConfig(
            name="cepae_harmonic",
            model_factory=lambda: CEPAE(observed_dim=6, treatment_dim=1, latent_dim=4),
            epochs=500,
            note="variational autoencoder with adversarial conditional-entropy penalty",
        ),
        CaseConfig(
            name="crn_harmonic",
            model_factory=lambda: CRN(observed_dim=6, treatment_dim=1, latent_dim=6, horizon=24),
            epochs=500,
            note="sequential latent-dynamics VAE (treatment enters the transition)",
        ),
        CaseConfig(
            name="ganite_harmonic",
            model_factory=lambda: GANITE(observed_dim=6, treatment_dim=1, noise_dim=4),
            epochs=600,
            note="adversarial generator/discriminator with abducted noise",
        ),
    ]


def _outcome(traj: torch.Tensor, dim: int) -> torch.Tensor:
    return traj[:, -1, dim]


def evaluate(
    model: BaseEffectEstimator, generator: HarmonicOscillatorGenerator, device: torch.device
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
    cf_mae = CFMAEMetric()(cf_hat, true_cf).value
    cf_mbe = MBEMetric()(cf_hat, true_cf).value
    mmd2 = MMD2Metric()(cf_hat[:256], true_cf[:256]).value
    recon_axiom = CFMAEMetric()(factual_hat, batch.x).value
    effectiveness = ite_hat.mean().item() - ite_true.mean().item()
    return {
        "pehe": pehe,
        "ate_error": ate_error,
        "cf_mae": cf_mae,
        "cf_mbe": cf_mbe,
        "mmd2": mmd2,
        "recon_axiom": recon_axiom,
        "effectiveness": effectiveness,
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
    generator = HarmonicOscillatorGenerator(n_units=3, observed_dim=6, horizon=24, seed=seed)
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
    )
    start = time.perf_counter()
    history = trainer.fit(model, generator)
    seconds = time.perf_counter() - start
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
        method=type(model).__name__,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CTEE benchmarks.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    configure_logging(args.out_dir)
    cases = build_cases()
    if args.only:
        cases = [c for c in cases if c.name in set(args.only)]
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        for cfg in cases:
            result = run_case(
                cfg, args.device, args.batch_size, args.out_dir,
                seed=seed, save_curves=(seed == 0),
            )
            rows.append(asdict(result))
    save_rows(rows, args.out_dir, stem="ctee_benchmarks")
    logger.info("%-18s %14s %14s %14s %14s", "case", "PEHE", "eATE", "CFMAE", "MBE")
    aggregated = aggregate_rows(rows)

    def stat(metrics: dict[str, Any], name: str) -> str:
        if name not in metrics:
            return "     --     "
        return f"{metrics[name]['mean']:.3f}±{metrics[name]['std']:.3f}"

    for case, metrics in aggregated.items():
        logger.info(
            "%-18s %14s %14s %14s %14s",
            case,
            stat(metrics, "pehe"),
            stat(metrics, "ate_error"),
            stat(metrics, "cf_mae"),
            stat(metrics, "cf_mbe"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
