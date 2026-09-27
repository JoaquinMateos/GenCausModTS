"""Static-dynamic disentanglement benchmarks on the multi-domain SCM.

Models are trained on a subset of domains and evaluated on a held-out domain
whose spurious correlation has the opposite sign, so only representations that
kept the causal factors generalise. Reports OOD MSE/MAE, domain-identification
accuracy and the MIG diagnostic.

Usage::

    uv run python scripts/run_sdd_benchmarks.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from gcmts.core import SimpleTrainer, move_batch, save_training_report
from gcmts.data.synthetic_sdd import MultiDomainGenerator
from gcmts.evaluation.metrics import MIGMetric
from gcmts.static_dynamic_disentanglement.base import BaseStaticDynamicDisentangler
from gcmts.static_dynamic_disentanglement.methods import DANN, ERM, SYNC

logger = logging.getLogger("gcmts.sdd")

TRAIN_DOMAINS = [0, 1, 2, 3]
TEST_DOMAIN = [4]


@dataclass
class CaseConfig:
    name: str
    model_factory: Callable[[], BaseStaticDynamicDisentangler]
    epochs: int
    note: str = ""


@dataclass
class CaseResult:
    case: str
    method: str
    epochs: int
    n_parameters: int
    final_loss: float
    id_mse: float
    ood_mse: float
    ood_mae: float
    domain_accuracy: float | None
    mig: float
    seconds: float
    device: str = "auto"


class _SplitGenerator:
    """Restrict a generator's sampling to a fixed list of domains."""

    def __init__(self, generator: MultiDomainGenerator, domains: list[int]) -> None:
        self.generator = generator
        self.domains = domains

    def sample(self, n_samples: int, **kwargs: Any) -> Any:
        return self.generator.sample(n_samples, domains=self.domains)


def build_cases() -> list[CaseConfig]:
    common = {
        "observed_dim": 8,
        "static_dim": 2,
        "dynamic_dim": 2,
        "spurious_static_dim": 1,
        "spurious_dynamic_dim": 1,
    }
    return [
        CaseConfig(
            name="sync_multidomain",
            model_factory=lambda: SYNC(n_domains=5, **common),
            epochs=600,
            note="four-way static-dynamic disentanglement with causal outcome head",
        ),
        CaseConfig(
            name="dann_multidomain",
            model_factory=lambda: DANN(n_domains=5, **common),
            epochs=600,
            note="domain-adversarial invariant outcome representation (UDA)",
        ),
        CaseConfig(
            name="erm_multidomain",
            model_factory=lambda: ERM(**common),
            epochs=600,
            note="empirical-risk-minimisation baseline (non-causal)",
        ),
    ]


def evaluate(
    model: BaseStaticDynamicDisentangler, generator: MultiDomainGenerator, device: torch.device
) -> dict[str, float | None]:
    def batch_for(domains: list[int]) -> Any:
        return move_batch(generator.sample(1024, domains=domains), device)

    def outcome_mse(model: Any, batch: Any) -> tuple[float, float]:
        pred = model.predict_outcome(batch.x)  # type: ignore[attr-defined]
        diff = pred - batch.outcome
        return diff.pow(2).mean().item(), diff.abs().mean().item()

    id_batch = batch_for(TRAIN_DOMAINS)
    ood_batch = batch_for(TEST_DOMAIN)
    id_mse, _ = outcome_mse(model, id_batch)
    ood_mse, ood_mae = outcome_mse(model, ood_batch)

    domain_accuracy: float | None = None
    if hasattr(model, "spurious_classifier"):
        mix_batch = move_batch(generator.sample(1024), device)
        factors = model.disentangle(mix_batch.x)
        logits = model.spurious_classifier(model._spurious(factors))  # type: ignore[attr-defined]
        domain_accuracy = (logits.argmax(-1) == mix_batch.domain).float().mean().item()

    factors = model.latent_factors(ood_batch.x)  # type: ignore[attr-defined]
    mig = MIGMetric()(factors, ood_batch.z).value
    return {
        "id_mse": id_mse,
        "ood_mse": ood_mse,
        "ood_mae": ood_mae,
        "domain_accuracy": domain_accuracy,
        "mig": mig,
    }


def run_case(cfg: CaseConfig, device: str, batch_size: int, out_dir: Path) -> CaseResult:
    logger.info("=" * 72)
    logger.info("CASE %s | %s", cfg.name, cfg.note)
    torch.manual_seed(0)
    generator = MultiDomainGenerator(n_domains=5, observed_dim=8, horizon=12, seed=0)
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
        seed=0,
        grad_clip=1.0,
        scheduler="cosine",
        val_size=512,
        val_every=20,
    )
    start = time.perf_counter()
    history = trainer.fit(model, _SplitGenerator(generator, TRAIN_DOMAINS))
    seconds = time.perf_counter() - start
    save_training_report(history, exp_dir, title=f"{cfg.name} ({type(model).__name__})")
    model.save(exp_dir / "model.pt")

    resolved = trainer.resolve_device(model)
    scores = evaluate(model, generator, resolved)
    logger.info(
        "RESULT %s | %s (%.1fs)",
        cfg.name,
        " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}=--" for k, v in scores.items()
        ),
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


def oracle_result(out_dir: Path) -> CaseResult:
    """Reference: ridge regression on the true causal factors (upper bound)."""
    import numpy as np
    from sklearn.linear_model import Ridge

    generator = MultiDomainGenerator(n_domains=5, observed_dim=8, horizon=12, seed=0)
    train = generator.sample(4096, domains=TRAIN_DOMAINS)
    test = generator.sample(2048, domains=TEST_DOMAIN)
    causal = slice(0, generator.static_dim + generator.dynamic_dim)
    z_tr = train.z.mean(1).numpy()
    z_te = test.z.mean(1).numpy()
    model = Ridge().fit(z_tr[:, causal], train.outcome.numpy())
    pred = model.predict(z_te[:, causal])
    diff = pred - test.outcome.numpy()
    id_pred = model.predict(z_tr[:, causal]) - train.outcome.numpy()
    return CaseResult(
        case="oracle_causal",
        method="Oracle",
        epochs=0,
        n_parameters=0,
        final_loss=float("nan"),
        id_mse=float((id_pred**2).mean()),
        ood_mse=float((diff**2).mean()),
        ood_mae=float(np.abs(diff).mean()),
        domain_accuracy=None,
        mig=float("nan"),
        seconds=0.0,
        device="cpu",
    )


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [
        logging.FileHandler(out_dir / "sdd_benchmarks.log", mode="w"),
        logging.StreamHandler(),
    ]


def save_results(results: list[CaseResult], out_dir: Path) -> None:
    payload = [asdict(r) for r in results]
    (out_dir / "sdd_benchmarks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "sdd_benchmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0].keys()))
        writer.writeheader()
        writer.writerows(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SDD benchmarks.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    configure_logging(args.out_dir)
    cases = build_cases()
    if args.only:
        cases = [c for c in cases if c.name in set(args.only)]
    results = [run_case(c, args.device, args.batch_size, args.out_dir) for c in cases]
    results.append(oracle_result(args.out_dir))
    save_results(results, args.out_dir)
    logger.info(
        "%-22s %-6s %8s %8s %8s %8s %8s",
        "case", "method", "ID-MSE", "OOD-MSE", "OOD-MAE", "DomAcc", "MIG",
    )
    for r in results:
        acc = "--" if r.domain_accuracy is None else f"{r.domain_accuracy:.3f}"
        logger.info(
            "%-22s %-6s %8.4f %8.4f %8.4f %8s %8.4f",
            r.case, r.method, r.id_mse, r.ood_mse, r.ood_mae, acc, r.mig,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
