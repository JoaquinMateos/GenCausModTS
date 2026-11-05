"""Benchmark the six newly integrated branch representatives.

The runner trains one representative per adjacent branch (plus the two
new II.C/II.D representatives) with :class:`gcmts.core.SimpleTrainer` and
reports the branch-specific soundness metrics on a fresh evaluation batch:

* ``CaDRe``  (I.E)   latent MCC, latent-adjacency SHD, observed-graph SHD;
* ``VGAN-CHG`` (II.C) PEHE and ATE error on a static panel (IHDP if available,
  otherwise an exactly simulated panel built here);
* ``CNF``    (II.D)  CFMAE on the paired texture-swap counterfactual;
* ``SCARF``  (III.A) decision-rate improvement, recourse cost, budget
  violation, fairness gap, and counterfactual validity against the exact
  replay of the generator;
* ``RCA-VAE`` (III.B) root-cause AUROC, path-F1 and source accuracy;
* ``Becker`` (III.C) faithfulness and mean Pearson agreement with the
  generator's exact mean-ablation attributions.

Usage::

    .venv/bin/python scripts/run_new_representatives.py --device cpu --seeds 1 --epochs 1
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from gcmts.causal_representation_learning.methods import CaDRe
from gcmts.core import SimpleTrainer, move_batch
from gcmts.core.utils import mcc
from gcmts.data.compositional_images import CompositionalImageGenerator
from gcmts.data.concept_factor_ts import ConceptFactorTimeSeriesGenerator
from gcmts.data.latent_scm import LatentStructuralGenerator
from gcmts.data.regime_shift import RegimeShiftPropagationGenerator
from gcmts.data.semi_synthetic import IHDPGenerator, IHDPLoader, IHDPReplication
from gcmts.data.sequential_recourse import SequentialRecourseGenerator
from gcmts.decision_making.methods import SCARF
from gcmts.effect_estimation.methods import CNF, VGANCHG
from gcmts.evaluation.benchmarks import EMA_DECAY
from gcmts.evaluation.metrics import (
    ATEMetric,
    CFMAEMetric,
    FaithfulnessMetric,
    PathF1Metric,
    PEHEMetric,
    SHDMetric,
    SourceAccuracyMetric,
)
from gcmts.evaluation.report import save_rows, wall_seconds
from gcmts.explanation.methods import BeckerAttribution
from gcmts.structure_discovery.methods import RCAVAE
from gcmts.typing import Batch

logger = logging.getLogger("gcmts.new_representatives")

ROOT = Path(__file__).resolve().parents[1]
METHOD_KEYS = ("CaDRe", "VGAN-CHG", "CNF", "SCARF", "RCA-VAE", "Becker")

_ALIASES = {
    "cadre": "CaDRe",
    "vgan": "VGAN-CHG",
    "vgan-chg": "VGAN-CHG",
    "vganchg": "VGAN-CHG",
    "cnf": "CNF",
    "scarf": "SCARF",
    "rca": "RCA-VAE",
    "rca-vae": "RCA-VAE",
    "rcavae": "RCA-VAE",
    "becker": "Becker",
    "becker-attribution": "Becker",
}


def _train(
    model: Any, generator: Any, epochs: int, device: str, seed: int
) -> tuple[float, SimpleTrainer]:
    """Train with the shared benchmark protocol (EMA, 20 steps per epoch)."""
    torch.manual_seed(seed)
    trainer = SimpleTrainer(
        max_epochs=epochs,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=20,
        ema_decay=EMA_DECAY,
        log_every=0,
        seed=seed,
        device=device,
    )
    start = time.perf_counter()
    trainer.fit(model, generator)
    return wall_seconds(start), trainer


def _cpu_batch(batch: Batch) -> Batch:
    """Copy the tensors needed for a CPU-side exact replay."""
    context = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in (batch.context or {}).items()
    }
    return Batch(x=batch.x.detach().cpu(), context=context)


# ---------------------------------------------------------------------------
# CaDRe (I.E)
# ---------------------------------------------------------------------------


def _cadre(device: str, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = LatentStructuralGenerator(
        latent_dim=3,
        observed_dim=6,
        horizon=16,
        n_regimes=4,
        mixing="tanh",
        seed=seed,
    )
    model = CaDRe(
        observed_dim=6,
        latent_dim=3,
        u_dim=4,
        hidden_dims=(64, 64),
        s_hidden=(32,),
        decoder_hidden=(64, 64),
        prior_hidden=(32,),
    )
    seconds, trainer = _train(model, generator, epochs, device, seed)
    resolved = trainer.resolve_device(model)
    batch = move_batch(generator.sample(64), resolved)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    if batch.z is None or batch.adjacency is None or batch.context is None:
        raise ValueError("LatentStructuralGenerator must expose z, adjacency and context.")
    latent_adjacency = model.get_latent_adjacency()
    observed_adjacency = model.get_observed_adjacency()
    if latent_adjacency is None:
        raise ValueError("CaDRe did not expose a latent adjacency.")
    return {
        "seconds": seconds,
        "mcc": mcc(batch.z, outputs.z),
        "latent_shd": SHDMetric(aligned=False)(latent_adjacency, batch.adjacency).value,
        "observed_shd": SHDMetric(aligned=False)(
            observed_adjacency, batch.context["observed_adjacency"]
        ).value,
    }


# ---------------------------------------------------------------------------
# VGAN-CHG (II.C) on a static panel
# ---------------------------------------------------------------------------


def _synthetic_static_panel(
    seed: int, n_train: int = 512, n_test: int = 256, p: int = 5
) -> tuple[IHDPReplication, IHDPReplication]:
    """Exact two-potential-outcome panel used when IHDP is not on disk."""
    rng = np.random.default_rng(seed)

    def draw(n: int) -> IHDPReplication:
        x = rng.normal(size=(n, p)).astype(np.float32)
        logit = 0.7 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * np.tanh(x[:, 2]) - 0.1
        t = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
        beta = np.array([0.5, -0.4, 0.3, 0.2, -0.1], dtype=np.float32)
        mu0 = (x @ beta + 0.5 * np.sin(x[:, 0]) + 0.3 * x[:, 1] * x[:, 2] + 0.5).astype(np.float32)
        tau = (1.0 + 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2] ** 2).astype(np.float32)
        mu1 = (mu0 + tau).astype(np.float32)
        noise = 0.1 * rng.normal(size=n).astype(np.float32)
        yf = np.where(t > 0, mu1, mu0).astype(np.float32) + noise
        ycf = np.where(t > 0, mu0, mu1).astype(np.float32)
        return IHDPReplication(x=x, t=t, yf=yf, ycf=ycf, mu0=mu0, mu1=mu1)

    return draw(n_train), draw(n_test)


def _ihdp_pair(seed: int) -> tuple[IHDPGenerator, IHDPGenerator, IHDPReplication, str]:
    """Return (train generator, test generator, test replication, dataset name)."""
    for directory in (
        ROOT / "data" / "static" / "ihdp",
        ROOT / "data" / "semi_synthetic",
    ):
        train_file = directory / "ihdp_npci_1-100.train.npz"
        test_file = directory / "ihdp_npci_1-100.test.npz"
        if train_file.exists() and test_file.exists():
            loader = IHDPLoader(train_file, test_file)
            index = seed % loader.n_replications
            train_rep = loader.replication(index, split="train")
            test_rep = loader.replication(index, split="test")
            train_gen = IHDPGenerator(train_rep, seed=seed)
            test_gen = IHDPGenerator(test_rep, seed=seed + 10_000, reference=train_gen)
            return train_gen, test_gen, test_rep, "ihdp"

    train_rep, test_rep = _synthetic_static_panel(seed)
    train_gen = IHDPGenerator(train_rep, seed=seed)
    test_gen = IHDPGenerator(test_rep, seed=seed + 10_000, reference=train_gen)
    return train_gen, test_gen, test_rep, "synthetic_static"


def _vgan_chg(device: str, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_gen, test_gen, test_rep, dataset = _ihdp_pair(seed)
    model = VGANCHG(
        observed_dim=train_gen.observed_dim,
        treatment_dim=1,
        window=3,
        latent_global=8,
        latent_regional=4,
        hidden_dims=(32, 32),
        hidden_dim=16,
        max_len=1,
    )
    seconds, trainer = _train(model, train_gen, epochs, device, seed)
    resolved = trainer.resolve_device(model)
    batch = move_batch(test_gen.sample_all(), resolved)
    action = batch.context["A"] if batch.context is not None else None
    if action is None:
        raise ValueError("The static panel must expose the treatment in context['A'].")
    model.eval()
    with torch.no_grad():
        y1 = model.counterfactual(batch, {"A": torch.ones_like(action)}).counterfactual
        y0 = model.counterfactual(batch, {"A": torch.zeros_like(action)}).counterfactual
    y1 = test_gen.rescale_outcome(y1[..., -1].detach().cpu())
    y0 = test_gen.rescale_outcome(y0[..., -1].detach().cpu())
    tau_hat = (y1 - y0).reshape(-1)
    tau_true = torch.from_numpy(test_rep.mu1 - test_rep.mu0).float()
    return {
        "seconds": seconds,
        "dataset": dataset,
        "pehe": PEHEMetric()(tau_hat, tau_true).value,
        "ate_error": ATEMetric()(tau_hat, tau_true).value,
    }


# ---------------------------------------------------------------------------
# CNF (II.D)
# ---------------------------------------------------------------------------


def _cnf(device: str, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = CompositionalImageGenerator(image_size=16, horizon=4, n_classes=6, seed=seed)
    model = CNF(
        image_size=16,
        n_classes=6,
        noise_dim=4,
        hidden_dims=(32, 32),
        inversion_steps=20,
    )
    seconds, trainer = _train(model, generator, epochs, device, seed)
    resolved = trainer.resolve_device(model)
    batch = move_batch(generator.sample(16), resolved)
    if batch.context is None or batch.counterfactual is None:
        raise ValueError("CompositionalImageGenerator must expose the paired counterfactual.")
    model.eval()
    with torch.no_grad():
        result = model.counterfactual(batch, {"texture": batch.context["cf_texture_label"]})
    return {
        "seconds": seconds,
        "cf_mae": CFMAEMetric()(result.counterfactual, batch.counterfactual).value,
    }


# ---------------------------------------------------------------------------
# SCARF (III.A)
# ---------------------------------------------------------------------------


def _fairness_gap(decision: Tensor, sensitive: Tensor) -> float:
    """Absolute demographic-parity gap of the favourable-decision rate."""
    group_one = decision[sensitive > 0.5]
    group_zero = decision[sensitive <= 0.5]
    if group_one.numel() == 0 or group_zero.numel() == 0:
        return float("nan")
    return float((group_one.mean() - group_zero.mean()).abs().item())


def _scarf(device: str, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = SequentialRecourseGenerator(state_dim=6, horizon=8, action_dim=3, seed=seed)
    model = SCARF(state_dim=6, action_dim=3, hidden_dim=64, horizon=8)
    seconds, trainer = _train(model, generator, epochs, device, seed)
    resolved = trainer.resolve_device(model)
    batch = move_batch(generator.sample(128), resolved)
    model.eval()
    with torch.no_grad():
        recourse = model.recourse(batch)
    model_cf = recourse["counterfactual"].detach().cpu()
    intervention = recourse["intervention"].detach().cpu()
    cost = recourse["cost"].detach().cpu()
    post_decision = recourse["decision"].detach().cpu()

    cpu_batch = _cpu_batch(batch)
    if cpu_batch.context is None:
        raise ValueError("SequentialRecourseGenerator must expose its replay context.")
    exact = generator.counterfactual(cpu_batch, intervention, project_budget=False)
    factual_decision = generator.decision(cpu_batch.x)
    sensitive = cpu_batch.context["sensitive"]
    budget = cpu_batch.context["budget"].reshape(-1)[0]
    violation = (cost > budget + 1e-6).float().mean().item()
    return {
        "seconds": seconds,
        "improvement": float((post_decision.mean() - factual_decision.mean()).item()),
        "recourse_cost": float(cost.mean().item()),
        "budget_violation": float(violation),
        "fairness_gap": _fairness_gap(post_decision, sensitive),
        "cf_mae": CFMAEMetric()(model_cf, exact.x).value,
    }


# ---------------------------------------------------------------------------
# RCA-VAE (III.B)
# ---------------------------------------------------------------------------


def _rca_vae(device: str, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = RegimeShiftPropagationGenerator(n_nodes=8, horizon=32, seed=seed)
    model = RCAVAE(n_nodes=8, window=8, hidden_dim=64, kl_weight=0.01)
    seconds, trainer = _train(model, generator, epochs, device, seed)
    resolved = trainer.resolve_device(model)
    batch = move_batch(generator.sample(32), resolved)
    if batch.context is None:
        raise ValueError("RegimeShiftPropagationGenerator must expose its context.")
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        root = model.root_causes(
            outputs["edge_probs_pre"], outputs["edge_probs_post"], parents=batch.x
        )
        paths = model.sample_paths(batch)
    truth = batch.context["root_causes"].reshape(-1).detach().cpu().numpy()
    # Primary score: the paper-literal edge-posterior KL, which is the sharper
    # signal in this compact model. The theory-faithful conditional-mean score
    # is reported alongside it.
    edge_kl = root["kl"].reshape(-1).detach().cpu().numpy()
    conditional = root["scores"].reshape(-1).detach().cpu().numpy()
    two_class = np.unique(truth).size == 2
    auroc = float(roc_auc_score(truth, edge_kl)) if two_class else float("nan")
    auroc_cond = (
        float(roc_auc_score(truth, conditional)) if two_class else float("nan")
    )
    return {
        "seconds": seconds,
        "root_cause_auroc": auroc,
        "root_cause_auroc_cond": auroc_cond,
        "path_f1": PathF1Metric()(paths, {"paths": batch.context["path"]}).value,
        "source_accuracy": SourceAccuracyMetric()(
            paths, {"sources": batch.context["source"]}
        ).value,
    }


# ---------------------------------------------------------------------------
# Becker attribution (III.C)
# ---------------------------------------------------------------------------


def _mean_pearson(prediction: Tensor, target: Tensor) -> float:
    """Mean per-concept Pearson correlation, skipping constant truth columns."""
    prediction = prediction.detach().cpu().double().reshape(prediction.shape[0], -1)
    target = target.detach().cpu().double().reshape(target.shape[0], -1)
    values: list[float] = []
    for index in range(prediction.shape[1]):
        a = prediction[:, index]
        b = target[:, index]
        if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
            continue
        corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
        if torch.isfinite(corr):
            values.append(float(corr))
    if not values:
        a = prediction.reshape(-1)
        b = target.reshape(-1)
        if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
            return float("nan")
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    return float(np.mean(values))


def _becker(device: str, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = ConceptFactorTimeSeriesGenerator(seed=seed)
    model = BeckerAttribution(n_channels=6, horizon=12, n_concepts=4)
    seconds, trainer = _train(model, generator, epochs, device, seed)
    resolved = trainer.resolve_device(model)
    batch = move_batch(generator.sample(128), resolved)
    if batch.context is None or "true_ablation" not in batch.context:
        raise ValueError("ConceptFactorTimeSeriesGenerator must expose true_ablation.")
    model.eval()
    explanation = model.explain(batch)
    attributions = explanation["attributions"]
    return {
        "seconds": seconds,
        "faithfulness": FaithfulnessMetric()(
            attributions, {"causal_concepts": batch.context["causal_concepts"]}
        ).value,
        "attribution_pearson": _mean_pearson(attributions, batch.context["true_ablation"]),
    }


METHODS: dict[str, Callable[[str, int, int], dict[str, Any]]] = {
    "CaDRe": _cadre,
    "VGAN-CHG": _vgan_chg,
    "CNF": _cnf,
    "SCARF": _scarf,
    "RCA-VAE": _rca_vae,
    "Becker": _becker,
}

DATASETS = {
    "CaDRe": "latent_structural_scm",
    "VGAN-CHG": "static_panel",
    "CNF": "compositional_images",
    "SCARF": "sequential_recourse",
    "RCA-VAE": "regime_shift_propagation",
    "Becker": "concept_factor_ts",
}


def _normalise_methods(values: list[str] | None) -> list[str]:
    if not values:
        return list(METHOD_KEYS)
    resolved: list[str] = []
    for value in values:
        key = value.strip().lower().replace("_", "-")
        if key not in _ALIASES:
            raise SystemExit(f"Unknown method {value!r}. Choose from: {', '.join(METHOD_KEYS)}.")
        if _ALIASES[key] not in resolved:
            resolved.append(_ALIASES[key])
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/new_representatives.json"),
        help="raw JSON path; the aggregate CSV is written alongside it",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    methods = _normalise_methods(args.methods)
    rows: list[dict[str, Any]] = []
    for name in methods:
        for seed in range(args.seeds):
            logger.info(
                "Running %s (seed=%d, epochs=%d, device=%s)",
                name,
                seed,
                args.epochs,
                args.device,
            )
            metrics = METHODS[name](args.device, args.epochs, seed)
            dataset = metrics.pop("dataset", DATASETS[name])
            row = {
                "case": name,
                "dataset": dataset,
                "method": name,
                "seed": seed,
                "seconds": metrics.pop("seconds"),
                **metrics,
            }
            rows.append(row)
            logger.info(
                "%s seed=%d %s",
                name,
                seed,
                {
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in metrics.items()
                },
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_rows(rows, args.out.parent, stem=args.out.stem)
    for row in rows:
        logger.info("ROW %s", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
