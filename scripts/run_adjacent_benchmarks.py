"""Adjacent-branch benchmarks (Paradigm III) and the II.D mechanism method.

Trains the representative method of each adjacent branch on a benchmark that
exposes its ground truth and reports the branch-specific soundness metrics over
a fixed number of seeds:

* III.A GBTPL on graph-constrained propagation paths (path-F1, source accuracy,
  structural-violation rate);
* III.B LacaDM on a multi-objective MDP (hypervolume, sparsity);
* III.C CausalConceptTS on concept-segmented series (faithfulness);
* II.D CausalGAN on a binary-label SCM (counterfactual MAE).

Per-seed rows are written to ``outputs/adjacent_benchmarks.json`` so that
``scripts/make_paper_tables.py`` can aggregate them into a results table.

Usage::

    uv run python scripts/run_adjacent_benchmarks.py --device cuda --seeds 10
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch

from gcmts.core import SimpleTrainer, move_batch
from gcmts.data.binary_scm import BinarySCMGenerator
from gcmts.data.concept_ts import ConceptTimeSeriesGenerator
from gcmts.data.multi_objective import MultiObjectiveMDPGenerator
from gcmts.data.propagation import PropagationGenerator
from gcmts.decision_making.methods import LacaDM
from gcmts.effect_estimation.methods import CausalGAN
from gcmts.evaluation.metrics import (
    CFMAEMetric,
    FaithfulnessMetric,
    HypervolumeMetric,
    PathF1Metric,
    SourceAccuracyMetric,
    SparsityMetric,
    StructuralViolationMetric,
)
from gcmts.evaluation.progress import ProgressTracker
from gcmts.evaluation.report import aggregate_rows, save_rows, wall_seconds
from gcmts.evaluation.tracking import StepTracker
from gcmts.explanation.methods import CausalConceptTS
from gcmts.structure_discovery.methods import GBTPL

logger = logging.getLogger("gcmts.adjacent")


def _train(model: Any, generator: Any, epochs: int, device: str, seed: int) -> float:
    torch.manual_seed(seed)
    trainer = SimpleTrainer(
        max_epochs=epochs,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=20,
        log_every=0,
        seed=seed,
        device=device,
    )
    start = time.perf_counter()
    trainer.fit(model, generator)
    return wall_seconds(start)


def _gbtpl(device: str, epochs: int, seed: int) -> dict[str, float]:
    generator = PropagationGenerator(n_nodes=6, horizon=16, max_path_len=4, seed=seed)
    model = GBTPL(n_nodes=6, horizon=16, max_path_len=3)
    seconds = _train(model, generator, epochs, device, seed)
    batch = move_batch(generator.sample(64), next(model.parameters()).device)
    paths = model.sample_paths(batch)
    truth = {"paths": batch.context["path"], "sources": batch.context["source"]}
    return {
        "seconds": seconds,
        "path_f1": PathF1Metric()(paths, truth).value,
        "source_accuracy": SourceAccuracyMetric()(paths, truth).value,
        "structural_violation": StructuralViolationMetric()(
            paths, {"graph": batch.context["graph"]}
        ).value,
    }


def _lacadm(device: str, epochs: int, seed: int) -> dict[str, float]:
    generator = MultiObjectiveMDPGenerator(state_dim=4, action_dim=2, horizon=12, seed=seed)
    model = LacaDM(state_dim=4, action_dim=2, n_steps=10)
    seconds = _train(model, generator, epochs, device, seed)
    batch = move_batch(generator.sample(32), next(model.parameters()).device)
    front = model.act(batch.x)
    rewards = batch.context["reward"][:, -1, :]
    reference = rewards.max(dim=0).values + 0.1
    return {
        "seconds": seconds,
        "hypervolume": HypervolumeMetric()(front, {"reference": reference}).value,
        "sparsity": SparsityMetric()(front).value,
    }


def _causal_concept_ts(device: str, epochs: int, seed: int) -> dict[str, float]:
    generator = ConceptTimeSeriesGenerator(
        n_channels=4, horizon=16, n_concepts=4, causal_concepts=(0, 2), seed=seed
    )
    model = CausalConceptTS(n_channels=4, horizon=16, n_concepts=4)
    seconds = _train(model, generator, epochs, device, seed)
    batch = move_batch(generator.sample(128), next(model.parameters()).device)
    explanation = model.explain(batch)
    return {
        "seconds": seconds,
        "faithfulness": FaithfulnessMetric()(
            explanation["attributions"], {"causal_concepts": torch.tensor([0, 2])}
        ).value,
    }


def _causalgan(device: str, epochs: int, seed: int) -> dict[str, float]:
    generator = BinarySCMGenerator(n_labels=3, observed_dim=6, horizon=8, seed=seed)
    model = CausalGAN(observed_dim=6, n_labels=3)
    seconds = _train(model, generator, epochs, device, seed)
    batch = move_batch(generator.sample(128), next(model.parameters()).device)
    counter = torch.zeros(batch.x.shape[0], 1, device=batch.x.device)
    cf = model.counterfactual(batch, {"A": counter}).counterfactual
    if batch.counterfactual is None:
        raise ValueError("BinarySCMGenerator must expose the counterfactual.")
    return {"seconds": seconds, "cf_mae": CFMAEMetric()(cf, batch.counterfactual).value}


METHODS = {
    "GBTPL": _gbtpl,
    "LacaDM": _lacadm,
    "CausalConceptTS": _causal_concept_ts,
    "CausalGAN": _causalgan,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    methods = args.methods or list(METHODS)
    rows: list[dict[str, Any]] = []
    progress = ProgressTracker(
        args.out_dir / "adjacent_benchmarks_progress.json",
        "Adjacent branches + II.D",
        len(methods) * args.seeds,
    ).start()
    step = StepTracker.start(
        name="Adjacent branches + II.D",
        params={"methods": list(methods), "seeds": args.seeds, "epochs": args.epochs},
        tags=["adjacent"],
        progress=progress,
    )
    for name in methods:
        for seed in range(args.seeds):
            metrics = METHODS[name](args.device, args.epochs, seed)
            row = {"case": name, "method": name, "seed": seed, **metrics}
            rows.append(row)
            logger.info("%s seed=%d %s", name, seed, {k: round(v, 4) for k, v in metrics.items()})
            progress.advance()
            step.log_run(method=name, iteration=seed, metrics=metrics)
    progress.finish()
    step.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, args.out_dir, stem="adjacent_benchmarks")
    aggregated = aggregate_rows(rows)
    for name in methods:
        if name in aggregated:
            summary = {k: round(v["mean"], 4) for k, v in aggregated[name].items()}
            logger.info("%s: %s", name, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
