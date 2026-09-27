"""Benchmark runner for deep, reproducible evaluation of all methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from gcmts.core.base import BaseModel
from gcmts.core.trainer import move_batch
from gcmts.evaluation.metrics import DEFAULT_REGISTRY, MetricRegistry, MetricResult
from gcmts.typing import Batch

__all__ = ["BenchmarkTask", "BenchmarkResult", "BenchmarkRunner"]


@dataclass
class BenchmarkTask:
    """Specification of one benchmark run.

    Attributes:
        name: benchmark/task name
        model: the method to evaluate
        data: a data generator, loader, or pre-loaded dataset
        metrics: list of metric names from a :class:`MetricRegistry`
        causal_level: Pearl level claimed by the task
        output_dir: where to write artefacts
    """

    name: str
    model: BaseModel
    data: Any
    metrics: list[str] = field(default_factory=list)
    causal_level: str = "L1_association"
    output_dir: Path | None = None


@dataclass
class BenchmarkResult:
    task_name: str
    causal_level: str
    metrics: list[MetricResult] = field(default_factory=list)
    artefacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "causal_level": self.causal_level,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "level": m.level,
                    "higher_is_better": m.higher_is_better,
                }
                for m in self.metrics
            ],
        }


class BenchmarkRunner:
    """Run a suite of benchmark tasks and collect results.

    The runner is intentionally thin: it does not prescribe how models are
    trained. It expects ``task.data`` to expose a ``sample`` or ``__iter__``
    contract that the caller controls, and a ``target`` that can be passed to
    each metric.
    """

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.results: list[BenchmarkResult] = []

    def run(
        self,
        task: BenchmarkTask,
        *,
        prediction: Any | None = None,
        target: Any | None = None,
    ) -> BenchmarkResult:
        """Evaluate ``task.model`` on ``task.data``.

        If ``prediction`` and ``target`` are supplied they are used directly;
        otherwise the caller must have produced them beforehand (e.g. after
        training).
        """
        if prediction is None or target is None:
            prediction, target = self._inference(task)

        metric_names = task.metrics or self.registry.list_metrics(task.causal_level)
        results = self.registry.evaluate(metric_names, prediction, target)
        result = BenchmarkResult(
            task_name=task.name,
            causal_level=task.causal_level,
            metrics=results,
        )
        self.results.append(result)
        return result

    def run_all(self, tasks: list[BenchmarkTask]) -> list[BenchmarkResult]:
        return [self.run(t) for t in tasks]

    def summary(self) -> dict[str, list[dict[str, Any]]]:
        return {"results": [r.to_dict() for r in self.results]}

    def _inference(self, task: BenchmarkTask) -> tuple[Any, Any]:
        # Placeholder: concrete benchmarks will override this by providing
        # pre-computed predictions/targets or by subclassing the runner.
        raise NotImplementedError(
            "Either provide prediction/target tensors or subclass BenchmarkRunner."
        )


def _model_device(model: BaseModel) -> torch.device:
    """Return the device the model's parameters live on (default CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_batch(data: Any, model: BaseModel) -> Batch:
    """Normalise ``task.data`` to a batch on the model's device."""
    if isinstance(data, Batch):
        batch = data
    elif hasattr(data, "sample"):
        batch = data.sample(100, 32)
    else:
        batch = next(iter(data))
    return move_batch(batch, _model_device(model))


class CrlBenchmarkRunner(BenchmarkRunner):
    """Runner specialised for causal representation-learning tasks."""

    def _inference(self, task: BenchmarkTask) -> tuple[Any, Any]:
        from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner

        if not isinstance(task.model, BaseCausalRepresentationLearner):
            raise TypeError("CRL benchmark requires a BaseCausalRepresentationLearner.")
        batch = _get_batch(task.data, task.model)
        z_pred = task.model.encode(batch.x, batch.context)
        z_true = batch.z
        if z_true is None:
            raise ValueError("CRL benchmark requires ground-truth latents in batch.z.")
        return z_pred, z_true


class EffectBenchmarkRunner(BenchmarkRunner):
    """Runner specialised for counterfactual / treatment-effect tasks."""

    def _inference(self, task: BenchmarkTask) -> tuple[Any, Any]:
        from gcmts.effect_estimation.base import BaseEffectEstimator

        if not isinstance(task.model, BaseEffectEstimator):
            raise TypeError("Effect benchmark requires a BaseEffectEstimator.")
        batch = _get_batch(task.data, task.model)
        cf_action = {"action": torch.zeros_like(batch.x[:, :, : task.model.treatment_dim])}
        out = task.model.counterfactual(batch, cf_action)
        return out.counterfactual, batch.x


class DisentanglementBenchmarkRunner(BenchmarkRunner):
    """Runner specialised for static-dynamic disentanglement tasks."""

    def _inference(self, task: BenchmarkTask) -> tuple[Any, Any]:
        from gcmts.static_dynamic_disentanglement.base import BaseStaticDynamicDisentangler

        if not isinstance(task.model, BaseStaticDynamicDisentangler):
            raise TypeError("Disentanglement benchmark requires a BaseStaticDynamicDisentangler.")
        batch = _get_batch(task.data, task.model)
        x_pred = task.model.predict(batch.x, batch.context.get("domain") if batch.context else None)
        return x_pred, batch.x
