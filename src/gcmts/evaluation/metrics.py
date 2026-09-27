"""Evaluation metrics organised by model family and causal level."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from gcmts.core.utils import hungarian_match, mcc

__all__ = [
    "Metric",
    "MetricResult",
    "MetricRegistry",
    "MCCMetric",
    "R2DiagMetric",
    "SHDMetric",
    "PEHEMetric",
    "ATEMetric",
    "CFMAEMetric",
    "OODMSEMetric",
]


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    level: str  # L1/L2/L3 or family tag
    higher_is_better: bool


class Metric(ABC):
    """A single evaluation metric with a ground-truth contract."""

    def __init__(self, name: str, level: str, *, higher_is_better: bool) -> None:
        self.name = name
        self.level = level
        self.higher_is_better = higher_is_better

    @abstractmethod
    def __call__(self, prediction: Any, target: Any) -> MetricResult:
        """Compute the metric from a prediction and ground-truth target."""

    def direction(self) -> int:
        return 1 if self.higher_is_better else -1


class MetricRegistry:
    """Central registry of metrics, indexed by family and causal level."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Metric '{metric.name}' already registered.")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        return self._metrics[name]

    def list_metrics(self, level: str | None = None) -> list[str]:
        if level is None:
            return list(self._metrics.keys())
        return [n for n, m in self._metrics.items() if m.level == level]

    def evaluate(
        self,
        names: list[str],
        prediction: Any,
        target: Any,
    ) -> list[MetricResult]:
        return [self._metrics[n](prediction, target) for n in names]


def _default_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register(MCCMetric())
    reg.register(R2DiagMetric())
    reg.register(SHDMetric())
    reg.register(PEHEMetric())
    reg.register(ATEMetric())
    reg.register(CFMAEMetric())
    reg.register(OODMSEMetric())
    return reg


# --- Identifiable CRL metrics ------------------------------------------------

class MCCMetric(Metric):
    """Mean correlation coefficient after Hungarian matching."""

    def __init__(self) -> None:
        super().__init__("mcc", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        return MetricResult(self.name, mcc(target, prediction), self.level, self.higher_is_better)


class R2DiagMetric(Metric):
    """Diagonal R² after Hungarian matching of estimated to true factors.

    Following the review, ``R²_ij`` is the coefficient of determination of the
    best *linear* map from estimated factor ``j`` to true factor ``i`` (i.e. the
    squared Pearson correlation). Matching uses the absolute correlation matrix,
    so the metric is invariant to permutation and scaling.
    """

    def __init__(self) -> None:
        super().__init__("r2_diag", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        pred = prediction.reshape(-1, prediction.shape[-1])
        tgt = target.reshape(-1, target.shape[-1])
        pred_std = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
        tgt_std = (tgt - tgt.mean(0)) / (tgt.std(0) + 1e-8)
        corr = (tgt_std.T @ pred_std) / tgt.shape[0]
        _, col = hungarian_match(-corr.abs())
        matched = corr[torch.arange(corr.shape[0]), col]
        return MetricResult(
            self.name,
            (matched**2).mean().item(),
            self.level,
            self.higher_is_better,
        )


class SHDMetric(Metric):
    """Structural Hamming Distance between estimated and true adjacency."""

    def __init__(self) -> None:
        super().__init__("shd", "L1_structure", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        pred_bin = (prediction != 0).int()
        tgt_bin = (target != 0).int()
        distance = (pred_bin != tgt_bin).sum().item()
        return MetricResult(self.name, float(distance), self.level, self.higher_is_better)


# --- Effect-estimation metrics -----------------------------------------------

class PEHEMetric(Metric):
    """Precision in Estimation of Heterogeneous Effect."""

    def __init__(self) -> None:
        super().__init__("pehe", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        diff = (prediction - target).reshape(-1)
        pehe = torch.sqrt((diff**2).mean()).item()
        return MetricResult(self.name, pehe, self.level, self.higher_is_better)


class ATEMetric(Metric):
    """Absolute error on the average treatment effect."""

    def __init__(self) -> None:
        super().__init__("ate_error", "L2_intervention", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        ate_pred = prediction.mean().item()
        ate_true = target.mean().item()
        return MetricResult(
            self.name, abs(ate_pred - ate_true), self.level, self.higher_is_better
        )


class CFMAEMetric(Metric):
    """Counterfactual mean absolute error."""

    def __init__(self) -> None:
        super().__init__("cf_mae", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        mae = (prediction - target).abs().mean().item()
        return MetricResult(self.name, mae, self.level, self.higher_is_better)


# --- Static-dynamic metrics --------------------------------------------------

class OODMSEMetric(Metric):
    """Out-of-distribution mean squared error."""

    def __init__(self) -> None:
        super().__init__("ood_mse", "L1_generalisation", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        mse = ((prediction - target) ** 2).mean().item()
        return MetricResult(self.name, mse, self.level, self.higher_is_better)


DEFAULT_REGISTRY = _default_registry()
