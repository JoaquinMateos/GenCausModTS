"""Evaluation module."""

from gcmts.evaluation.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTask,
    CrlBenchmarkRunner,
    DisentanglementBenchmarkRunner,
    EffectBenchmarkRunner,
)
from gcmts.evaluation.metrics import DEFAULT_REGISTRY, MetricRegistry
from gcmts.evaluation.report import aggregate_rows, paired_wilcoxon, save_rows

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "CrlBenchmarkRunner",
    "DisentanglementBenchmarkRunner",
    "EffectBenchmarkRunner",
    "DEFAULT_REGISTRY",
    "MetricRegistry",
    "aggregate_rows",
    "paired_wilcoxon",
    "save_rows",
]
