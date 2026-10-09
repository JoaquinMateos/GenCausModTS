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
from gcmts.evaluation.stats import (
    Comparison,
    friedman_test,
    metric_matrix,
    nemenyi,
    paired_wilcoxon_holm,
    plot_cd_diagram,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "CrlBenchmarkRunner",
    "DisentanglementBenchmarkRunner",
    "EffectBenchmarkRunner",
    "Comparison",
    "DEFAULT_REGISTRY",
    "MetricRegistry",
    "aggregate_rows",
    "friedman_test",
    "metric_matrix",
    "nemenyi",
    "paired_wilcoxon",
    "paired_wilcoxon_holm",
    "plot_cd_diagram",
    "save_rows",
]
