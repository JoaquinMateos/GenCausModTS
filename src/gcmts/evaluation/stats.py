"""Statistical comparison utilities for the benchmark tables.

This module implements the protocol fixed in the repository plan
(Section 12.2 of ``.AGENTS.md``):

* :func:`paired_wilcoxon_holm` -- paired Wilcoxon signed-rank tests between
  every pair of methods, Holm-corrected over the family of tests, with pairs
  formed by ``(problem, seed)`` so that every comparison uses the same data
  instances.
* :func:`friedman_test` -- omnibus Friedman test over a rank matrix of shape
  ``(n_problems, n_methods)``.
* :func:`nemenyi` -- Nemenyi post-hoc test and critical-difference diagram.
* :func:`save_cd_figure` -- write a StaTDS figure as vector PDF and SVG.

Friedman, Nemenyi and the diagrams are delegated to StaTDS (Luna et al.,
Neurocomputing 595:127877, 2024), so the reported test and its figure come from
one implementation.

The statistics are computed on per-seed rows (the raw JSON records written by
:func:`gcmts.evaluation.report.save_rows`), so they are consistent with the
reported mean/std.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BootstrapCI",
    "Comparison",
    "bootstrap_ci",
    "bootstrap_case_cis",
    "load_rows",
    "paired_wilcoxon_holm",
    "metric_matrix",
    "friedman_test",
    "nemenyi",
    "save_cd_figure",
    "plot_cd_diagram",
]

@dataclass(frozen=True)
class Comparison:
    """One Holm-corrected paired Wilcoxon comparison between two methods."""

    case_a: str
    case_b: str
    metric: str
    n_pairs: int
    statistic: float
    p_value: float
    p_holm: float
    significant: bool


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap confidence interval for a mean."""

    mean: float
    low: float
    high: float
    n: int
    alpha: float


def bootstrap_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    rng_seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``values`` (non-finite dropped).

    Resampling is deterministic for a given ``rng_seed`` so reported intervals
    are reproducible. Fewer than two finite values yields a degenerate interval
    at the point estimate.
    """
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), 0, alpha)
    mean = float(array.mean())
    if array.size < 2:
        return BootstrapCI(mean, mean, mean, int(array.size), alpha)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, array.size, size=(max(int(n_boot), 1), array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return BootstrapCI(mean, float(low), float(high), int(array.size), alpha)


def bootstrap_case_cis(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    case_key: str = "case",
    alpha: float = 0.05,
    n_boot: int = 10_000,
    rng_seed: int = 0,
) -> dict[str, BootstrapCI]:
    """Per-case bootstrap CIs for ``metric`` over the rows written by ``save_rows``."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get(metric) is None or row.get(case_key) is None:
            continue
        grouped.setdefault(str(row[case_key]), []).append(float(row[metric]))
    return {
        case: bootstrap_ci(values, alpha=alpha, n_boot=n_boot, rng_seed=rng_seed + index)
        for index, (case, values) in enumerate(sorted(grouped.items()))
    }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load raw per-seed benchmark rows from the JSON written by ``save_rows``."""
    import json

    return list(json.loads(Path(path).read_text(encoding="utf-8")))


def _keyed_values(
    rows: Sequence[Mapping[str, Any]],
    case: str,
    metric: str,
    *,
    case_key: str,
    problem_key: str,
    seed_key: str,
) -> dict[tuple[str, int], float]:
    """Map ``(problem, seed) -> metric`` for one case, averaging duplicates."""
    collected: dict[tuple[str, int], list[float]] = {}
    counter: dict[str, int] = {}
    for row in rows:
        if row.get(case_key) != case or row.get(metric) is None:
            continue
        problem = str(row.get(problem_key, "all"))
        raw_seed = row.get(seed_key)
        if raw_seed is None:
            raw_seed = counter.get(problem, 0)
            counter[problem] = int(raw_seed) + 1
        collected.setdefault((problem, int(raw_seed)), []).append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in collected.items()}


def paired_wilcoxon_holm(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    alpha: float = 0.05,
    case_key: str = "case",
    problem_key: str = "dataset",
    seed_key: str = "seed",
    min_pairs: int = 3,
) -> list[Comparison]:
    """Holm-corrected paired Wilcoxon signed-rank tests for every method pair.

    Rows are paired by ``(problem, seed)``; a pair of methods enters the
    correction only when at least ``min_pairs`` common instances exist. The
    returned list is sorted by the corrected p-value; ``significant`` uses the
    family-wise level ``alpha``.
    """
    from scipy.stats import wilcoxon

    cases = sorted({str(row[case_key]) for row in rows if row.get(case_key) is not None})
    values = {
        case: _keyed_values(
            rows,
            case,
            metric,
            case_key=case_key,
            problem_key=problem_key,
            seed_key=seed_key,
        )
        for case in cases
    }

    raw: list[tuple[str, str, int, float, float]] = []
    for i, case_a in enumerate(cases):
        for case_b in cases[i + 1 :]:
            common = sorted(set(values[case_a]) & set(values[case_b]))
            if len(common) < max(min_pairs, 2):
                continue
            a = np.array([values[case_a][key] for key in common])
            b = np.array([values[case_b][key] for key in common])
            differences = a - b
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(a, b)
                statistic, p_value = float(result.statistic), float(result.pvalue)
            raw.append((case_a, case_b, len(common), statistic, p_value))

    if not raw:
        return []

    p_values = np.array([entry[4] for entry in raw])
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        value = (total - rank) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)

    comparisons = [
        Comparison(
            case_a=case_a,
            case_b=case_b,
            metric=metric,
            n_pairs=n_pairs,
            statistic=statistic,
            p_value=p_value,
            p_holm=float(adjusted[index]),
            significant=bool(adjusted[index] < alpha),
        )
        for index, (case_a, case_b, n_pairs, statistic, p_value) in enumerate(raw)
    ]
    return sorted(comparisons, key=lambda c: (c.p_holm, c.p_value))


def metric_matrix(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    case_key: str = "case",
    problem_key: str = "dataset",
    seed_key: str = "seed",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Aggregate rows into a ``(n_problems, n_cases)`` matrix of seed means.

    Missing ``(problem, case)`` entries are ``NaN`` so that callers can decide
    whether to drop incomplete blocks (the plan's "complete blocks only" rule).
    Returns ``(matrix, cases, problems)``.
    """
    cases = sorted({str(row[case_key]) for row in rows if row.get(case_key) is not None})
    problems = sorted({str(row.get(problem_key, "all")) for row in rows})
    matrix = np.full((len(problems), len(cases)), np.nan)
    problem_index = {name: i for i, name in enumerate(problems)}
    case_index = {name: j for j, name in enumerate(cases)}
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get(metric) is None or row.get(case_key) is None:
            continue
        problem = str(row.get(problem_key, "all"))
        grouped.setdefault((problem, str(row[case_key])), []).append(float(row[metric]))
    for (problem, case), values in grouped.items():
        matrix[problem_index[problem], case_index[case]] = float(np.mean(values))
    return matrix, cases, problems


def _complete_matrix(rank_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rank_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("rank_matrix must be two-dimensional.")
    return matrix[~np.isnan(matrix).any(axis=1)]


def friedman_test(
    rank_matrix: np.ndarray, *, higher_is_better: bool = True
) -> dict[str, float]:
    """Omnibus Friedman test on a ``(n_problems, n_methods)`` metric matrix.

    The statistic and p-value come from ``statds.no_parametrics.friedman``.
    ``higher_is_better`` only fixes the orientation of the internal ranks; the
    statistic itself is invariant to reversing them.
    """
    import pandas as pd
    from statds.no_parametrics import friedman as statds_friedman

    matrix = _complete_matrix(rank_matrix)
    if matrix.shape[0] < 2:
        raise ValueError("Friedman test requires at least two complete problems.")
    if matrix.shape[1] < 3:
        raise ValueError("Friedman test requires at least three methods.")
    frame = pd.DataFrame(
        matrix, columns=[f"method_{index}" for index in range(matrix.shape[1])]
    )
    frame.insert(0, "problem", np.arange(matrix.shape[0]))
    _, statistic, p_value, _, _ = statds_friedman(
        frame, alpha=0.05, minimize=not higher_is_better
    )
    if p_value is None:  # StaTDS uses a table for very small designs
        from scipy.stats import chi2

        p_value = chi2.sf(statistic, matrix.shape[1] - 1)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_problems": float(matrix.shape[0]),
        "n_methods": float(matrix.shape[1]),
    }


def nemenyi(
    rank_matrix: np.ndarray,
    *,
    alpha: float = 0.05,
    higher_is_better: bool = True,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Nemenyi post-hoc test, critical difference and StaTDS diagram.

    Ranks are oriented so that the best method receives rank one. The critical
    difference and the figure are produced by ``statds.no_parametrics.nemenyi``;
    the returned ``figure`` can be saved with :func:`save_cd_figure`.
    """
    from scipy.stats import rankdata
    from statds.no_parametrics import nemenyi as statds_nemenyi

    matrix = _complete_matrix(rank_matrix)
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError("Nemenyi requires >= 2 complete problems and >= 3 methods.")
    n_problems, n_methods = matrix.shape
    orientation = -matrix if higher_is_better else matrix
    ranks = np.apply_along_axis(rankdata, 1, orientation)
    average_ranks = ranks.mean(axis=0)
    if names is None:
        labels = [f"method_{index}" for index in range(n_methods)]
    else:
        labels = list(names)
        if len(labels) != n_methods:
            raise ValueError("names must have one entry per method.")
    rank_map = {
        str(label): float(rank)
        for label, rank in zip(labels, average_ranks, strict=True)
    }
    _, critical_difference, figure = statds_nemenyi(
        rank_map, n_problems, alpha=alpha
    )
    return {
        "average_ranks": average_ranks,
        "critical_difference": float(critical_difference),
        "alpha": alpha,
        "n_problems": n_problems,
        "n_methods": n_methods,
        "ranks": ranks,
        "figure": figure,
    }


def save_cd_figure(
    figure: Any, path: str | Path, *, size: tuple[float, float] = (10.0, 2.2)
) -> Path:
    """Write a StaTDS critical-difference figure as vector PDF and SVG.

    The StaTDS canvas is very wide (15 in); resizing it to ``size`` keeps the
    labels legible when the figure is included at text width.
    """
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vector = out.suffix.lower() in {".pdf", ".svg"}
    if vector:
        figure.set_size_inches(*size)
    figure.savefig(out, bbox_inches="tight")
    if vector:
        sibling = out.with_suffix(".svg" if out.suffix.lower() != ".svg" else ".pdf")
        figure.savefig(sibling, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_cd_diagram(
    average_ranks: Mapping[str, float] | Sequence[float],
    critical_difference: float | None = None,
    *,
    names: Sequence[str] | None = None,
    path: str | Path | None = None,
    title: str = "",
    alpha: float = 0.05,
    num_cases: int | None = None,
) -> Any:
    """Draw a critical-difference diagram through StaTDS.

    Kept for callers that already hold average ranks and a critical difference.
    StaTDS needs the number of problems, so when ``num_cases`` is omitted it is
    inferred from ``critical_difference``; prefer :func:`nemenyi`, which returns
    the figure together with the exact test result.
    """
    from statds.no_parametrics import nemenyi as statds_nemenyi
    from statds.stats import get_q_alpha_nemenyi

    if isinstance(average_ranks, Mapping):
        rank_map = {str(name): float(rank) for name, rank in average_ranks.items()}
    else:
        if names is None:
            raise ValueError("names are required when ranks are given as a sequence.")
        rank_map = {
            str(name): float(rank)
            for name, rank in zip(names, average_ranks, strict=True)
        }
    if num_cases is None:
        n_methods = len(rank_map)
        if critical_difference and critical_difference > 0:
            q_alpha = get_q_alpha_nemenyi(n_methods, alpha)
            num_cases = max(
                2,
                int(
                    round(
                        q_alpha**2
                        * n_methods
                        * (n_methods + 1)
                        / (6.0 * critical_difference**2)
                    )
                ),
            )
        else:
            num_cases = 2 * n_methods
    _, _, figure = statds_nemenyi(rank_map, num_cases, alpha=alpha)
    if title:
        figure.suptitle(title)
    if path is not None:
        save_cd_figure(figure, path)
    return figure
