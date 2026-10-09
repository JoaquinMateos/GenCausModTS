"""Statistical comparison utilities for the benchmark tables.

This module implements the protocol fixed in the repository plan
(Section 12.2 of ``.AGENTS.md``):

* :func:`paired_wilcoxon_holm` -- paired Wilcoxon signed-rank tests between
  every pair of methods, Holm-corrected over the family of tests, with pairs
  formed by ``(problem, seed)`` so that every comparison uses the same data
  instances.
* :func:`friedman_test` -- omnibus Friedman test over a rank matrix of shape
  ``(n_problems, n_methods)``.
* :func:`nemenyi` -- Nemenyi post-hoc test with the critical difference
  ``CD = q_alpha * sqrt(k (k + 1) / (6 N))`` (Demsar, 2006).
* :func:`plot_cd_diagram` -- critical-difference diagram for average ranks.

The statistics are computed on per-seed rows (the raw JSON records written by
:func:`gcmts.evaluation.report.save_rows`), so they are consistent with the
reported mean/std.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

__all__ = [
    "Comparison",
    "load_rows",
    "paired_wilcoxon_holm",
    "metric_matrix",
    "friedman_test",
    "nemenyi",
    "plot_cd_diagram",
]

# Studentized-range critical values q_alpha for the two-tailed Nemenyi test at
# alpha = 0.05 (Demsar, 2006, Table 5). Index by the number of methods k.
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}


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


def friedman_test(rank_matrix: np.ndarray) -> dict[str, float]:
    """Omnibus Friedman test on a ``(n_problems, n_methods)`` metric matrix."""
    from scipy.stats import friedmanchisquare

    matrix = _complete_matrix(rank_matrix)
    if matrix.shape[0] < 2:
        raise ValueError("Friedman test requires at least two complete problems.")
    if matrix.shape[1] < 3:
        raise ValueError("Friedman test requires at least three methods.")
    statistic, p_value = friedmanchisquare(*matrix.T)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_problems": float(matrix.shape[0]),
        "n_methods": float(matrix.shape[1]),
    }


def nemenyi(
    rank_matrix: np.ndarray, *, alpha: float = 0.05, higher_is_better: bool = True
) -> dict[str, Any]:
    """Nemenyi post-hoc test and critical difference on a metric matrix.

    Ranks are oriented so that the best method receives rank one: pass
    ``higher_is_better=False`` for error metrics (lower is better). Returns a
    dictionary with the average ranks per column, the pairwise Nemenyi p-value
    matrix, the critical difference and the raw rank matrix.
    """
    from scikit_posthocs import posthoc_nemenyi_friedman
    from scipy.stats import rankdata

    matrix = _complete_matrix(rank_matrix)
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError("Nemenyi requires >= 2 complete problems and >= 3 methods.")
    n_problems, n_methods = matrix.shape
    if n_methods not in _NEMENYI_Q05:
        raise ValueError(
            f"No Nemenyi q_alpha entry for k={n_methods} methods (max 15)."
        )
    orientation = -matrix if higher_is_better else matrix
    ranks = np.apply_along_axis(rankdata, 1, orientation)
    average_ranks = ranks.mean(axis=0)
    p_matrix = posthoc_nemenyi_friedman(matrix).to_numpy(dtype=float)
    critical_difference = _NEMENYI_Q05[n_methods] * math.sqrt(
        n_methods * (n_methods + 1) / (6.0 * n_problems)
    )
    return {
        "average_ranks": average_ranks,
        "p_matrix": p_matrix,
        "critical_difference": critical_difference,
        "q_alpha": _NEMENYI_Q05[n_methods],
        "alpha": alpha,
        "n_problems": n_problems,
        "n_methods": n_methods,
        "ranks": ranks,
    }


def _cliques(sorted_ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal intervals of methods whose ranks differ by at most ``cd``."""
    n = len(sorted_ranks)
    cliques: list[tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            if cliques and cliques[-1][0] <= i and cliques[-1][1] >= j:
                continue
            cliques.append((i, j))
    return cliques


def plot_cd_diagram(
    average_ranks: Mapping[str, float] | Sequence[float],
    critical_difference: float | None = None,
    *,
    names: Sequence[str] | None = None,
    path: str | Path | None = None,
    title: str = "",
    alpha: float = 0.05,
) -> Any:
    """Draw a Demsar-style critical-difference diagram.

    ``average_ranks`` may be a mapping ``name -> rank`` or a sequence of ranks
    (then ``names`` must be given). When ``critical_difference`` is omitted
    (allowed with a mapping or sequence), only the rank axis is drawn. The
    figure is saved to ``path`` when provided. Returns the matplotlib axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(average_ranks, Mapping):
        ordered = sorted(average_ranks.items(), key=lambda item: item[1])
        names_sorted = [name for name, _ in ordered]
        ranks_sorted = np.array([rank for _, rank in ordered], dtype=float)
    else:
        if names is None:
            raise ValueError("names are required when ranks are given as a sequence.")
        pairs = sorted(zip(names, average_ranks, strict=True), key=lambda item: item[1])
        names_sorted = [name for name, _ in pairs]
        ranks_sorted = np.array([rank for _, rank in pairs], dtype=float)

    n = len(ranks_sorted)
    labels = [f"{name} ({rank:.2f})" for name, rank in zip(names_sorted, ranks_sorted, strict=True)]

    height = max(2.0, 0.62 * n + 1.2)
    _, ax = plt.subplots(figsize=(8.0, height))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-0.5, n - 0.1 + 0.8)

    axis_y = n - 1
    ax.hlines(axis_y, 0.5, n + 0.5, color="black", linewidth=1.2)
    for tick in range(1, n + 1):
        ax.vlines(tick, axis_y - 0.06, axis_y + 0.06, color="black", linewidth=1.0)
        ax.text(float(tick), axis_y + 0.16, str(tick), ha="center", va="bottom", fontsize=10)

    for index, (label, rank) in enumerate(zip(labels, ranks_sorted, strict=True)):
        y = n - 1.6 - index * 0.55
        ax.plot([rank], [y], marker="o", markersize=5, color="black")
        ax.hlines(y, rank, n + 0.45, color="gray", linewidth=0.7, linestyle="-")
        ax.text(n + 0.5, y, label, va="center", ha="left", fontsize=9)

    if critical_difference is not None and critical_difference > 0:
        for start, end in _cliques(ranks_sorted, critical_difference):
            y = axis_y - 0.28 - 0.001
            ax.hlines(
                y,
                ranks_sorted[start],
                ranks_sorted[end],
                color="black",
                linewidth=3.0,
                alpha=0.85,
            )

    ax.set_title(title or "Critical difference diagram", fontsize=11)
    ax.axis("off")
    from matplotlib.figure import Figure

    figure = cast(Figure, ax.get_figure())
    figure.tight_layout()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(figure)
    return ax
