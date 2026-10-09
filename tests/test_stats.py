"""Tests for statistical comparison utilities (Section 12.2 of the plan)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gcmts.evaluation.metrics import (
    JFTSDMetric,
    TripletDistanceMetric,
    TwoSampleDiscriminatorMetric,
)
from gcmts.evaluation.stats import (
    friedman_test,
    load_rows,
    metric_matrix,
    nemenyi,
    paired_wilcoxon_holm,
    plot_cd_diagram,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Method B is uniformly better than A by a constant shift; C is noisier.
    rng = np.random.default_rng(0)
    for problem in ("p1", "p2", "p3"):
        for seed in range(6):
            a = float(rng.normal(1.0, 0.1))
            rows.append({"case": "A", "dataset": problem, "seed": seed, "mcc": a})
            rows.append({"case": "B", "dataset": problem, "seed": seed, "mcc": a + 0.5})
            noisy = a + float(rng.normal(0, 0.4))
            rows.append({"case": "C", "dataset": problem, "seed": seed, "mcc": noisy})
    return rows


def test_paired_wilcoxon_holm_finds_shift() -> None:
    comparisons = paired_wilcoxon_holm(_rows(), "mcc")
    assert comparisons
    shift = next(c for c in comparisons if {c.case_a, c.case_b} == {"A", "B"})
    assert shift.significant
    assert shift.p_holm < 0.05
    assert shift.n_pairs == 18
    assert all(0.0 <= c.p_holm <= 1.0 for c in comparisons)
    assert [c.p_holm for c in comparisons] == sorted(c.p_holm for c in comparisons)


def test_metric_matrix_and_ranks() -> None:
    matrix, cases, problems = metric_matrix(_rows(), "mcc")
    assert cases == ["A", "B", "C"]
    assert problems == ["p1", "p2", "p3"]
    assert matrix.shape == (3, 3)
    result = nemenyi(matrix)
    # B is a constant shift better than A, so it must rank first (rank one).
    assert result["average_ranks"][1] < result["average_ranks"][0]
    assert result["average_ranks"][1] == pytest.approx(1.0)
    error_result = nemenyi(matrix, higher_is_better=False)
    assert error_result["average_ranks"][1] > error_result["average_ranks"][0]
    assert result["critical_difference"] > 0
    statistic = friedman_test(matrix)
    assert 0.0 <= statistic["p_value"] <= 1.0


def test_paired_wilcoxon_missing_seed_pairs_by_order() -> None:
    # Freedman-style rows without a seed column: pairs are formed by order.
    rows = [
        {"case": "A", "mcc": value} for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ] + [{"case": "B", "mcc": value} for value in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)]
    comparisons = paired_wilcoxon_holm(rows, "mcc", min_pairs=3)
    assert len(comparisons) == 1
    assert comparisons[0].n_pairs == 8
    assert comparisons[0].significant


def test_plot_cd_diagram_writes_file(tmp_path: Path) -> None:
    ranks = {"A": 1.0, "B": 2.5, "C": 2.5, "D": 4.0}
    path = tmp_path / "cd.png"
    plot_cd_diagram(ranks, 1.5, path=path, title="unit test")
    assert path.exists() and path.stat().st_size > 0


def test_load_rows_roundtrip(tmp_path: Path) -> None:
    import json

    path = tmp_path / "rows.json"
    rows = _rows()[:4]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert len(load_rows(path)) == 4


def test_triplet_distance_isolated_intervention() -> None:
    torch.manual_seed(0)
    n = 200
    truth = torch.randn(n, 3)
    before = truth + 0.01 * torch.randn(n, 3)
    after = before.clone()
    targets = torch.full((n,), 2, dtype=torch.long)
    after[:, 2] = after[:, 2] + 2.0
    metric = TripletDistanceMetric()
    isolated = metric({"before": before, "after": after, "targets": targets}, truth).value
    after_leaky = after.clone()
    after_leaky[:, 0] = after_leaky[:, 0] + 2.0
    leaky = metric(
        {"before": before, "after": after_leaky, "targets": targets}, truth
    ).value
    assert isolated < 0.2
    assert leaky > isolated


def test_two_sample_discriminator_matches_distributions() -> None:
    torch.manual_seed(0)
    real = torch.randn(400, 4)
    same = torch.randn(400, 4)
    shifted = torch.randn(400, 4) + 5.0
    metric = TwoSampleDiscriminatorMetric()
    indistinguishable = metric(same, real).value
    separable = metric(shifted, real).value
    assert 0.0 <= indistinguishable <= 0.15
    assert separable > 0.45


def test_policy_risk_threshold_rule() -> None:
    from gcmts.evaluation.metrics import PolicyRiskMetric

    target = {
        "yf": torch.tensor([10.0, 0.0]),
        "ycf": torch.tensor([0.0, 10.0]),
        "t": torch.tensor([1, 0]),
    }
    tau = torch.tensor([2.0, -2.0])
    result = PolicyRiskMetric()(tau, target)
    assert result.value == pytest.approx(5.0)


@pytest.mark.parametrize(
    "metric_name",
    ["triplet", "two_sample", "policy_risk", "jftsd", "edge_auprc"],
)
def test_new_metrics_registered(metric_name: str) -> None:
    from gcmts.evaluation.metrics import DEFAULT_REGISTRY

    assert metric_name in DEFAULT_REGISTRY.list_metrics()


def test_jftsd_zero_for_identical_embeddings() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(200, 5)
    metric = JFTSDMetric()
    assert metric(embeddings, embeddings).value == pytest.approx(0.0, abs=1e-4)
    assert metric(embeddings + 1.0, embeddings).value > 0.5
