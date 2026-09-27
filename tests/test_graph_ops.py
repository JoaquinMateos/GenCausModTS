"""Tests for graph utilities."""

from __future__ import annotations

import pytest
import torch

from gcmts.core.graph_ops import (
    acyclicity,
    is_dag,
    sample_dag,
    shd,
    topological_order,
    weighted_shd,
)


def test_sample_dag_is_acyclic():
    adj = sample_dag(10, sparsity=0.4, generator=torch.Generator().manual_seed(0))
    assert is_dag(adj)
    assert acyclicity(adj).abs().item() < 1e-4
    assert adj.shape == (10, 10)


def test_acyclicity_positive_for_cycle():
    adj = torch.zeros(3, 3)
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    adj[2, 0] = 1.0
    assert acyclicity(adj).item() > 1e-3
    assert not is_dag(adj)


def test_shd_identical_is_zero():
    adj = sample_dag(6, sparsity=0.5)
    assert shd(adj, adj) == 0
    assert weighted_shd(adj, adj) == 0.0


def test_shd_counts_differences():
    true = torch.zeros(3, 3)
    true[0, 1] = 1.0
    est = torch.zeros(3, 3)
    est[0, 2] = 1.0
    assert shd(est, true) == 2


def test_topological_order():
    adj = torch.zeros(4, 4)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[1, 3] = 1.0
    order = topological_order(adj)
    position = {node: i for i, node in enumerate(order)}
    assert position[0] < position[1] < position[3]
    assert position[0] < position[2]


def test_topological_order_raises_on_cycle():
    adj = torch.zeros(2, 2)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    with pytest.raises(ValueError):
        topological_order(adj)
