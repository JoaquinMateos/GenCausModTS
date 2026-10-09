"""Tests for graph utilities."""

from __future__ import annotations

import pytest
import torch

from gcmts.core.graph_ops import (
    acyclicity,
    aligned_shd,
    edge_auprc,
    estimate_latent_adjacency,
    is_dag,
    latent_permutation,
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


def test_estimate_latent_adjacency_recovers_known_graph():
    torch.manual_seed(0)
    dim, steps, batch = 3, 300, 8
    true = torch.zeros(dim, dim, 1)
    true[1, 0, 0] = 0.9  # z1_t <- z0_{t-1}
    true[2, 1, 0] = 0.9  # z2_t <- z1_{t-1}
    z = torch.zeros(batch, steps + 1, dim)
    z[:, 0] = torch.randn(batch, dim)
    for t in range(1, steps + 1):
        z[:, t] = z[:, t - 1] @ true[:, :, 0].T + 0.01 * torch.randn(batch, dim)
    est = estimate_latent_adjacency(z[:, 1:], max_lag=1, threshold=0.1)
    assert torch.equal((est.abs() > 0).float(), (true.abs() > 0).float())


def test_aligned_shd_is_permutation_invariant():
    true = torch.zeros(3, 3, 1)
    true[1, 0, 0] = 1.0
    true[2, 1, 0] = 1.0
    perm = [2, 0, 1]
    permuted = true[torch.tensor(perm)][:, torch.tensor(perm)]
    assert aligned_shd(permuted, true) == 0.0


def test_aligned_shd_counts_missing_edge():
    true = torch.zeros(3, 3, 1)
    true[1, 0, 0] = 1.0
    est = torch.zeros(3, 3, 1)
    assert aligned_shd(est, true) == 1.0


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


def test_estimate_latent_adjacency_keeps_weak_selected_edges():
    """Support selection must keep weak true edges a fixed 0.1 cutoff drops."""
    torch.manual_seed(0)
    dim, steps, batch = 3, 800, 16
    true = torch.zeros(dim, dim, 1)
    true[1, 0, 0] = 0.05  # weak edge, below the old 0.1 magnitude cutoff
    true[2, 1, 0] = 0.9
    z = torch.zeros(batch, steps + 1, dim)
    z[:, 0] = torch.randn(batch, dim)
    for t in range(1, steps + 1):
        z[:, t] = z[:, t - 1] @ true[:, :, 0].T + 0.02 * torch.randn(batch, dim)
    est = estimate_latent_adjacency(z[:, 1:], max_lag=1)
    assert torch.equal((est.abs() > 0).float(), (true.abs() > 0).float())


def test_estimate_latent_adjacency_lag_axis_order():
    """A lag-2 edge must land on the correct lag slice."""
    torch.manual_seed(1)
    dim, steps, batch = 2, 800, 16
    true = torch.zeros(dim, dim, 2)
    true[0, 1, 0] = 0.9  # z0_t <- z1_{t-1}
    true[1, 0, 1] = 0.7  # z1_t <- z0_{t-2}
    z = torch.zeros(batch, steps + 2, dim)
    z[:, 0] = torch.randn(batch, dim)
    z[:, 1] = torch.randn(batch, dim)
    for t in range(2, steps + 2):
        z[:, t] = (
            z[:, t - 1] @ true[:, :, 0].T
            + z[:, t - 2] @ true[:, :, 1].T
            + 0.01 * torch.randn(batch, dim)
        )
    est = estimate_latent_adjacency(z, max_lag=2)
    assert torch.equal((est.abs() > 0).float(), (true.abs() > 0).float())


def test_latent_permutation_recovers_shuffle():
    torch.manual_seed(0)
    true = torch.randn(64, 3)
    shuffle = torch.tensor([2, 0, 1])
    predicted = true[:, shuffle] + 0.01 * torch.randn(64, 3)
    recovered = latent_permutation(predicted, true)
    assert torch.equal(recovered, torch.argsort(shuffle))


def test_edge_auprc_prefers_correct_ranking():
    torch.manual_seed(0)
    true = torch.zeros(3, 3, 1)
    true[1, 0, 0] = 1.0
    true[2, 1, 0] = 1.0
    good = true * 0.9
    bad = torch.zeros_like(true)
    bad[0, 1, 0] = 0.9  # scores a wrong edge highest
    assert edge_auprc(good, true) > edge_auprc(bad, true)


def test_aligned_shd_greedy_path_for_large_dim():
    dim = 9
    true = torch.zeros(dim, dim, 1)
    true[1, 0, 0] = 1.0
    true[2, 1, 0] = 1.0
    perm = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    index = torch.tensor(perm)
    permuted = true[index][:, index]
    assert aligned_shd(permuted, true) == 0.0
