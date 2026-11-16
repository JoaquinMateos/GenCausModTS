"""Tests for graph utilities."""

from __future__ import annotations

import pytest
import torch

from gcmts.core.graph_ops import (
    acyclicity,
    aligned_shd,
    edge_auprc,
    estimate_instantaneous_adjacency,
    estimate_latent_adjacency,
    estimate_latent_structure,
    is_dag,
    latent_permutation,
    sample_dag,
    shd,
    topological_order,
    weighted_shd,
)


def _support(adj: torch.Tensor) -> torch.Tensor:
    return (adj.abs() > 0).int()


def _lagged_system(
    lag: torch.Tensor,
    *,
    instantaneous: torch.Tensor | None = None,
    seed: int,
    steps: int = 32,
    batch: int = 2048,
    noise: float = 0.5,
    burn_in: int = 32,
) -> torch.Tensor:
    """Simulate a linear latent SCM with the same-time innovations first.

    The innovation of node ``i`` is added *before* node ``i`` feeds the
    same-time parents of later nodes, which is the convention of the benchmark
    generators in :mod:`gcmts.data` (forward substitution in the causal order).
    Adding the innovation afterwards would leave each declared same-time edge
    absent from the observed variables.
    """
    torch.manual_seed(seed)
    p = lag.shape[-1]
    dim = lag.shape[0]
    inst = torch.zeros(dim, dim) if instantaneous is None else instantaneous
    z = torch.zeros(batch, steps + burn_in + p, dim)
    z[:, :p] = torch.randn(batch, p, dim)
    for t in range(p, steps + burn_in + p):
        current = noise * torch.randn(batch, dim)
        for lag_index in range(p):
            current = current + z[:, t - 1 - lag_index] @ lag[:, :, lag_index].T
        for i in range(dim):
            current[:, i] = current[:, i] + current[:, :i] @ inst[i, :i]
        z[:, t] = current
    return z[:, burn_in + p :]


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
    """A statistically supported edge below the old fixed 0.1 cutoff is kept.

    The pre-change rule selected a large spurious support with the pooled-row
    BIC and relied on a magnitude cutoff to remove it, which also deleted true
    edges of magnitude < 0.1. The calibrated rule keeps this edge because its
    cluster-robust t-statistic clears ``sqrt(log B)``.
    """
    torch.manual_seed(0)
    dim, steps, batch = 3, 64, 4096
    true = torch.zeros(dim, dim, 1)
    true[1, 0, 0] = 0.08  # weak edge, below the old 0.1 magnitude cutoff
    true[2, 1, 0] = 0.9
    z = torch.zeros(batch, steps + 1, dim)
    z[:, 0] = torch.randn(batch, dim)
    for t in range(1, steps + 1):
        z[:, t] = z[:, t - 1] @ true[:, :, 0].T + torch.randn(batch, dim)
    est = estimate_latent_adjacency(z[:, 1:], max_lag=1)
    assert torch.equal(_support(est), _support(true))
    assert est[1, 0, 0].abs().item() < 0.1


def test_estimate_latent_adjacency_reports_no_edges_for_iid_noise():
    """Dependent pooled rows must not manufacture edges from pure noise.

    The old pooled-row BIC accepted roughly ``T - p`` spurious parents per
    target at this sample size.
    """
    for seed in range(4):
        torch.manual_seed(seed)
        z = torch.randn(512, 32, 3)
        est = estimate_latent_adjacency(z, max_lag=1)
        assert int(_support(est).sum()) == 0


def test_estimate_latent_structure_recovers_lag_and_instantaneous_blocks():
    """Joint recovery of a known sparse lagged + same-time graph."""
    lag = torch.zeros(4, 4, 1)
    lag[0, 0, 0] = 0.2
    lag[1, 1, 0] = 0.3
    lag[2, 1, 0] = 0.7
    inst = torch.zeros(4, 4)
    inst[1, 0] = 0.8
    inst[3, 2] = 0.6
    for seed in range(3):
        z = _lagged_system(lag, instantaneous=inst, seed=seed)
        lag_est, inst_est = estimate_latent_structure(z, max_lag=1)
        assert torch.equal(_support(lag_est), _support(lag))
        assert torch.equal(_support(inst_est), _support(inst))
        assert shd(lag_est, lag) == 0
        assert shd(inst_est, inst) == 0


def test_estimate_latent_structure_removes_omitted_variable_bias():
    """A same-time parent must not be re-read as a lagged edge.

    ``z1`` depends only on ``z0`` at the same time step and ``z2`` on
    ``z1`` at lag one. A lagged-only regression of ``z1`` picks up
    ``z0_{t-1}`` (through its shadow of ``z0_t``); the conditional joint fit
    must drop it while keeping the genuine instantaneous edge.
    """
    torch.manual_seed(0)
    batch, steps, burn = 2048, 32, 32
    z = torch.zeros(batch, steps + burn + 1, 3)
    z[:, 0] = torch.randn(batch, 3)
    for t in range(1, steps + burn + 1):
        noise = 0.3 * torch.randn(batch, 3)
        z0 = 0.8 * z[:, t - 1, 0] + noise[:, 0]
        z1 = 0.7 * z0 + noise[:, 1]
        z2 = 0.9 * z[:, t - 1, 1] + noise[:, 2]
        z[:, t] = torch.stack([z0, z1, z2], dim=-1)
    z = z[:, burn + 1 :]

    lag_only = estimate_latent_adjacency(z, max_lag=1)
    assert lag_only[1, 0, 0].abs() > 0.3  # spurious shadow edge

    lag_est, inst_est = estimate_latent_structure(z, max_lag=1)
    assert lag_est[1, 0, 0].item() == 0.0
    assert lag_est[0, 0, 0].abs() > 0.5  # genuine self-persistence
    assert lag_est[2, 1, 0].abs() > 0.5  # genuine lagged edge
    assert inst_est[1, 0].abs() > 0.5  # genuine instantaneous edge
    assert int(_support(inst_est).sum()) == 1


def test_estimate_latent_adjacency_respects_edge_direction():
    """The estimated edge points forward; a reversal is one SHD edit."""
    true = torch.zeros(2, 2, 1)
    true[1, 0, 0] = 0.8
    z = _lagged_system(true, seed=0, steps=64, batch=2048, noise=0.1)
    est = estimate_latent_adjacency(z, max_lag=1)
    assert torch.equal(_support(est), _support(true))
    assert est[0, 1, 0].item() == 0.0
    reversed_est = est.clone()
    reversed_est[0, 1, 0] = est[1, 0, 0]
    reversed_est[1, 0, 0] = 0.0
    assert shd(reversed_est, true) == 1


def test_shd_counts_a_reversal_as_one_edit():
    true = torch.zeros(3, 3)
    true[1, 0] = 1.0
    reversed_est = true.T.clone()
    assert shd(reversed_est, true) == 1


def test_estimate_latent_structure_support_is_insensitive_to_t_threshold():
    """The support is stable across the plausible t-statistic cutoffs."""
    lag = torch.zeros(4, 4, 1)
    for i in range(4):
        lag[i, i, 0] = 0.6
    lag[2, 1, 0] = 0.4
    z = _lagged_system(lag, seed=3, steps=32, batch=1024, noise=0.5)
    supports = []
    for tau in (1.5, 2.0, 2.76, 3.5, 4.5):
        lag_est, _ = estimate_latent_structure(z, max_lag=1, t_threshold=tau)
        supports.append(_support(lag_est))
    assert all(torch.equal(s, supports[0]) for s in supports)
    assert shd(supports[0].float(), lag) == 0


def test_estimate_latent_structure_handles_two_lags():
    lag = torch.zeros(3, 3, 2)
    lag[0, 0, 0] = 0.4
    lag[0, 1, 1] = 0.5
    lag[1, 0, 0] = 0.6
    inst = torch.zeros(3, 3)
    inst[2, 1] = 0.7
    z = _lagged_system(lag, instantaneous=inst, seed=0, steps=48, noise=0.4)
    lag_est, inst_est = estimate_latent_structure(z, max_lag=2)
    assert torch.equal(_support(lag_est), _support(lag))
    assert torch.equal(_support(inst_est), _support(inst))


def test_estimate_instantaneous_adjacency_uses_intervention_masks():
    """Dropping intervened steps recovers the attenuated mechanism edge."""
    torch.manual_seed(0)
    batch, steps, dim = 2048, 32, 3
    z = torch.zeros(batch, steps, dim)
    masks = torch.zeros(batch, steps, dim, dtype=torch.bool)
    z[:, 0] = torch.randn(batch, dim)
    for t in range(1, steps):
        intervened = torch.rand(batch) < 0.4
        z0 = 0.6 * z[:, t - 1, 0] + 0.1 * torch.randn(batch)
        z1 = 0.8 * z0 + 0.1 * torch.randn(batch)
        z[:, t, 0] = z0
        z[:, t, 1] = torch.where(intervened, torch.randn(batch), z1)
        z[:, t, 2] = 0.5 * z[:, t - 1, 2] + 0.1 * torch.randn(batch)
        masks[:, t, 1] = intervened
    plain = estimate_instantaneous_adjacency(z)
    masked = estimate_instantaneous_adjacency(z, interventions=masks)
    assert plain[1, 0].abs() > 0.0
    assert masked[1, 0].abs() > 2.0 * plain[1, 0].abs()
    assert int(_support(masked).sum()) == 1


def test_estimate_latent_structure_shape_edges_and_dtype():
    z = torch.randn(16, 2, 3, dtype=torch.float64)
    lag, inst = estimate_latent_structure(z, max_lag=3)
    # T <= p -> all-zero blocks, right shapes and dtype.
    assert lag.shape == (3, 3, 3)
    assert inst.shape == (3, 3)
    assert lag.dtype == inst.dtype == torch.float64
    assert float(lag.abs().sum()) == 0.0


def test_estimate_latent_structure_mask_shape_is_validated():
    z = torch.randn(4, 12, 3)
    with pytest.raises(ValueError, match="intervention_masks"):
        estimate_latent_structure(z, interventions=torch.zeros(4, 11, 3))


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
