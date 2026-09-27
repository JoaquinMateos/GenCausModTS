"""Graph utilities for latent causal structure.

Provides DAG sampling, the smooth acyclicity constraint used by
differentiable causal discovery, structural Hamming distance, and topological
ordering. Lagged adjacency helpers live in :mod:`gcmts.core.utils` and are
re-exported here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gcmts.core.utils import lagged_adjacency_to_dag

__all__ = [
    "sample_dag",
    "acyclicity",
    "is_dag",
    "shd",
    "weighted_shd",
    "topological_order",
    "lagged_adjacency_to_dag",
]


def sample_dag(
    n_nodes: int,
    *,
    sparsity: float = 0.3,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample a random DAG adjacency matrix of shape ``(n_nodes, n_nodes)``.

    Edges are drawn in the strict upper triangle, so the result is acyclic by
    construction. Values are uniform in ``(0, 1)`` masked by ``sparsity``.
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must lie in [0, 1].")
    weights = torch.rand(n_nodes, n_nodes, generator=generator, dtype=dtype)
    mask = (torch.rand(n_nodes, n_nodes, generator=generator, dtype=dtype) < sparsity).to(dtype)
    upper = torch.triu(torch.ones(n_nodes, n_nodes, dtype=dtype), diagonal=1)
    return weights * mask * upper


def acyclicity(adj: Tensor) -> Tensor:
    r"""Smooth acyclicity constraint ``h(A) = Tr(e^{A \circ A}) - d``.

    Equals ``0`` for a DAG and is strictly positive otherwise (NOTEARS,
    Zheng et al., 2018). Works with batched or single adjacency matrices.
    """
    if adj.ndim == 2:
        a = adj.unsqueeze(0)
        squeeze = True
    else:
        a = adj
        squeeze = False
    d = a.shape[-1]
    h = torch.matrix_exp(a * a).diagonal(dim1=-2, dim2=-1).sum(-1) - d
    return h.squeeze(0) if squeeze else h


def is_dag(adj: Tensor, *, tol: float = 1e-6) -> bool:
    """Return whether ``adj`` is acyclic (checked via the exact constraint)."""
    return bool(acyclicity(adj).abs().max().item() < tol)


def _binarise(adj: Tensor, threshold: float) -> Tensor:
    return (adj.abs() > threshold).to(torch.int64)


def shd(
    adj_est: Tensor,
    adj_true: Tensor,
    *,
    threshold: float = 1e-3,
) -> int:
    """Structural Hamming Distance (edge add + delete + reverse).

    Reversed edges are counted as two structural differences, matching the
    convention used in the review. Adjacency matrices must share a shape.
    """
    if adj_est.shape != adj_true.shape:
        raise ValueError("Adjacency matrices must share a shape.")
    est = _binarise(adj_est, threshold)
    true = _binarise(adj_true, threshold)
    return int((est != true).sum().item())


def weighted_shd(
    adj_est: Tensor,
    adj_true: Tensor,
    *,
    threshold: float = 1e-3,
) -> float:
    """SHD variant that weights each structural difference by edge magnitude."""
    est = _binarise(adj_est, threshold)
    true = _binarise(adj_true, threshold)
    weight = torch.maximum(adj_est.abs(), adj_true.abs())
    return float(((est != true).float() * weight).sum().item())


def topological_order(adj: Tensor) -> list[int]:
    """Kahn's algorithm; raises :class:`ValueError` if the graph is cyclic."""
    n = adj.shape[0]
    a = _binarise(adj, 0.0)
    in_degree = a.sum(dim=0).tolist()
    queue = [i for i in range(n) if in_degree[i] == 0]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in range(n):
            if a[node, child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    if len(order) != n:
        raise ValueError("Adjacency matrix contains a cycle.")
    return order
