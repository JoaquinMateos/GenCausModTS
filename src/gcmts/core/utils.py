"""Shared tensor/graph utilities."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

__all__ = [
    "lagged_adjacency_to_dag",
    "mcc",
    "hungarian_match",
    "make_mlp",
]


def lagged_adjacency_to_dag(adj: Tensor) -> Tensor:
    """Convert a lagged adjacency ``(d, d, p)`` into a full DAG tensor.

    Returns a ``(d * (p + 1), d * (p + 1))`` binary matrix where the first
    ``d*p`` rows correspond to past time slices and the last ``d`` rows to the
    present.
    """
    d, _, p = adj.shape
    total = d * (p + 1)
    dag = torch.zeros(total, total, dtype=adj.dtype, device=adj.device)
    for lag in range(p):
        dag[lag * d : (lag + 1) * d, -d:] = adj[:, :, lag]
    return dag


def hungarian_match(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Solve a linear assignment problem given a ``(n, m)`` cost matrix.

    Returns ``(row_idx, col_idx)`` such that ``cost[row_idx, col_idx]`` is
    minimised.
    """
    row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(row_idx), torch.as_tensor(col_idx)


def mcc(z_true: Tensor, z_pred: Tensor) -> float:
    """Mean correlation coefficient after Hungarian matching.

    Computes ``max_{π} (1/d) Σ_i |C_{i,π(i)}|`` where ``C`` is the Pearson
    correlation matrix. This is invariant to permutation and monotone rescaling
    of individual components.
    """
    z_true = z_true.reshape(-1, z_true.shape[-1])
    z_pred = z_pred.reshape(-1, z_pred.shape[-1])
    z_true = (z_true - z_true.mean(0)) / (z_true.std(0) + 1e-8)
    z_pred = (z_pred - z_pred.mean(0)) / (z_pred.std(0) + 1e-8)
    corr = torch.abs((z_true.T @ z_pred) / z_true.shape[0])
    _, col_idx = hungarian_match(-corr)
    score = corr[torch.arange(corr.shape[0]), col_idx].mean().item()
    return float(score)


def make_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    *,
    activation: str = "relu",
    batchnorm: bool = False,
    final_activation: bool = False,
) -> torch.nn.Module:
    """Build a small MLP backbone used by many methods."""
    layers: list[torch.nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]
    act = {"relu": torch.nn.ReLU, "elu": torch.nn.ELU, "silu": torch.nn.SiLU}[
        activation
    ]
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
            layers.append(act())
        elif final_activation:
            layers.append(act())
    return torch.nn.Sequential(*layers)
