"""Graph utilities for latent causal structure.

Provides DAG sampling, the smooth acyclicity constraint used by
differentiable causal discovery, structural Hamming distance, and topological
ordering. Lagged adjacency helpers live in :mod:`gcmts.core.utils` and are
re-exported here.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch
from sklearn.linear_model import LassoLarsIC
from sklearn.metrics import average_precision_score
from torch import Tensor

from gcmts.core.utils import hungarian_match, lagged_adjacency_to_dag

__all__ = [
    "sample_dag",
    "acyclicity",
    "is_dag",
    "shd",
    "weighted_shd",
    "aligned_shd",
    "latent_permutation",
    "estimate_latent_adjacency",
    "edge_auprc",
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
    true = _binarise(adj_true.to(est.device), threshold)
    return int((est != true).sum().item())


def weighted_shd(
    adj_est: Tensor,
    adj_true: Tensor,
    *,
    threshold: float = 1e-3,
) -> float:
    """SHD variant that weights each structural difference by edge magnitude."""
    adj_true = adj_true.to(adj_est.device)
    est = _binarise(adj_est, threshold)
    true = _binarise(adj_true, threshold)
    weight = torch.maximum(adj_est.abs(), adj_true.abs())
    return float(((est != true).float() * weight).sum().item())


def _sparse_lagged_coefficients(design: Tensor, target: Tensor, p: int) -> Tensor:
    r"""Per-target sparse regression with debiasing on the selected support.

    Follows the post-hoc graph-recovery protocol of TDRL and LEAP, which fit a
    sparse predictor (LassoNet, hierarchical soft-thresholding) on the learned
    latents and read the selected inputs off its support. We use the LassoLars
    path with the BIC criterion (Zou, Hastie & Tibshirani, 2007), the standard
    convex surrogate for that selection, and re-fit ordinary least squares on
    the selected support to remove the :math:`\ell_1` shrinkage bias.

    Both the design and each target are standardised column-wise first, so the
    returned coefficients are scale-free and comparable across methods and
    datasets (a fixed magnitude threshold is not meaningful otherwise).

    Returns a ``(d * p, d)`` matrix whose entry ``(j * p + l, i)`` is the
    coefficient of ``z_{j,t-1-l}`` in the regression for ``z_{i,t}``.
    """
    x_np = design.detach().cpu().double().numpy()
    y_np = target.detach().cpu().double().numpy()
    x_scale = x_np.std(axis=0, keepdims=True)
    dead = x_scale.ravel() < 1e-8
    x_np = (x_np - x_np.mean(axis=0, keepdims=True)) / np.where(dead, 1.0, x_scale)
    x_np[:, dead] = 0.0
    coefficients = np.zeros((x_np.shape[1], y_np.shape[1]), dtype=np.float64)
    for i in range(y_np.shape[1]):
        y = y_np[:, i]
        scale = float(y.std())
        if scale < 1e-8:
            continue
        y = (y - y.mean()) / scale
        selector = LassoLarsIC(criterion="bic").fit(x_np, y)
        support = np.flatnonzero(np.abs(selector.coef_) > 0.0)
        if support.size == 0:
            continue
        debiased, *_ = np.linalg.lstsq(x_np[:, support], y, rcond=None)
        coefficients[support, i] = debiased
    return torch.from_numpy(coefficients).to(design.dtype)


def estimate_latent_adjacency(
    z: Tensor,
    *,
    max_lag: int = 1,
    threshold: float = 0.0,
    standardize: bool = True,
) -> Tensor:
    """Recover a sparse lagged latent graph from trajectories.

    Post-hoc protocol of TDRL and LEAP (LassoNet on the learned latents): each
    target latent ``z_{i,t}`` is regressed on all lagged latents
    ``{z_{j,t-1-l}}`` after standardisation; the support is selected with a
    BIC-penalised Lasso path and the coefficients are debiased by ordinary
    least squares on that support. No untuned magnitude cutoff is applied by
    default (``threshold=0``), because the generator edge weights are uniform
    in ``(0, mechanism_scale)`` and a fixed cutoff silently deletes weak edges.

    The result is an ``(d, d, p)`` tensor where ``A[i, j, l] != 0`` means
    ``z_{i,t}`` depends on ``z_{j,t-1-l}``. Entries on the selected support
    carry the debiased regression coefficient; everything else is exactly
    zero.

    Args:
        z: latents of shape ``(B, T, d)``.
        max_lag: temporal lag ``p``.
        threshold: optional extra magnitude cutoff applied after debiasing.
        standardize: center/scale each latent across batch and time first.
    """
    if z.ndim != 3:
        raise ValueError("estimate_latent_adjacency expects z of shape (B, T, d).")
    batch, time, dim = z.shape
    p = max(1, int(max_lag))
    if time <= p:
        return torch.zeros(dim, dim, p, dtype=z.dtype)
    zc = z
    if standardize:
        mean = z.mean(dim=(0, 1), keepdim=True)
        scale = z.std(dim=(0, 1), keepdim=True) + 1e-6
        zc = (z - mean) / scale
    features = torch.stack(
        [zc[:, p - 1 - lag : time - 1 - lag, :] for lag in range(p)], dim=-1
    )  # (B, T-p, d, p); column index j * p + l
    design = features.reshape(-1, dim * p)
    target = zc[:, p:, :].reshape(-1, dim)
    coef = _sparse_lagged_coefficients(design, target, p)  # (d*p, d)
    adj = coef.reshape(dim, p, dim).permute(2, 0, 1).contiguous()  # (d, d, p)
    if threshold > 0:
        adj = torch.where(adj.abs() > threshold, adj, torch.zeros_like(adj))
    return adj.to(z.dtype)


def latent_permutation(z_pred: Tensor, z_true: Tensor) -> Tensor:
    """Match predicted latent coordinates to ground-truth factors.

    Returns ``perm`` of shape ``(d,)`` with ``perm[a]`` the predicted-latent
    index assigned to true factor ``a``, obtained by Hungarian matching on the
    absolute Pearson correlation matrix. TDRL and LEAP align structural errors
    with exactly this assignment (the same one used by MCC), so applying it
    before an SHD/WSHD/AUPRC comparison makes those metrics permutation
    invariant without an exhaustive graph search.
    """
    if z_pred.shape[-1] != z_true.shape[-1]:
        raise ValueError("Latent and ground-truth dimensions must match.")
    pred = z_pred.reshape(-1, z_pred.shape[-1]).detach().cpu().float()
    true = z_true.reshape(-1, z_true.shape[-1]).detach().cpu().float()
    if pred.shape[0] != true.shape[0]:
        raise ValueError("Latents and factors must share a sample axis.")
    pred = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
    true = (true - true.mean(0)) / (true.std(0) + 1e-8)
    corr = (true.T @ pred) / pred.shape[0]
    _, col = hungarian_match(-corr.abs())
    return col.long()


def edge_auprc(adj_est: Tensor, adj_true: Tensor, *, threshold: float = 1e-3) -> float:
    """Threshold-agnostic edge recovery: average precision of edge scores.

    Following CaDRe, structural recovery is reported both as a thresholded SHD
    and as a ranking score: the absolute estimated coefficients are ranked
    against the binarised ground-truth edges and scored with average precision
    (area under the precision--recall curve). Unlike SHD this does not depend
    on a binarisation cutoff. Returns ``nan`` when the ground truth has no
    edges or is complete.
    """
    truth = (adj_true.abs() > threshold).reshape(-1).detach().cpu().numpy()
    scores = adj_est.abs().reshape(-1).detach().cpu().numpy()
    if truth.size == 0 or truth.all() or not truth.any():
        return float("nan")
    return float(average_precision_score(truth, scores))


def aligned_shd(
    adj_est: Tensor,
    adj_true: Tensor,
    *,
    threshold: float = 1e-3,
    weighted: bool = False,
) -> float:
    """SHD/WSHD minimised over latent permutations.

    Latent-variable models recover their factors only up to a permutation, so a
    raw comparison is meaningless. The exact minimum is found by exhaustive
    enumeration for ``d <= 8``; for larger dimensions a deterministic
    swap-based hill climb is used instead (exactness is traded for tractability
    and the routine is documented accordingly). ``weighted=True`` uses
    :func:`weighted_shd`. When the true latents are available, prefer aligning
    once with :func:`latent_permutation`, which is the convention used by the
    source methods and is also valid for ``d > 8``.
    """
    if adj_est.shape != adj_true.shape:
        raise ValueError("Adjacency matrices must share a shape.")
    adj_true = adj_true.to(adj_est.device)
    dim = int(adj_est.shape[0])

    def score(perm: tuple[int, ...]) -> float:
        index = torch.tensor(perm, device=adj_est.device)
        permuted = adj_est[index][:, index]
        return (
            weighted_shd(permuted, adj_true, threshold=threshold)
            if weighted
            else float(shd(permuted, adj_true, threshold=threshold))
        )

    if dim <= 8:
        return min(score(perm) for perm in permutations(range(dim)))

    from scipy.optimize import quadratic_assignment

    est_np = adj_est.abs().sum(-1).detach().cpu().numpy()
    true_np = adj_true.abs().sum(-1).detach().cpu().numpy()
    result = quadratic_assignment(
        true_np, est_np, method="faq", options={"maximize": True, "rng": 0}
    )
    col = np.asarray(result.col_ind, dtype=int)
    candidates = [
        tuple(int(v) for v in col),
        tuple(int(v) for v in np.argsort(col)),
        tuple(range(dim)),
    ]
    best_perm = min(candidates, key=score)
    best = score(best_perm)
    improved = True
    while improved:
        improved = False
        for i in range(dim):
            for j in range(i + 1, dim):
                candidate = list(best_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                value = score(tuple(candidate))
                if value < best - 1e-12:
                    best, best_perm, improved = value, tuple(candidate), True
    return best


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
