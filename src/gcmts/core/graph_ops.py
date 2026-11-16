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
from sklearn.linear_model import lars_path
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
    "estimate_instantaneous_adjacency",
    "estimate_latent_structure",
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

    A reversed edge (``i -> j`` in the estimate where the truth has
    ``j -> i``) is one structural edit, the convention of Tsamardinos et al.
    (2006), not two.  For a lagged adjacency ``(d, d, p)`` reversals are
    counted within each lag slice.  Adjacency matrices must share a shape.
    """
    if adj_est.shape != adj_true.shape:
        raise ValueError("Adjacency matrices must share a shape.")
    est = _binarise(adj_est, threshold)
    true = _binarise(adj_true.to(est.device), threshold)
    if est.ndim == 2:
        est = est.unsqueeze(-1)
        true = true.unsqueeze(-1)
    total = 0
    for lag in range(est.shape[-1]):
        e = est[..., lag]
        t = true[..., lag]
        differing = int((e != t).sum().item())
        reversed_edges = int(((e == 1) & (t == 0) & (e.T == 0) & (t.T == 1)).sum().item())
        total += differing - reversed_edges
    return total


def weighted_shd(
    adj_est: Tensor,
    adj_true: Tensor,
    *,
    threshold: float = 1e-3,
) -> float:
    """SHD variant that weights each structural difference by edge magnitude.

    A reversal is one weighted edit, matching :func:`shd`; the weight is the
    maximum of the two opposing edge magnitudes.
    """
    adj_true = adj_true.to(adj_est.device)
    est = _binarise(adj_est, threshold)
    true = _binarise(adj_true, threshold)
    weight = torch.maximum(adj_est.abs(), adj_true.abs())
    if est.ndim == 2:
        est = est.unsqueeze(-1)
        true = true.unsqueeze(-1)
        weight = weight.unsqueeze(-1)
    total = 0.0
    for lag in range(est.shape[-1]):
        e, t, w = est[..., lag], true[..., lag], weight[..., lag]
        value = float(((e != t).float() * w).sum().item())
        reversal = (e == 1) & (t == 0) & (e.T == 0) & (t.T == 1)
        if bool(reversal.any()):
            # The two differing cells are weighted by the two opposing
            # magnitudes; keep one edit with the larger weight.
            value -= float(torch.minimum(w, w.T)[reversal].sum().item())
        total += value
    return total


def _standardise_columns(matrix: np.ndarray) -> np.ndarray:
    """Column-wise z-scoring; constant columns become exactly zero."""
    scale = matrix.std(axis=0, keepdims=True)
    dead = scale.ravel() < 1e-8
    out = (matrix - matrix.mean(axis=0, keepdims=True)) / np.where(dead, 1.0, scale)
    out[:, dead] = 0.0
    return out


def _cluster_robust_t(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OLS fit on ``support`` plus cluster-robust (trajectory-level) t-stats.

    Pooling ``T - p`` consecutive steps of a trajectory produces strongly
    dependent rows, so the textbook i.i.d. standard error understates the
    sampling uncertainty by roughly a factor ``sqrt(T - p)``. Clustering the
    sandwich estimator by trajectory restores honest standard errors without
    discarding the within-trajectory information used for the point estimate.
    """
    xs = x[:, support]
    beta, *_ = np.linalg.lstsq(xs, y, rcond=None)
    residual = y - xs @ beta
    xtx_inv = np.linalg.pinv(xs.T @ xs)
    order = np.argsort(groups, kind="stable")
    xs_sorted = xs[order]
    residual_sorted = residual[order]
    groups_sorted = groups[order]
    starts = np.flatnonzero(np.r_[True, groups_sorted[1:] != groups_sorted[:-1]])
    scores = np.add.reduceat(xs_sorted * residual_sorted[:, None], starts, axis=0)
    n_clusters = int(scores.shape[0])
    correction = n_clusters / max(n_clusters - 1, 1)
    covariance = correction * (xtx_inv @ (scores.T @ scores) @ xtx_inv)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_stat = np.where(
        standard_error > 0.0,
        beta / np.where(standard_error > 0.0, standard_error, 1.0),
        np.where(np.abs(beta) > 0.0, np.inf, 0.0),
    )
    return beta, t_stat


def _lars_bic_support(x: np.ndarray, y: np.ndarray, n_eff: int) -> np.ndarray:
    r"""Lasso/LARS path whose support minimises an effective-size BIC.

    ``criterion = n log(RSS / n) + df * (n / n_eff) log(n_eff)``.

    The pooled design has ``n = B (T - p)`` rows but only ``n_eff = B``
    independent trajectories. The usual BIC penalty ``df log(n)`` therefore
    treats ``T - p`` correlated rows as independent evidence and adds roughly
    ``T - p`` spurious variables per target; multiplying the penalty by
    ``n / n_eff`` restores the correct evidence scale. Along the LARS path this
    is asymptotically the same rule as keeping coefficients whose
    cluster-robust t-statistic exceeds ``sqrt(log n_eff)``.
    """
    n_samples = x.shape[0]
    n_eff = max(int(n_eff), 2)
    try:
        _, _, path = lars_path(x, y, method="lasso")
    except Exception:  # pragma: no cover - degenerate design (all-zero column)
        return np.zeros(x.shape[1])
    residual = y[:, None] - x @ path
    rss = np.einsum("ij,ij->j", residual, residual)
    degrees = (np.abs(path) > 0.0).sum(axis=0)
    criterion = n_samples * np.log(np.maximum(rss, 1e-300) / n_samples) + degrees * (
        n_samples / n_eff
    ) * np.log(n_eff)
    return np.asarray(path[:, int(np.argmin(criterion))], dtype=np.float64)


def _prune_support(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, coefficient: np.ndarray, tau: float
) -> np.ndarray:
    """Backward elimination on cluster-robust t-stats at threshold ``tau``.

    The LARS path is greedy and unstable when candidate parents are collinear
    (a one-step-lagged parent is nearly a linear function of its own parents).
    Re-fitting OLS on the selected support and dropping the least significant
    coefficient until every survivor clears ``tau`` gives a support that is
    self-consistent with the reported (debiased) coefficients.
    """
    support = list(np.flatnonzero(np.abs(coefficient) > 0.0))
    while support:
        beta, t_stat = _cluster_robust_t(x, y, groups, np.asarray(support))
        if np.all(np.abs(t_stat) > tau):
            break
        support.pop(int(np.argmin(np.abs(t_stat))))
    out = np.zeros(x.shape[1], dtype=np.float64)
    if support:
        beta, _ = _cluster_robust_t(x, y, groups, np.asarray(support))
        out[np.asarray(support)] = beta
    return out


def _select_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    t_threshold: float | None,
) -> np.ndarray:
    """Per-target sparse regression with a calibrated, data-driven support.

    Both the design and each target are standardised column-wise first, so the
    returned coefficients are scale-free and comparable across methods and
    datasets (a fixed magnitude threshold is not meaningful otherwise).
    """
    x = _standardise_columns(design)
    n_eff = max(int(np.unique(groups).size), 2)
    tau = float(np.sqrt(np.log(n_eff))) if t_threshold is None else float(t_threshold)
    coefficients = np.zeros((x.shape[1], target.shape[1]), dtype=np.float64)
    for i in range(target.shape[1]):
        y = target[:, i]
        scale = float(y.std())
        if not np.isfinite(scale) or scale < 1e-8:
            continue
        y = (y - y.mean()) / scale
        selected = _lars_bic_support(x, y, n_eff)
        coefficients[:, i] = _prune_support(x, y, groups, selected, tau)
    return coefficients


def _conditional_lagged_fit(
    z: Tensor,
    *,
    p: int,
    include_same_time: bool,
    standardize: bool,
    t_threshold: float | None,
    intervention_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, torch.dtype, torch.device]:
    r"""Structural conditional fit of the lagged and same-time latent graph.

    For each node ``i`` the mechanism is regressed on the same-time
    predecessors ``z_{j,t}, j < i`` (when ``include_same_time``) *and* every
    lagged latent ``z_{j,t-1-l}``. Conditioning on the same-time predecessors
    is what removes the omitted-variable bias that otherwise turns each
    instantaneous edge into spurious lagged edges: ``z_{j,t}`` is strongly
    predicted by the lagged latents, so a lagged-only regression for
    ``z_{i,t}`` soaks up the instantaneous parent through its lagged shadow.

    ``intervention_masks`` (optional, shape ``(B, T, k)``, ``k <= d``) marks
    steps at which target ``i < k`` was replaced by an intervention.  Those
    rows are dropped from target ``i``'s regression, because the intervention
    cuts the incoming mechanism edges and mixing autonomous and intervened
    steps attenuates every coefficient.

    Returns NumPy blocks ``(lag[d, d, p], inst[d, d])`` in float64.
    """
    device, dtype = z.device, z.dtype
    values = z.detach().cpu().to(torch.float64).numpy()
    batch, time, dim = values.shape
    if standardize:
        scale = values.std(axis=(0, 1), keepdims=True)
        values = (values - values.mean(axis=(0, 1), keepdims=True)) / np.where(
            scale < 1e-8, 1.0, scale
        )
        values[:, :, scale.ravel() < 1e-8] = 0.0
    current = values[:, p:]  # (B, T-p, d)
    lagged = np.stack(
        [values[:, p - 1 - lag : time - 1 - lag, :] for lag in range(p)], axis=-1
    )  # (B, T-p, d, p)
    lagged_flat = lagged.reshape(batch, time - p, dim * p)
    groups = np.repeat(np.arange(batch), time - p)
    masks = None
    if intervention_masks is not None:
        masks = np.asarray(intervention_masks, dtype=bool)[:, p:]
        if masks.shape[:2] != (batch, time - p):
            raise ValueError("intervention_masks must have shape (B, T, k) with k <= d.")
        masks = masks.reshape(batch * (time - p), -1)

    lag_coefficients = np.zeros((dim, dim, p), dtype=np.float64)
    inst_coefficients = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        parts = [current[:, :, :i]] if include_same_time else []
        parts.append(lagged_flat)
        wide = np.concatenate(parts, axis=-1).reshape(-1, sum(a.shape[-1] for a in parts))
        target = current[:, :, i].reshape(-1)
        rows: slice | np.ndarray = slice(None)
        if masks is not None and i < masks.shape[1]:
            rows = ~masks[:, i]
        coefficient = _select_coefficients(
            wide[rows], target[rows, None], groups[rows], t_threshold
        )[:, 0]
        offset = i if include_same_time else 0
        if include_same_time:
            inst_coefficients[i, :i] = coefficient[:i]
        lag_coefficients[i] = coefficient[offset:].reshape(dim, p)
    return lag_coefficients, inst_coefficients, dtype, device


def _apply_magnitude_cutoff(adj: np.ndarray, threshold: float) -> np.ndarray:
    if threshold > 0.0:
        return np.where(np.abs(adj) > threshold, adj, 0.0)
    return adj


def estimate_latent_structure(
    z: Tensor,
    *,
    max_lag: int = 1,
    threshold: float = 0.0,
    standardize: bool = True,
    t_threshold: float | None = None,
    interventions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Joint post-hoc recovery of the lagged *and* same-time latent graph.

    Each latent mechanism is estimated once, conditional on the same-time
    predecessors ``z_{j,t}, j < i`` (the causal order supplied by
    :func:`latent_permutation`) and all lagged latents ``z_{j,t-1-l}``. The
    support is selected by an effective-sample-size BIC on the LARS path and
    then pruned by cluster-robust (trajectory-level) t-statistics, so the
    reported edges are exactly zero unless they are statistically supported.
    See :func:`estimate_latent_adjacency` and
    :func:`estimate_instantaneous_adjacency` for the two blocks.

    Args:
        z: latents of shape ``(B, T, d)``, already aligned to the causal order.
        max_lag: temporal lag ``p``.
        threshold: optional extra magnitude cutoff after selection (unneeded by
            default: off-support entries are exactly zero).
        standardize: center/scale each latent across batch and time first.
        t_threshold: cluster-robust t-statistic cutoff. Defaults to
            ``sqrt(log B)``, the effective-sample-size BIC threshold.
        interventions: optional binary mask ``(B, T, k)`` (``k <= d``) marking
            the target replaced by an intervention at each step.  It is used
            for the *same-time* block only: an intervened target is drawn
            independently of its parents, so pooling autonomous and
            intervened steps attenuates the instantaneous coefficients.  The
            lagged block is estimated on all rows, where masking reduced
            power more than it removed bias in validation.

    Returns:
        ``(lagged, instantaneous)`` with shapes ``(d, d, p)`` and ``(d, d)``.
        ``lagged[i, j, l]`` is the coefficient of ``z_{j,t-1-l}`` and
        ``instantaneous[i, j]`` that of ``z_{j,t}`` in the mechanism for
        ``z_{i,t}``.
    """
    if z.ndim != 3:
        raise ValueError("estimate_latent_structure expects z of shape (B, T, d).")
    p = max(1, int(max_lag))
    batch, time, dim = z.shape
    if time <= p:
        zeros_lag = torch.zeros(dim, dim, p, dtype=z.dtype, device=z.device)
        zeros_inst = torch.zeros(dim, dim, dtype=z.dtype, device=z.device)
        return zeros_lag, zeros_inst
    lag, inst, dtype, device = _conditional_lagged_fit(
        z,
        p=p,
        include_same_time=True,
        standardize=standardize,
        t_threshold=t_threshold,
    )
    if interventions is not None:
        _, inst, _, _ = _conditional_lagged_fit(
            z,
            p=p,
            include_same_time=True,
            standardize=standardize,
            t_threshold=t_threshold,
            intervention_masks=interventions.detach().cpu().numpy(),
        )
    # The support estimate is a NumPy regression; the result is returned on the
    # CPU in the input dtype (callers move it to their device).
    lag_t = torch.from_numpy(_apply_magnitude_cutoff(lag, threshold)).to(dtype=dtype)
    inst_t = torch.from_numpy(_apply_magnitude_cutoff(inst, threshold)).to(dtype=dtype)
    return lag_t, inst_t


def estimate_latent_adjacency(
    z: Tensor,
    *,
    max_lag: int = 1,
    threshold: float = 0.0,
    standardize: bool = True,
    include_same_time: bool = False,
    t_threshold: float | None = None,
) -> Tensor:
    r"""Recover a sparse lagged latent graph from trajectories.

    Post-hoc protocol of TDRL and LEAP (LassoNet on the learned latents): each
    target latent ``z_{i,t}`` is regressed on the lagged latents
    ``{z_{j,t-1-l}}`` after standardisation, the support is selected with an
    effective-sample-size BIC on the LARS path, and the coefficients are
    debiased by ordinary least squares on that support.

    Two corrections over the textbook LassoLars-BIC protocol matter here:

    * the BIC penalty is evaluated at the number of independent trajectories
      ``B`` rather than the pooled row count ``B (T - p)``, because
      consecutive steps are strongly dependent. Without it the path accepts
      roughly ``T - p`` spurious parents per target, which is why a fixed
      magnitude cutoff was needed to remove them (and why SHD was sensitive to
      that cutoff);
    * the surviving support is re-tested with cluster-robust t-statistics, so
      off-support entries are exactly zero and ``threshold`` is only an
      optional extra guard.

    The result is an ``(d, d, p)`` tensor where ``A[i, j, l] != 0`` means
    ``z_{i,t}`` depends on ``z_{j,t-1-l}``.

    Args:
        z: latents of shape ``(B, T, d)``.
        max_lag: temporal lag ``p``.
        threshold: optional extra magnitude cutoff applied after debiasing.
        standardize: center/scale each latent across batch and time first.
        include_same_time: condition each mechanism on ``z_{j,t}, j < i`` as
            well. Use this when the generator has instantaneous edges: a
            lagged-only regression then suffers omitted-variable bias. Prefer
            :func:`estimate_latent_structure`, which returns both blocks from
            the same fit.
        t_threshold: cluster-robust t cutoff; defaults to ``sqrt(log B)``.
    """
    if z.ndim != 3:
        raise ValueError("estimate_latent_adjacency expects z of shape (B, T, d).")
    p = max(1, int(max_lag))
    batch, time, dim = z.shape
    if time <= p:
        return torch.zeros(dim, dim, p, dtype=z.dtype, device=z.device)
    lag, _, dtype, device = _conditional_lagged_fit(
        z,
        p=p,
        include_same_time=include_same_time,
        standardize=standardize,
        t_threshold=t_threshold,
    )
    return torch.from_numpy(_apply_magnitude_cutoff(lag, threshold)).to(dtype=dtype)


def estimate_instantaneous_adjacency(
    z: Tensor,
    *,
    max_lag: int = 1,
    threshold: float = 0.0,
    standardize: bool = True,
    t_threshold: float | None = None,
    interventions: Tensor | None = None,
) -> Tensor:
    r"""Recover a same-time (instantaneous) latent graph from aligned latents.

    The latent factors are assumed to be in a known causal order (the module
    calls this *after* :func:`latent_permutation`, which aligns the learned
    coordinates to the true factors).  For each node ``i`` we regress
    ``z_{i,t}`` on the same-time values of the lower-index nodes
    ``z_{j,t}, j < i`` together with all lagged latents.  Restricting the
    same-time parents to ``j < i`` enforces acyclicity and matches the
    lower-triangular generators used by the interventional benchmarks.

    Returns an ``(d, d)`` tensor whose entry ``A[i, j]`` is the debiased
    coefficient of ``z_{j,t}`` in the mechanism for ``z_{i,t}`` (zero for
    ``j >= i``).  This is the post-hoc analogue of the lagged
    :func:`estimate_latent_adjacency`; it is a support estimate, not a proof of
    direction, because the causal order is supplied by the generator.

    Args:
        z: latents of shape ``(B, T, d)``, aligned to the causal order.
        max_lag: temporal lag ``p``.
        threshold: optional extra magnitude cutoff after selection.
        standardize: center/scale each latent across batch and time first.
        t_threshold: cluster-robust t cutoff; defaults to ``sqrt(log B)``.
        interventions: optional binary mask ``(B, T, k)``, ``k <= d``, marking
            intervened targets. Rows where target ``i`` was intervened are
            dropped from ``i``'s regression.
    """
    if z.ndim != 3:
        raise ValueError("estimate_instantaneous_adjacency expects z of shape (B, T, d).")
    p = max(1, int(max_lag))
    batch, time, dim = z.shape
    if time <= p:
        return torch.zeros(dim, dim, dtype=z.dtype, device=z.device)
    _, inst, dtype, device = _conditional_lagged_fit(
        z,
        p=p,
        include_same_time=True,
        standardize=standardize,
        t_threshold=t_threshold,
        intervention_masks=(
            interventions.detach().cpu().numpy() if interventions is not None else None
        ),
    )
    return torch.from_numpy(_apply_magnitude_cutoff(inst, threshold)).to(dtype=dtype)


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
