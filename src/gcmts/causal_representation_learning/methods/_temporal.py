"""Shared temporal helpers for sequential CRL methods."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["lagged_features", "temporal_kl", "hmm_log_marginal"]


def lagged_features(z: Tensor, max_lag: int) -> Tensor | None:
    """Return windows ``(B, T-p, p*d)`` of the ``p`` previous latent states.

    Returns ``None`` when the sequence is shorter than the lag.
    """
    batch, time, dim = z.shape
    p = max_lag
    if time <= p:
        return None
    windows = z.unfold(1, p, 1)[:, : time - p]  # (B, T-p, d, p)
    return windows.permute(0, 1, 3, 2).reshape(batch, time - p, p * dim)


def temporal_kl(
    q_mean: Tensor,
    q_logvar: Tensor,
    p_mean: Tensor,
    p_logvar: Tensor,
) -> Tensor:
    """KL(q||p) summed over time and latent dims, averaged over the batch."""
    from gcmts.core.backbones import gaussian_kl

    return gaussian_kl(
        q_mean.reshape(-1, q_mean.shape[-1]),
        q_logvar.reshape(-1, q_logvar.shape[-1]),
        p_mean.reshape(-1, p_mean.shape[-1]),
        p_logvar.reshape(-1, p_logvar.shape[-1]),
    ).reshape(q_mean.shape[0], q_mean.shape[1]).sum(dim=1).mean()


def hmm_log_marginal(
    log_emission: Tensor,
    log_transition: Tensor,
    log_initial: Tensor,
) -> Tensor:
    """Log marginal ``log p(z_1:T)`` of an HMM via the forward algorithm.

    Args:
        log_emission: ``(B, T, R)`` log emission probabilities per regime.
        log_transition: ``(R, R)`` log transition matrix ``p(c_t | c_{t-1})``.
        log_initial: ``(R,)`` log initial distribution.

    Returns:
        ``(B,)`` log marginal likelihoods.
    """
    batch, time, n_regimes = log_emission.shape
    alpha = log_initial.unsqueeze(0) + log_emission[:, 0]  # (B, R)
    for step in range(1, time):
        # (B, R_prev, 1) + (1, R_prev, R_next) -> logsumexp over prev
        alpha = torch.logsumexp(
            alpha.unsqueeze(-1) + log_transition.unsqueeze(0), dim=1
        ) + log_emission[:, step]
    return torch.logsumexp(alpha, dim=-1)
