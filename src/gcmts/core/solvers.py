"""Numerical helpers for continuous-time (SDE) CRL methods.

Provides an Euler–Maruyama integrator and the Bures/Wasserstein-2 distance
between Gaussians, which CEGEN uses in closed form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

__all__ = ["euler_maruyama", "bures_w2", "gaussian_w2", "matrix_sqrt"]

Drift = Callable[[Tensor, float], Tensor]
Diffusion = Callable[[Tensor, float], Tensor]


def euler_maruyama(
    drift: Drift,
    diffusion: Diffusion,
    z0: Tensor,
    n_steps: int,
    dt: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Integrate ``dz = f(z, t) dt + g(z, t) dW`` with the Euler–Maruyama scheme.

    Args:
        drift: ``f(z, t) -> (B, d)``.
        diffusion: ``g(z, t) -> (B, d)`` (diagonal, added as ``g * dW``).
        z0: initial state ``(B, d)``.
        n_steps: number of Euler steps.
        dt: step size.
        generator: optional RNG for reproducibility.

    Returns:
        Trajectory ``(B, n_steps + 1, d)`` including ``z0``.
    """
    z = z0
    trajectory = [z0]
    sqrt_dt = dt**0.5
    for step in range(n_steps):
        t = step * dt
        noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        z = z + drift(z, t) * dt + diffusion(z, t) * sqrt_dt * noise
        trajectory.append(z)
    return torch.stack(trajectory, dim=1)


def matrix_sqrt(matrix: Tensor) -> Tensor:
    """Symmetric positive-semidefinite square root via eigendecomposition."""
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    eigenvalues = eigenvalues.clamp(min=0.0)
    root = (eigenvectors * eigenvalues.sqrt().unsqueeze(-2)) @ eigenvectors.transpose(
        -1, -2
    )
    return cast(Tensor, root)


def bures_w2(
    mean_a: Tensor,
    cov_a: Tensor,
    mean_b: Tensor,
    cov_b: Tensor,
) -> Tensor:
    r"""Squared 2-Wasserstein distance between Gaussians (Bures metric).

    ``W2^2 = ||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a^{1/2} S_b S_a^{1/2})^{1/2})``

    Inputs may be batched: ``mean_*`` of shape ``(B, d)`` and ``cov_*`` of
    shape ``(B, d, d)``. Returns ``(B,)``.
    """
    diff = ((mean_a - mean_b) ** 2).sum(dim=-1)
    sqrt_a = matrix_sqrt(cov_a)
    middle = sqrt_a @ cov_b @ sqrt_a
    trace = (cov_a + cov_b).diagonal(dim1=-2, dim2=-1).sum(-1) - 2.0 * matrix_sqrt(
        middle
    ).diagonal(dim1=-2, dim2=-1).sum(-1)
    return diff + trace.clamp(min=0.0)


def gaussian_w2(
    mean_a: Tensor,
    cov_a: Tensor,
    mean_b: Tensor,
    cov_b: Tensor,
) -> Tensor:
    """Alias for :func:`bures_w2` kept for methodological clarity."""
    return bures_w2(mean_a, cov_a, mean_b, cov_b)
