"""Tests for continuous-time solvers."""

from __future__ import annotations

import torch

from gcmts.core.solvers import bures_w2, euler_maruyama, matrix_sqrt


def test_euler_maruyama_shape_deterministic():
    def drift(z, _t):
        return -z

    def zero_diffusion(z, _t):
        return torch.zeros_like(z)

    z0 = torch.ones(5, 2)
    trajectory = euler_maruyama(drift, zero_diffusion, z0, n_steps=10, dt=0.01)
    assert trajectory.shape == (5, 11, 2)
    # Deterministic decay should shrink the state.
    assert trajectory[:, -1].abs().mean() < z0.abs().mean()


def test_euler_maruyama_reproducible():
    def drift(z, _t):
        return torch.zeros_like(z)

    def diffusion(z, _t):
        return torch.ones_like(z)

    gen = torch.Generator().manual_seed(0)
    a = euler_maruyama(drift, diffusion, torch.zeros(3, 2), 5, 0.1, generator=gen)
    gen = torch.Generator().manual_seed(0)
    b = euler_maruyama(drift, diffusion, torch.zeros(3, 2), 5, 0.1, generator=gen)
    assert torch.allclose(a, b)


def test_matrix_sqrt():
    matrix = torch.tensor([[4.0, 0.0], [0.0, 9.0]])
    root = matrix_sqrt(matrix)
    assert torch.allclose(root @ root, matrix, atol=1e-5)


def test_bures_w2_identical_is_zero():
    mean = torch.randn(4, 3)
    cov = torch.stack([torch.eye(3) for _ in range(4)])
    distance = bures_w2(mean, cov, mean, cov)
    assert torch.allclose(distance, torch.zeros(4), atol=1e-5)


def test_bures_w2_known_value():
    # 1D: W2^2(N(0,1), N(m, s^2)) = m^2 + (1 - s)^2.
    mean_a = torch.zeros(1, 1)
    cov_a = torch.ones(1, 1, 1)
    mean_b = torch.tensor([[2.0]])
    cov_b = torch.tensor([[[4.0]]])
    value = bures_w2(mean_a, cov_a, mean_b, cov_b).item()
    expected = 2.0**2 + (1.0 - 2.0) ** 2
    assert abs(value - expected) < 1e-5
