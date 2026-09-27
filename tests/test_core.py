"""Tests for core TSCM / base abstractions."""

from __future__ import annotations

import torch

from gcmts.core import LTSCM, TSCM, mcc
from gcmts.core.utils import hungarian_match, lagged_adjacency_to_dag


def test_tscm_shape():
    tscm = TSCM(
        observed_dim=3,
        max_lag=2,
        mechanisms=[lambda p, e, i=i: p[:, -1, i] + e[:, i] for i in range(3)],
    )
    x = tscm.sample(10, 20)
    assert x.shape == (10, 20, 3)


def test_ltscm_returns_latents():
    def mech(parents, eps, _ctx, i):
        return parents[:, -1, i] + eps[:, i]

    ltscm = LTSCM(
        latent_dim=2,
        observed_dim=4,
        max_lag=1,
        mechanisms=[lambda p, e, c, i=i: mech(p, e, c, i) for i in range(2)],
        decoder=lambda z, eta: z
        @ torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, -1.0]])
        + eta,
    )
    x, z = ltscm.sample(5, 10, return_latents=True)
    assert x.shape == (5, 10, 4)
    assert z.shape == (5, 10, 2)


def test_mcc_perfect_recovery():
    z = torch.randn(1000, 4)
    z_perm = z[:, [2, 0, 3, 1]]
    assert mcc(z, z_perm) > 0.99


def test_hungarian_match_shape():
    cost = torch.randn(5, 7)
    r, c = hungarian_match(cost)
    assert len(r) == len(c) == 5


def test_lagged_adjacency_to_dag():
    adj = torch.zeros(3, 3, 2)
    adj[0, 1, 0] = 1.0  # lag 1: z0 -> z1
    adj[2, 0, 1] = 1.0  # lag 2: z2 -> z0
    dag = lagged_adjacency_to_dag(adj)
    assert dag.shape == (9, 9)
    # lag 1 rows 0..2, present columns 6..8
    assert dag[0, 7].item() == 1.0
    # lag 2 rows 3..5, present columns 6..8
    assert dag[5, 6].item() == 1.0
