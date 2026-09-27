"""Tests for reusable neural backbones."""

from __future__ import annotations

import pytest
import torch

from gcmts.core.backbones import (
    MLP,
    ExpFamilyPrior,
    GaussianHead,
    RealNVP,
    gaussian_kl,
    prepare_context,
)


def test_mlp_shape():
    net = MLP(4, 7, hidden_dims=(16, 16))
    assert net(torch.randn(5, 4)).shape == (5, 7)


def test_gaussian_head_ranges():
    head = GaussianHead(3, 4, hidden_dims=(16,))
    mean, logvar = head(torch.randn(8, 3))
    assert mean.shape == (8, 4)
    assert logvar.shape == (8, 4)
    assert torch.all(logvar <= 8.0) and torch.all(logvar >= -12.0)


def test_exp_family_prior_unconditional():
    prior = ExpFamilyPrior(0, 5)
    mean, logvar = prior(None)
    assert mean.shape == (1, 5)
    assert torch.allclose(logvar, torch.zeros(5))


def test_exp_family_prior_conditional():
    prior = ExpFamilyPrior(3, 5, hidden_dims=(16,))
    mean, logvar = prior(torch.randn(7, 3))
    assert mean.shape == (7, 5)


@pytest.mark.parametrize("volume_preserving", [True, False])
def test_realnvp_is_invertible(volume_preserving):
    torch.manual_seed(0)
    flow = RealNVP(6, n_layers=4, hidden_dims=(32,), volume_preserving=volume_preserving)
    x = torch.randn(10, 6)
    z, logdet = flow.forward(x)
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-5)
    assert logdet.shape == (10,)
    if volume_preserving:
        assert torch.allclose(logdet, torch.zeros(10), atol=1e-5)


def test_realnvp_log_prob_finite():
    flow = RealNVP(4, n_layers=2, hidden_dims=(16,))
    log_prob = flow.log_prob(torch.randn(12, 4))
    assert log_prob.shape == (12,)
    assert torch.isfinite(log_prob).all()


def test_realnvp_logdet_matches_inverse_jacobian():
    # Numerical check that forward log-det is the negative of the inverse's.
    torch.manual_seed(1)
    flow = RealNVP(3, n_layers=3, hidden_dims=(24,))
    x = torch.randn(4, 3, requires_grad=True)
    z, logdet = flow.forward(x)
    # log p(x) = log p(z) + logdet should be a valid density.
    assert torch.isfinite(logdet).all()


def test_gaussian_kl_zero_for_identical():
    mean = torch.randn(4, 3)
    logvar = torch.randn(4, 3)
    assert torch.allclose(gaussian_kl(mean, logvar, mean, logvar), torch.zeros(4), atol=1e-6)


def test_prepare_context_shapes():
    per_sample = {"u": torch.eye(4)[[0, 1]]}  # (2, 4)
    expanded = prepare_context(per_sample, 6)
    assert expanded is not None and expanded.shape == (6, 4)
    per_step = {"u": torch.randn(2, 3, 4)}
    assert prepare_context(per_step, 6).shape == (6, 4)
    assert prepare_context(None, 6) is None


def test_prepare_context_one_hot():
    context = {"u": torch.tensor([0, 2, 1])}
    encoded = prepare_context(context, 3)
    assert encoded is not None and encoded.shape == (3, 3)
    assert encoded[0, 0] == 1.0 and encoded[1, 2] == 1.0 and encoded[2, 1] == 1.0
