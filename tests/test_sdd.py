"""Tests for static-dynamic disentanglement generators and methods."""

from __future__ import annotations

import torch

from gcmts.data.synthetic_sdd import MultiDomainGenerator
from gcmts.evaluation.metrics import DomainAccuracyMetric, MIGMetric
from gcmts.static_dynamic_disentanglement.methods import DANN, ERM, SYNC


def test_multidomain_generator_shapes_and_domains():
    generator = MultiDomainGenerator(n_domains=5, observed_dim=8, horizon=10, seed=0)
    batch = generator.sample(16, domains=[0, 1])
    assert batch.x.shape == (16, 10, 8)
    assert batch.z is not None and batch.z.shape == (16, 10, 6)
    assert batch.domain is not None and set(batch.domain.tolist()) <= {0, 1}
    assert batch.outcome is not None and batch.outcome.shape == (16,)


def test_sdd_methods_forward_loss_and_prediction():
    generator = MultiDomainGenerator(n_domains=5, observed_dim=8, horizon=10, seed=0)
    batch = generator.sample(8)
    common = {
        "observed_dim": 8,
        "static_dim": 2,
        "dynamic_dim": 2,
        "spurious_static_dim": 1,
        "spurious_dynamic_dim": 1,
    }
    cases = [
        (SYNC, {**common, "n_domains": 5}),
        (DANN, {**common, "n_domains": 5}),
        (ERM, common),
    ]
    for cls, kwargs in cases:
        model = cls(**kwargs)
        losses = model.loss(model.forward(batch), batch)
        losses["loss"].backward()
        assert torch.isfinite(losses["loss"])
        assert model.predict_outcome(batch.x).shape == (8,)
        factors = model.disentangle(batch.x)
        assert factors.z_stc.shape[0] == 8 and factors.z_dyc.shape[0] == 8


def test_sdd_metrics():
    prediction = torch.tensor([0, 1, 2, 1, 0])
    target = torch.tensor([0, 1, 2, 0, 0])
    assert abs(DomainAccuracyMetric()(prediction, target).value - 0.8) < 1e-6
    latents = torch.randn(128, 4)
    factors = latents + 0.01 * torch.randn(128, 4)
    assert MIGMetric()(latents, factors).value > 0.0
