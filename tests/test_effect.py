"""Tests for counterfactual / treatment-effect generators and methods."""

from __future__ import annotations

import torch

from gcmts.data.synthetic_effect import HarmonicOscillatorGenerator
from gcmts.effect_estimation.methods import CEPAE, CRN, GANITE, CaTSG
from gcmts.evaluation.metrics import CFMAEMetric, MBEMetric, MMD2Metric


def test_harmonic_counterfactual_is_exact_and_heterogeneous():
    generator = HarmonicOscillatorGenerator(n_units=2, observed_dim=4, horizon=10, seed=0)
    batch = generator.sample(64)
    assert batch.counterfactual is not None
    assert batch.counterfactual.shape == batch.x.shape
    assert (batch.counterfactual - batch.x).abs().mean() > 1e-3
    outcomes = generator.potential_outcomes(256)
    assert (outcomes["y1"] - outcomes["y0"]).std() > 1e-3


def test_effect_methods_forward_loss_and_counterfactual():
    generator = HarmonicOscillatorGenerator(n_units=2, observed_dim=4, horizon=8, seed=0)
    batch = generator.sample(8)
    cases = [
        (CEPAE, {"observed_dim": 4, "treatment_dim": 1, "latent_dim": 3}),
        (GANITE, {"observed_dim": 4, "treatment_dim": 1, "noise_dim": 3}),
        (CRN, {"observed_dim": 4, "treatment_dim": 1, "latent_dim": 4, "horizon": 8}),
        (CaTSG, {"observed_dim": 4, "treatment_dim": 1, "horizon": 8, "n_steps": 4}),
    ]
    for cls, kwargs in cases:
        model = cls(**kwargs)
        losses = model.loss(model.forward(batch), batch)
        losses["loss"].backward()
        assert torch.isfinite(losses["loss"])
        cf = model.counterfactual(batch, {"A": torch.zeros(8, 1)})
        assert cf.counterfactual.shape == batch.x.shape


def test_counterfactual_metrics():
    target = torch.zeros(32, 5, 3)
    pred = target + 0.1
    assert abs(MBEMetric()(pred, target).value - 0.1) < 1e-6
    assert CFMAEMetric()(pred, target).value > 0
    assert MMD2Metric()(pred, target).value >= 0
