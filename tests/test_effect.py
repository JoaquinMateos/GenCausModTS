"""Tests for counterfactual / treatment-effect generators and methods."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gcmts.core import SimpleTrainer
from gcmts.core.trainer import training_view
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
    # Losses are computed on the factual training view: the trainer strips the
    # ground-truth counterfactual from every training/validation batch.
    train_batch = training_view(batch)
    assert train_batch.counterfactual is None
    cases = [
        (CEPAE, {"observed_dim": 4, "treatment_dim": 1, "latent_dim": 3}),
        (GANITE, {"observed_dim": 4, "treatment_dim": 1, "noise_dim": 3}),
        (CRN, {"observed_dim": 4, "treatment_dim": 1, "latent_dim": 4, "horizon": 8}),
        (CaTSG, {"observed_dim": 4, "treatment_dim": 1, "horizon": 8, "n_steps": 4}),
    ]
    for cls, kwargs in cases:
        model = cls(**kwargs)
        losses = model.loss(model.forward(train_batch), train_batch)
        losses["loss"].backward()
        assert torch.isfinite(losses["loss"])
        cf = model.counterfactual(batch, {"A": torch.zeros(8, 1)})
        assert cf.counterfactual.shape == batch.x.shape


def test_ganite_rejects_counterfactual_supervision():
    """C1: the training loss must never read the ground-truth counterfactual."""
    generator = HarmonicOscillatorGenerator(n_units=2, observed_dim=4, horizon=6, seed=0)
    batch = generator.sample(8)
    assert batch.counterfactual is not None
    model = GANITE(observed_dim=4, treatment_dim=1, noise_dim=3)
    with pytest.raises(AssertionError, match="counterfactual"):
        model.forward(batch)
    factual = replace(batch, counterfactual=None)
    with pytest.raises(AssertionError, match="counterfactual"):
        model.loss(model.forward(factual), batch)


def test_ganite_ite_term_is_factual_only_and_finite():
    generator = HarmonicOscillatorGenerator(n_units=2, observed_dim=4, horizon=6, seed=0)
    batch = training_view(generator.sample(8))
    model = GANITE(observed_dim=4, treatment_dim=1, noise_dim=3)
    torch.manual_seed(0)
    outputs = model.forward(batch)
    torch.manual_seed(0)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["i_supervised"])
    # The supervised I-block term reconstructs the observed factual outcome.
    assert losses["i_supervised"] >= 0.0


def test_simple_trainer_strips_counterfactuals_from_training_batches():
    generator = HarmonicOscillatorGenerator(n_units=1, observed_dim=2, horizon=4, seed=0)
    seen: list[bool] = []
    model = GANITE(observed_dim=2, treatment_dim=1, noise_dim=2, hidden_dims=(16, 16))

    original_forward = model.forward

    def recording_forward(batch):  # noqa: ANN001, ANN202
        seen.append(batch.counterfactual is not None)
        return original_forward(batch)

    model.forward = recording_forward  # type: ignore[method-assign]
    trainer = SimpleTrainer(
        max_epochs=1, steps_per_epoch=2, batch_size=4, device="cpu", val_size=0,
        seed=0, scheduler="none",
    )
    history = trainer.fit(model, generator)
    assert seen and not any(seen)
    assert all(torch.isfinite(torch.tensor(v)) for v in history["loss"])


def test_counterfactual_metrics():
    target = torch.zeros(32, 5, 3)
    pred = target + 0.1
    assert abs(MBEMetric()(pred, target).value - 0.1) < 1e-6
    assert CFMAEMetric()(pred, target).value > 0
    assert MMD2Metric()(pred, target).value >= 0
