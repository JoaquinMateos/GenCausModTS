"""Tests for the compact Counterfactual Generative Network (CNF)."""

from __future__ import annotations

import math

import torch

from gcmts.data.compositional_images import CompositionalImageGenerator
from gcmts.effect_estimation.methods.cnf import CNF
from gcmts.evaluation.metrics import CFMAEMetric

IMAGE_SIZE = 16
N_CLASSES = 6


def _generator(seed: int = 0) -> CompositionalImageGenerator:
    return CompositionalImageGenerator(
        image_size=IMAGE_SIZE, horizon=4, n_classes=N_CLASSES, seed=seed
    )


def _model(**kwargs: object) -> CNF:
    defaults: dict[str, object] = {
        "image_size": IMAGE_SIZE,
        "n_classes": N_CLASSES,
        "noise_dim": 4,
        "hidden_dims": (32, 32),
        "inversion_steps": 15,
        "inversion_lr": 0.1,
    }
    defaults.update(kwargs)
    return CNF(**defaults)  # type: ignore[arg-type]


def test_generator_shapes_determinism_and_range():
    batch = _generator(seed=0).sample(4)
    assert batch.x.shape == (4, 4, 3 * IMAGE_SIZE * IMAGE_SIZE)
    assert torch.isfinite(batch.x).all()
    assert float(batch.x.min()) >= 0.0
    assert float(batch.x.max()) <= 1.0

    clone = _generator(seed=0).sample(4)
    assert torch.allclose(batch.x, clone.x)
    assert torch.allclose(batch.counterfactual, clone.counterfactual)

    different = _generator(seed=1).sample(4)
    assert not torch.allclose(batch.x, different.x)

    short = _generator(seed=2).sample(3, horizon=2)
    assert short.x.shape == (3, 2, 3 * IMAGE_SIZE * IMAGE_SIZE)


def test_context_exposes_factors_and_paired_counterfactuals():
    batch = _generator(seed=3).sample(5)
    context = batch.context
    assert context is not None

    assert context["mask"].shape == (5, 4, IMAGE_SIZE, IMAGE_SIZE)
    assert context["foreground"].shape == (5, 4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert context["background"].shape == (5, 4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert float(context["mask"].min()) >= 0.0
    assert float(context["mask"].max()) <= 1.0

    for mechanism in ("shape", "texture", "background"):
        factual_label = context[f"{mechanism}_label"]
        swapped_label = context[f"cf_{mechanism}_label"]
        assert factual_label.shape == (5,)
        assert not torch.equal(factual_label, swapped_label)
        swapped_image = context[f"cf_{mechanism}"]
        assert swapped_image.shape == batch.x.shape
        assert (swapped_image - batch.x).abs().mean() > 1e-4

    assert torch.allclose(batch.counterfactual, context["cf_texture"])
    # The latent class is correlated with, but not identical to, the mechanisms.
    assert context["class"].shape == (5,)
    assert context["labels"].shape == (5, 3)


def test_forward_loss_and_backward_are_finite():
    torch.manual_seed(0)
    batch = _generator(seed=4).sample(4)
    model = _model()

    outputs = model.forward(batch)
    assert outputs["recon"].shape == batch.x.shape
    assert outputs["mask"].shape == (4, 4, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert outputs["foreground"].shape == (4, 4, 3, IMAGE_SIZE, IMAGE_SIZE)

    losses = model.loss(outputs, batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    for name in ("recon", "mask_entropy", "mask_mean", "texture"):
        assert torch.isfinite(losses[name]), name
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.causal_level() == "L3_counterfactual"


def test_counterfactual_swaps_texture_and_differs_from_factual():
    torch.manual_seed(1)
    batch = _generator(seed=5).sample(3)
    model = _model()
    context = batch.context
    assert context is not None

    target = (context["texture_label"] + 1) % N_CLASSES
    result = model.counterfactual(batch, {"texture": target})

    assert torch.equal(result.factual, batch.x)
    assert result.counterfactual.shape == batch.x.shape
    assert torch.isfinite(result.counterfactual).all()
    assert float(result.counterfactual.detach().min()) >= 0.0
    assert float(result.counterfactual.detach().max()) <= 1.0
    change = (result.counterfactual - result.factual).abs().mean()
    assert float(change.detach()) > 1e-4

    # The inferred noise is the same dict used by the counterfactual.
    assert result.noise is not None
    assert "noise_shape" in result.noise

    # The generic "A" shortcut maps to the texture mechanism.
    intervened = model.intervene(batch, {"A": target})
    assert intervened.shape == batch.x.shape

    metric = CFMAEMetric()(result.counterfactual, context["cf_texture"])
    assert math.isfinite(metric.value)


def test_abduction_reduces_reconstruction_error():
    torch.manual_seed(2)
    batch = _generator(seed=6).sample(2)
    model = _model(inversion_steps=20)
    noise = model.abduction(batch)

    trace = noise["trace"]
    assert trace.numel() == 21
    assert torch.isfinite(trace).all()
    assert float(trace[-1]) < float(trace[0])

    final_error = (noise["recon"] - batch.x).pow(2).mean()
    assert torch.allclose(final_error, trace[-1], atol=1e-5)
    assert noise["recon"].shape == batch.x.shape
