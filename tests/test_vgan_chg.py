"""Tests for the compact VGAN-CHG estimator (branch II.C surrogate)."""

from __future__ import annotations

import torch

from gcmts.effect_estimation.methods.vgan_chg import VGANCHG
from gcmts.typing import Batch


def _tiny_batch() -> Batch:
    """B=8, T=4, D=6; treatment in context ``A`` and outcome in the last channel."""
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(8, 4, 6, generator=generator)
    action = (torch.rand(8, 4, 1, generator=generator) > 0.5).float()
    target = x.clone()
    masked = x.clone()
    masked[..., -1] = 0.0  # outcome channel withheld from the encoder input
    return Batch(x=masked, target=target, context={"A": action})


def _model() -> VGANCHG:
    return VGANCHG(
        observed_dim=6,
        treatment_dim=1,
        window=3,
        latent_global=8,
        latent_regional=4,
        hidden_dims=(32, 32),
        hidden_dim=16,
    )


def test_forward_loss_keys_and_gradients():
    batch = _tiny_batch()
    model = _model()
    outputs = model(batch)
    losses = model.loss(outputs, batch)

    assert outputs["recon"].shape == batch.x.shape
    assert outputs["y_cf"].shape == (8, 4, 1)
    for key in ("loss", "recon", "kl", "dr", "gan_d", "gan_g"):
        assert key in losses, f"missing loss key {key}"
        assert torch.isfinite(losses[key]), f"non-finite loss {key}"
    assert losses["kl"] >= 0

    losses["loss"].backward()
    for module in (
        model.global_encoder,
        model.regional_encoder,
        model.decoder,
        model.propensity,
        model.hidden,
        model.generator,
        model.discriminator,
    ):
        assert any(p.grad is not None for p in module.parameters()), (
            f"no gradient populated for {type(module).__name__}"
        )


def test_counterfactual_shape_and_treatment_flip():
    batch = _tiny_batch()
    model = _model()
    model.eval()
    action = batch.context["A"]
    assert action is not None

    with torch.no_grad():
        torch.manual_seed(0)
        flipped = model.counterfactual(batch, {"A": 1.0 - action})
        torch.manual_seed(0)
        same = model.counterfactual(batch, {"A": action})

    assert flipped.counterfactual.shape == batch.x.shape
    assert flipped.factual.shape == batch.x.shape
    assert flipped.noise is not None
    for key in ("z_global", "z_regional", "y_hat_f", "propensity_logit"):
        assert key in flipped.noise, f"missing abduction key {key}"

    # Flipping the treatment must change the generated outcome channel.
    assert not torch.allclose(flipped.counterfactual[..., -1], same.counterfactual[..., -1])
    assert (flipped.counterfactual[..., -1] - same.counterfactual[..., -1]).abs().max() > 1e-6


def test_target_falls_back_to_last_channel_and_extreme_propensity_is_clipped():
    batch = _tiny_batch()
    assert batch.target is not None
    full = Batch(x=batch.target, context=batch.context)
    model = _model()
    outputs = model(full)
    assert torch.equal(outputs["target"], full.x)

    # All-treated batch: the propensity must be clipped, keeping DR finite.
    extreme = Batch(
        x=batch.target,
        target=batch.target,
        context={"A": torch.ones(8, 4, 1)},
    )
    losses = model.loss(model(extreme), extreme)
    assert torch.isfinite(losses["dr"])
    assert torch.isfinite(losses["loss"])
