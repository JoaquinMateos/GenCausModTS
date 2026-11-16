"""Tests for the compact CaDRe (branch I.E) implementation."""

from __future__ import annotations

import math

import torch

from gcmts.causal_representation_learning.methods.cadre import CaDRe
from gcmts.core import SimpleTrainer
from gcmts.data.latent_scm import LatentStructuralGenerator
from gcmts.typing import Batch


def _generator(horizon: int = 16, seed: int = 7) -> LatentStructuralGenerator:
    return LatentStructuralGenerator(
        latent_dim=3,
        observed_dim=6,
        horizon=horizon,
        n_regimes=4,
        mixing="tanh",
        seed=seed,
    )


def _model() -> CaDRe:
    return CaDRe(
        observed_dim=6,
        latent_dim=3,
        u_dim=4,
        hidden_dims=(16,),
        s_hidden=(16,),
        decoder_hidden=(16,),
        prior_hidden=(16,),
    )


def test_generator_shapes_determinism_and_spectral_radius():
    first = _generator()
    second = _generator()
    batch_a = first.sample(8)
    batch_b = second.sample(8)

    assert batch_a.x.shape == (8, 16, 6)
    assert batch_a.z is not None and batch_a.z.shape == (8, 16, 3)
    assert batch_a.adjacency is not None and batch_a.adjacency.shape == (3, 3, 1)
    assert batch_a.context is not None
    assert batch_a.context["u"].shape == (8, 4)
    assert torch.allclose(batch_a.context["u"].sum(-1), torch.ones(8))
    assert batch_a.context["instantaneous_adjacency"].shape == (3, 3)
    assert batch_a.context["observed_adjacency"].shape == (6, 6)

    torch.testing.assert_close(batch_a.x, batch_b.x)
    torch.testing.assert_close(batch_a.z, batch_b.z)

    instant = batch_a.context["instantaneous_adjacency"]
    observed = batch_a.context["observed_adjacency"]
    # Both causal graphs are strictly lower triangular in a known order.
    assert torch.triu(instant).abs().max() == 0
    assert torch.triu(observed).abs().max() == 0
    assert torch.equal(batch_a.adjacency, first.latent_lag_adjacency.unsqueeze(-1))

    radius = float(torch.linalg.eigvals(first.latent_lag_adjacency).abs().max())
    assert math.isfinite(radius)
    assert radius <= first.max_radius + 1e-6
    assert torch.isfinite(batch_a.x).all()


def test_model_forward_loss_backward_and_graph_shapes():
    generator = _generator(horizon=8)
    batch = generator.sample(4)
    model = _model()
    outputs = model(batch)

    assert outputs.z.shape == (4, 8, 3)
    assert outputs.x_recon is not None and outputs.x_recon.shape == (4, 8, 6)
    assert outputs.adjacency is not None and outputs.adjacency.shape == (3, 3, 1)

    losses = model.loss(outputs, batch)
    for name in ("loss", "recon", "kl_z", "kl_s", "sparsity", "acyclicity"):
        assert name in losses
        assert torch.isfinite(losses[name])
    losses["loss"].backward()
    assert model.z_encoder.net.net[0].weight.grad is not None
    assert model.z_prior_nets[0].net[0].weight.grad is not None

    latent_adjacency = model.get_latent_adjacency()
    instant = model.get_instantaneous_adjacency()
    observed = model.get_observed_adjacency()
    sensitivity = model.get_module_sensitivity()
    assert latent_adjacency is not None and latent_adjacency.shape == (3, 3, 1)
    assert instant.shape == (3, 3)
    assert observed.shape == (6, 6)
    assert sensitivity.shape == (6, 3)
    assert torch.isfinite(latent_adjacency).all()
    assert torch.isfinite(instant).all()
    assert torch.isfinite(observed).all()
    assert torch.isfinite(sensitivity).all()
    # The residual prior has a triangular same-time Jacobian and the observed
    # graph has self-loops masked.
    assert torch.triu(instant).abs().max() == 0
    assert observed.diag().abs().max() == 0


def test_model_transition_and_decode():
    generator = _generator(horizon=8)
    batch = generator.sample(4)
    model = _model()
    outputs = model(batch)

    latent = model.encode(batch.x, batch.context)
    assert latent.shape == (4, 8, 3)
    decoded = model.decode(outputs.z)
    assert decoded.shape == (4, 8, 6)
    next_latent = model.transition(batch.z[:, -1:], batch.context)
    assert next_latent.shape == (4, 3)
    assert torch.isfinite(next_latent).all()


def test_cadre_two_epoch_cpu_smoke_fit():
    generator = _generator(horizon=8)
    model = _model()
    trainer = SimpleTrainer(
        max_epochs=2,
        lr=1e-3,
        batch_size=4,
        steps_per_epoch=2,
        device="cpu",
        log_every=0,
        seed=0,
        prefetch=False,
    )
    history = trainer.fit(model, generator)
    assert len(history["loss"]) == 2
    assert all(math.isfinite(value) for value in history["loss"])


# --- conditional-context contract -------------------------------------------


def test_unconditional_model_ignores_the_batch_context():
    """``u_dim=0`` builds an unconditional model even when batches carry ``u``.

    Benchmark generators always attach a regime context to the batch, so the
    unconditional path must drop it rather than append a tensor the encoder was
    not sized for.
    """
    generator = _generator()
    batch = generator.sample(6)
    assert batch.context is not None and "u" in batch.context

    model = CaDRe(
        observed_dim=6,
        latent_dim=3,
        u_dim=0,
        hidden_dims=(16,),
        s_hidden=(16,),
        decoder_hidden=(16,),
        prior_hidden=(16,),
    )
    outputs = model(batch)
    assert outputs.z.shape == (6, 16, 3)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])

    with torch.no_grad():
        with_context = model.encode(batch.x, batch.context)
        without_context = model.encode(batch.x, None)
        torch.testing.assert_close(with_context, without_context)
        z_next = model.transition(batch.z[:, -2:], batch.context)
    assert z_next.shape == (6, 3)
    assert torch.isfinite(z_next).all()


def test_context_width_mismatch_is_reported():
    """A context of the wrong width fails with an actionable message."""
    generator = _generator()
    batch = generator.sample(4)
    model = CaDRe(
        observed_dim=6,
        latent_dim=3,
        u_dim=3,  # generator emits a 4-wide one-hot
        hidden_dims=(16,),
        s_hidden=(16,),
        decoder_hidden=(16,),
        prior_hidden=(16,),
    )
    try:
        model(batch)
    except ValueError as error:
        assert "width 4" in str(error) and "u_dim=3" in str(error)
    else:  # pragma: no cover - the shape error must be raised, not swallowed
        raise AssertionError("expected a ValueError for a misdeclared context width")


# --- separation metric floor on the control suites ---------------------------


def test_cartpole_true_latents_are_not_separable_under_r2_sep():
    """On CartPole, ``R^2_sep`` has a floor near 0.7 even for a perfect model.

    The four physical state variables are mutually predictable (x from x-dot,
    theta-dot from theta, ...), so feeding the *true* latents back through the
    benchmark's leakage metric already scores ``R^2_sep ~= 0.70`` at
    ``R^2_diag = 1``.  A reported ``R^2_sep`` of that size on this suite is the
    metric's floor, not evidence that a representation leaks across factors.
    """
    from gcmts.data.synthetic import CartPoleGenerator
    from gcmts.evaluation.metrics import R2DiagMetric, R2SepMetric

    batch = CartPoleGenerator(horizon=16, observed_dim=4, n_regimes=3, seed=0).sample(256)
    assert batch.z is not None
    diagonal = R2DiagMetric()(batch.z, batch.z).value
    separation = R2SepMetric()(batch.z, batch.z).value
    assert diagonal > 0.95
    assert separation > 0.6, (
        "the CartPole metric floor changed; re-check how R2_sep is read on the control suites"
    )


def test_conditional_model_requires_a_context():
    """A conditional model cannot be evaluated without the regime it was built for."""
    generator = _generator()
    batch = generator.sample(4)
    model = CaDRe(
        observed_dim=6,
        latent_dim=3,
        u_dim=4,
        hidden_dims=(16,),
        s_hidden=(16,),
        decoder_hidden=(16,),
        prior_hidden=(16,),
    )
    for call in (
        lambda: model.encode(batch.x, None),
        lambda: model(Batch(x=batch.x, z=batch.z)),
        lambda: model.transition(batch.z[:, -1:], None),
    ):
        try:
            call()
        except ValueError as error:
            assert "u_dim=4" in str(error)
        else:  # pragma: no cover - the missing-context path must be reported
            raise AssertionError("expected a ValueError for a missing context")
