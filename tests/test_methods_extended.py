"""Tests for the extended CRL methods: TDRL, NCTRL, MOSAIC, CEGEN."""

from __future__ import annotations

import pytest
import torch

from gcmts.causal_representation_learning.methods import CEGEN, MOSAIC, NCTRL, TDRL
from gcmts.causal_representation_learning.methods._temporal import hmm_log_marginal
from gcmts.core import SimpleTrainer, move_batch
from gcmts.core.utils import mcc
from gcmts.data.synthetic import LinearSDEGenerator, TemporalNonlinearGenerator


def _temporal(n_regimes: int = 2, seed: int = 0) -> TemporalNonlinearGenerator:
    return TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=12, n_regimes=n_regimes, seed=seed
    )


# --- shared temporal helpers ------------------------------------------------

def test_hmm_log_marginal_shapes():
    batch, time, regimes = 5, 8, 3
    log_emission = torch.log_softmax(torch.randn(batch, time, regimes), dim=-1)
    log_transition = torch.log_softmax(torch.randn(regimes, regimes), dim=-1)
    log_initial = torch.log_softmax(torch.randn(regimes), dim=-1)
    result = hmm_log_marginal(log_emission, log_transition, log_initial)
    assert result.shape == (batch,)
    assert torch.isfinite(result).all()


# --- TDRL -------------------------------------------------------------------

def test_tdrl_forward_and_grad():
    generator = _temporal(n_regimes=2)
    batch = generator.sample(16)
    model = TDRL(
        observed_dim=6,
        latent_fix_dim=1,
        latent_dyn_dim=1,
        latent_obs_dim=1,
        u_dim=2,
    )
    outputs = model.forward(batch)
    assert outputs.z.shape == (16, 12, 3)
    assert outputs.x_recon is not None and outputs.x_recon.shape == batch.x.shape
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.dyn_transition.net.net[0].weight.grad is not None
    assert model.obs_prior is not None


def test_tdrl_training_reduces_loss():
    generator = _temporal(n_regimes=2)
    model = TDRL(
        observed_dim=6, latent_fix_dim=1, latent_dyn_dim=1, latent_obs_dim=1, u_dim=2
    )
    trainer = SimpleTrainer(
        max_epochs=30, lr=1e-3, batch_size=64, steps_per_epoch=10, log_every=0, seed=0
    )
    history = trainer.fit(model, generator)
    assert history["loss"][-1] < history["loss"][0]


# --- NCTRL ------------------------------------------------------------------

def test_nctrl_forward_and_grad():
    generator = _temporal(n_regimes=3, seed=1)
    batch = generator.sample(16)
    model = NCTRL(observed_dim=6, latent_dim=3, n_regimes=3)
    outputs = model.forward(batch)
    assert outputs.z.shape == (16, 12, 3)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.transition_nets[0].net[0].weight.grad is not None


def test_nctrl_regime_parameters_normalised():
    model = NCTRL(observed_dim=4, latent_dim=2, n_regimes=4)
    transition = torch.log_softmax(model.hmm_transition, dim=-1).exp()
    assert torch.allclose(transition.sum(dim=-1), torch.ones(4), atol=1e-5)


# --- MOSAIC -----------------------------------------------------------------

def test_mosaic_additive_decoder_forward():
    generator = _temporal(n_regimes=2)
    batch = generator.sample(16)
    model = MOSAIC(observed_dim=6, latent_dim=3, u_dim=2)
    outputs = model.forward(batch)
    assert outputs.z.shape == (16, 12, 3)
    assert outputs.extras is not None
    assert outputs.extras["importance"].shape == (6, 3)
    losses = model.loss(outputs, batch)
    assert {"loss", "recon", "kl", "sparsity"} <= set(losses)
    losses["loss"].backward()
    assert model.support is not None and model.support.shape == (6, 3)


def test_mosaic_sparsity_penalty_positive():
    generator = _temporal(n_regimes=2)
    batch = generator.sample(8)
    model = MOSAIC(observed_dim=6, latent_dim=3, u_dim=2, sparsity=0.1)
    losses = model.loss(model.forward(batch), batch)
    assert losses["sparsity"].item() > 0


# --- CEGEN ------------------------------------------------------------------

def test_sde_generator_shapes():
    generator = LinearSDEGenerator(observed_dim=4, horizon=16, seed=0)
    batch = generator.sample(7)
    assert batch.x.shape == (7, 16, 4)
    assert batch.z is not None and torch.allclose(batch.z, batch.x)


def test_cegen_forward_and_grad():
    generator = LinearSDEGenerator(observed_dim=4, horizon=16, seed=0)
    batch = generator.sample(32)
    model = CEGEN(observed_dim=4, n_regions=4)
    outputs = model.forward(batch)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["w2"])
    losses["loss"].backward()
    assert model.drift_net.net[0].weight.grad is not None


def test_cegen_training_reduces_w2():
    generator = LinearSDEGenerator(observed_dim=3, horizon=20, seed=0)
    model = CEGEN(observed_dim=3, n_regions=4)
    trainer = SimpleTrainer(
        max_epochs=30, lr=1e-3, batch_size=64, steps_per_epoch=10, log_every=0, seed=0
    )
    history = trainer.fit(model, generator)
    assert history["loss"][-1] < history["loss"][0]


# --- vectorised CITRIS ------------------------------------------------------

def test_citris_transition_is_vectorised_over_time():
    from gcmts.causal_representation_learning.methods import CITRIS
    from gcmts.data.synthetic import InterventionalTemporalGenerator

    generator = InterventionalTemporalGenerator(
        n_vars=2, var_dim=2, n_shared=1, horizon=10, seed=0
    )
    batch = generator.sample(8)
    model = CITRIS(observed_dim=5, n_vars=2, var_dim=2, n_shared=1, n_flow_layers=3)
    outputs = model.forward(batch)
    assert outputs.log_prob is not None and outputs.log_prob.shape == (8,)
    assert torch.isfinite(outputs.log_prob).all()


@pytest.mark.slow
def test_mosaic_recovers_latents():
    torch.manual_seed(0)
    generator = TemporalNonlinearGenerator(
        observed_dim=5, latent_dim=3, horizon=1, n_regimes=2, mixing="linear", seed=0
    )
    model = MOSAIC(observed_dim=5, latent_dim=3, u_dim=2)
    trainer = SimpleTrainer(
        max_epochs=1200, lr=1e-3, batch_size=128, steps_per_epoch=15, log_every=0, seed=0
    )
    trainer.fit(model, generator)
    batch = move_batch(generator.sample(2048), next(model.parameters()).device)
    z_pred = model.encode(batch.x, batch.context)
    assert batch.z is not None
    assert mcc(batch.z, z_pred) > 0.3


def test_predict_next_observations_all_temporal_methods():
    """Every temporal method must support one-step forecasting (transition path)."""
    import torch

    from gcmts.causal_representation_learning.methods import IVAE, LEAP, MOSAIC, NCTRL, TDRL
    from gcmts.data.synthetic import TemporalNonlinearGenerator

    torch.manual_seed(0)
    generator = TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=8, max_lag=1, n_regimes=3,
        regime_mode="time", mixing="linear", seed=0,
    )
    batch = generator.sample(8)
    models = [
        IVAE(observed_dim=6, latent_dim=3, u_dim=3),
        LEAP(observed_dim=6, latent_dim=3, u_dim=3),
        TDRL(observed_dim=6, latent_fix_dim=1, latent_dyn_dim=1, latent_obs_dim=1, u_dim=3),
        NCTRL(observed_dim=6, latent_dim=3, n_regimes=3),
        MOSAIC(observed_dim=6, latent_dim=3, u_dim=3),
    ]
    context = {"u": torch.zeros(8, 3)}
    for model in models:
        forecast = model.predict_next_observations(batch.x[:, :-1], 1, context)
        assert forecast.shape == (8, 1, 6)


def test_forecast_with_per_timestep_context():
    """Multi-step rollout must accept a per-timestep (B, T, R) context."""
    import torch

    from gcmts.causal_representation_learning.methods import LEAP, MOSAIC, TDRL
    from gcmts.data.synthetic import TemporalNonlinearGenerator

    torch.manual_seed(0)
    generator = TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=12, max_lag=1, n_regimes=4,
        regime_mode="time", mixing="linear", seed=0,
    )
    batch = generator.sample(8)
    assert batch.context["u"].ndim == 3
    context = {"u": batch.context["u"][:, :-1]}
    models = [
        LEAP(observed_dim=6, latent_dim=3, u_dim=4),
        TDRL(observed_dim=6, latent_fix_dim=1, latent_dyn_dim=1, latent_obs_dim=1, u_dim=4),
        MOSAIC(observed_dim=6, latent_dim=3, u_dim=4),
    ]
    for model in models:
        forecast = model.predict_next_observations(batch.x[:, :-1], 2, context)
        assert forecast.shape == (8, 2, 6)
