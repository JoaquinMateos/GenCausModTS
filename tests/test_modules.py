"""Tests for CRL methods, the effect/disentanglement stubs, and runners."""

from __future__ import annotations

import pytest
import torch

from gcmts.causal_representation_learning.methods import CITRIS, IVAE, LEAP, SlowFlows
from gcmts.core import SimpleTrainer, move_batch
from gcmts.core.utils import mcc
from gcmts.data.synthetic import (
    InterventionalTemporalGenerator,
    NonlinearICAGenerator,
    TemporalNonlinearGenerator,
)
from gcmts.effect_estimation.methods import DummyEffectEstimator
from gcmts.evaluation import (
    BenchmarkTask,
    CrlBenchmarkRunner,
    DisentanglementBenchmarkRunner,
    EffectBenchmarkRunner,
)
from gcmts.static_dynamic_disentanglement.methods import DummyDisentangler

# --- iVAE -------------------------------------------------------------------

def test_ivae_forward_shapes(ivae_batch):
    model = IVAE(observed_dim=5, latent_dim=3, u_dim=4)
    outputs = model.forward(ivae_batch)
    assert outputs.z.shape == (32, 1, 3)
    assert outputs.x_recon.shape == ivae_batch.x.shape
    losses = model.loss(outputs, ivae_batch)
    assert set(losses) == {"loss", "recon", "kl"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.encoder.net.net[0].weight.grad is not None


def test_ivae_generator_context_shape(ivae_batch):
    assert ivae_batch.context is not None
    assert ivae_batch.context["u"].shape == (32, 4)


def test_ivae_training_reduces_loss(ivae_generator):
    model = IVAE(observed_dim=5, latent_dim=3, u_dim=4)
    trainer = SimpleTrainer(
        max_epochs=40,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=10,
        log_every=0,
        seed=0,
    )
    history = trainer.fit(model, ivae_generator)
    assert history["loss"][-1] < history["loss"][0]


@pytest.mark.slow
def test_ivae_recovers_latents_linear():
    torch.manual_seed(0)
    generator = NonlinearICAGenerator(
        observed_dim=5,
        latent_dim=3,
        horizon=1,
        n_regimes=4,
        mixing="linear",
        seed=0,
    )
    model = IVAE(observed_dim=5, latent_dim=3, u_dim=4, hidden_dims=(128, 128))
    trainer = SimpleTrainer(
        max_epochs=2000,
        lr=1e-3,
        batch_size=128,
        steps_per_epoch=15,
        log_every=0,
        seed=0,
    )
    trainer.fit(model, generator)
    batch = move_batch(generator.sample(2048), next(model.parameters()).device)
    z_pred = model.encode(batch.x, batch.context)
    assert batch.z is not None
    score = mcc(batch.z, z_pred)
    assert score > 0.55, f"iVAE failed to identify latents (MCC={score:.3f})"


@pytest.mark.slow
def test_ivae_learns_nonlinear_benchmark():
    torch.manual_seed(0)
    generator = NonlinearICAGenerator(
        observed_dim=5, latent_dim=3, horizon=1, n_regimes=4, mixing="nonlinear", seed=0
    )
    model = IVAE(observed_dim=5, latent_dim=3, u_dim=4)
    trainer = SimpleTrainer(
        max_epochs=1500, lr=1e-3, batch_size=128, steps_per_epoch=15, log_every=0, seed=0
    )
    trainer.fit(model, generator)
    batch = move_batch(generator.sample(2048), next(model.parameters()).device)
    z_pred = model.encode(batch.x, batch.context)
    assert batch.z is not None
    score = mcc(batch.z, z_pred)
    assert score > 0.2, f"iVAE barely learned anything (MCC={score:.3f})"


# --- LEAP -------------------------------------------------------------------

def test_leap_forward_shapes(temporal_generator):
    batch = temporal_generator.sample(16)
    model = LEAP(observed_dim=6, latent_dim=3, max_lag=1, u_dim=2)
    outputs = model.forward(batch)
    assert outputs.z.shape == (16, 12, 3)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.transition_net.net.net[0].weight.grad is not None


def test_leap_transition_shape():
    model = LEAP(observed_dim=4, latent_dim=2, max_lag=2, u_dim=0)
    z_past = torch.randn(5, 2, 2)
    mean = model.transition(z_past)
    assert mean.shape == (5, 2)


# --- Slow Flows -------------------------------------------------------------

def test_slow_flows_roundtrip():
    model = SlowFlows(observed_dim=4, n_flow_layers=4)
    x = torch.randn(6, 3, 4)
    z = model.encode(x)
    x_rec = model.decode(z)
    assert torch.allclose(x, x_rec, atol=1e-4)


def test_slow_flows_forward_loss():
    generator = TemporalNonlinearGenerator(
        observed_dim=4, latent_dim=4, horizon=10, seed=1
    )
    batch = generator.sample(8)
    model = SlowFlows(observed_dim=4, n_flow_layers=4)
    outputs = model.forward(batch)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["nll"])
    losses["loss"].backward()


# --- CITRIS -----------------------------------------------------------------

def test_citris_forward_shapes():
    generator = InterventionalTemporalGenerator(
        n_vars=2, var_dim=2, n_shared=1, horizon=8, seed=0
    )
    batch = generator.sample(12)
    model = CITRIS(observed_dim=5, n_vars=2, var_dim=2, n_shared=1, n_flow_layers=3)
    outputs = model.forward(batch)
    assert outputs.z.shape == (12, 8, 5)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.classifier is not None


def test_citris_requires_square_dims():
    with pytest.raises(ValueError):
        CITRIS(observed_dim=7, n_vars=2, var_dim=2, n_shared=1)


# --- benchmark runners ------------------------------------------------------

def test_crl_benchmark_runner(ivae_generator):
    model = IVAE(observed_dim=5, latent_dim=3, u_dim=4)
    task = BenchmarkTask(
        name="ivae",
        model=model,
        data=ivae_generator.sample(64),
        metrics=["mcc"],
        causal_level="L1_association",
    )
    result = CrlBenchmarkRunner().run(task)
    assert any(m.name == "mcc" for m in result.metrics)


def test_effect_benchmark_runner(simple_var_batch):
    model = DummyEffectEstimator(observed_dim=4)
    task = BenchmarkTask(
        name="effect_dummy",
        model=model,
        data=simple_var_batch,
        metrics=["cf_mae"],
        causal_level="L3_counterfactual",
    )
    result = EffectBenchmarkRunner().run(task)
    assert any(m.name == "cf_mae" for m in result.metrics)


def test_disentanglement_benchmark_runner(simple_var_batch):
    model = DummyDisentangler(observed_dim=4, latent_static_dim=2, latent_dynamic_dim=2)
    task = BenchmarkTask(
        name="disentangle_dummy",
        model=model,
        data=simple_var_batch,
        metrics=["ood_mse"],
        causal_level="L1_generalisation",
    )
    result = DisentanglementBenchmarkRunner().run(task)
    assert any(m.name == "ood_mse" for m in result.metrics)
