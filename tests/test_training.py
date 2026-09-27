"""Tests for training-time utilities: scheduling, history and plotting."""

from __future__ import annotations

import torch

from gcmts.causal_representation_learning.methods import IVAE
from gcmts.core import SimpleTrainer
from gcmts.core.backbones import gaussian_reconstruction_nll
from gcmts.core.history import plot_training_curves, save_history
from gcmts.data.synthetic import NonlinearICAGenerator


def test_trainer_history_scheduling_and_validation():
    torch.manual_seed(0)
    generator = NonlinearICAGenerator(
        observed_dim=4, latent_dim=2, horizon=1, n_regimes=3, mixing="linear", seed=0
    )
    model = IVAE(observed_dim=4, latent_dim=2, u_dim=3)
    trainer = SimpleTrainer(
        max_epochs=5,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=10,
        val_size=64,
        val_every=2,
        scheduler="cosine",
        log_every=0,
        seed=0,
    )
    history = trainer.fit(model, generator)
    assert {"loss", "lr", "val_loss"} <= set(history)
    assert len(history["lr"]) == 5
    assert len(history["loss"]) == 5
    assert history["loss"][-1] < history["loss"][0]
    assert all(value > 0 for value in history["lr"])
    assert len(history["val_loss"]) >= 1


def test_history_persistence_and_plot(tmp_path):
    history = {
        "loss": [1.0, 0.5, 0.25],
        "recon": [1.0, 0.4, 0.2],
        "lr": [1e-3, 5e-4, 1e-4],
    }
    csv_path = save_history(history, tmp_path)
    assert csv_path.exists()
    assert (tmp_path / "history.json").exists()
    png_path = plot_training_curves(history, tmp_path, title="unit-test")
    assert png_path.exists()
    assert (tmp_path / "training_curves.pdf").exists()


def test_gaussian_reconstruction_nll_prefers_better_fit():
    torch.manual_seed(0)
    target = torch.randn(64, 3)
    good = target + 0.01 * torch.randn(64, 3)
    bad = target + torch.randn(64, 3)
    logvar = torch.zeros(1)
    assert gaussian_reconstruction_nll(good, target, logvar) < gaussian_reconstruction_nll(
        bad, target, logvar
    )


def test_methods_expose_observation_noise_buffer():
    model = IVAE(observed_dim=4, latent_dim=2, u_dim=3, obs_noise=0.1)
    assert "obs_logvar" in dict(model.named_buffers())
    assert torch.isclose(model.obs_logvar, torch.full((1,), 2.0 * torch.log(torch.tensor(0.1))))
