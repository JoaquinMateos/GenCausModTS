"""Tests for training-time utilities: scheduling, history and plotting."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gcmts.causal_representation_learning.methods import IVAE
from gcmts.core import SimpleTrainer
from gcmts.core.backbones import gaussian_reconstruction_nll
from gcmts.core.base import BaseModel
from gcmts.core.history import plot_training_curves, save_history
from gcmts.core.trainer import AdaptiveScheduler
from gcmts.data.synthetic import NonlinearICAGenerator
from gcmts.evaluation.report import paired_wilcoxon
from gcmts.typing import Batch


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


def test_paired_wilcoxon_detects_separation():
    rows = [{"case": "a", "m": 0.1}, {"case": "a", "m": 0.2}, {"case": "a", "m": 0.15},
            {"case": "a", "m": 0.12}, {"case": "a", "m": 0.18},
            {"case": "b", "m": 0.9}, {"case": "b", "m": 0.8}, {"case": "b", "m": 0.95},
            {"case": "b", "m": 0.85}, {"case": "b", "m": 0.88}]
    assert paired_wilcoxon(rows, "a", "b", "m") <= 0.1


def test_adaptive_scheduler_warms_up_then_decays_to_floor():
    optimiser = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = AdaptiveScheduler(
        optimiser, warmup_epochs=2, patience=2, factor=0.5, threshold=0.0, min_lr_ratio=0.25
    )
    lrs = []
    for _ in range(30):
        scheduler.step(1.0)  # a constant metric is always a plateau
        lrs.append(optimiser.param_groups[0]["lr"])
    assert lrs[0] == pytest.approx(0.5)  # warm-up epoch 1/2
    assert lrs[1] == pytest.approx(1.0)  # warm-up epoch 2/2
    assert lrs[2] == pytest.approx(1.0)  # nominal rate restored before decaying
    assert lrs[-1] == pytest.approx(0.25)  # floored at min_lr_ratio
    assert all(b <= a + 1e-12 for a, b in zip(lrs[1:], lrs[2:], strict=False))


def test_adaptive_scheduler_keeps_lr_while_improving():
    optimiser = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = AdaptiveScheduler(optimiser, warmup_epochs=0, patience=1, factor=0.5)
    for step in range(20):
        scheduler.step(100.0 - step)  # strictly improving
    assert optimiser.param_groups[0]["lr"] == pytest.approx(1.0)


class _ScalarModel(BaseModel):
    """Trivial model whose loss is ``(param - batch.mean())**2``."""

    def __init__(self, value: float) -> None:
        super().__init__(observed_dim=1)
        self.weight = nn.Parameter(torch.tensor([value]))

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.weight

    def loss(self, outputs: torch.Tensor, batch: Batch) -> dict[str, torch.Tensor]:
        return {"loss": ((outputs - batch.x.mean()) ** 2).mean()}

    def identifiability_statement(self) -> str:
        return "scalar"


class _ConstantSource:
    """Serves constant-mean batches (minimal ``sample`` contract)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: object) -> Batch:
        return Batch(x=torch.full((n_samples, 1, 1), self.value))


def test_permissive_early_stopping_fires_and_restores_best_weights():
    # Train pushes the scalar towards +10, while validation (best at init)
    # degrades, so a permissive stop fires and the best weights are restored.
    def run(restore: bool) -> tuple[SimpleTrainer, float, dict[str, list[float]]]:
        model = _ScalarModel(0.0)
        trainer = SimpleTrainer(
            max_epochs=50,
            lr=0.1,
            batch_size=4,
            steps_per_epoch=2,
            val_size=4,
            val_every=1,
            early_stopping_patience=3,
            restore_best_weights=restore,
            scheduler="none",
            log_every=0,
            seed=0,
        )
        history = trainer.fit(model, _ConstantSource(10.0), _ConstantSource(-10.0))
        return trainer, model.weight.item(), history

    trainer_restored, restored, history = run(restore=True)
    _, final, _ = run(restore=False)

    assert trainer_restored.stopped_epoch is not None
    assert trainer_restored.stopped_epoch < 50
    assert trainer_restored.best_epoch == 0
    assert len(history["loss"]) == trainer_restored.stopped_epoch
    # Restoring the best epoch's weights must undo later (degrading) updates.
    assert restored < final


def test_early_stopping_disabled_by_default_runs_full_budget():
    model = _ScalarModel(0.0)
    trainer = SimpleTrainer(
        max_epochs=6,
        lr=1e-3,
        batch_size=4,
        steps_per_epoch=2,
        val_size=4,
        val_every=1,
        scheduler="none",
        log_every=0,
        seed=0,
    )
    history = trainer.fit(model, _ConstantSource(1.0))
    assert len(history["loss"]) == 6
    assert trainer.stopped_epoch is None


class _KLModel(BaseModel):
    """Scalar model exposing an annealable ``kl_weight``."""

    def __init__(self, kl_weight: float) -> None:
        super().__init__(observed_dim=1)
        self.weight = nn.Parameter(torch.zeros(1))
        self.kl_weight = kl_weight

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.weight

    def loss(self, outputs: torch.Tensor, batch: Batch) -> dict[str, torch.Tensor]:
        return {
            "loss": ((outputs - batch.x.mean()) ** 2).mean(),
            "kl_weight": torch.as_tensor(self.kl_weight),
        }

    def identifiability_statement(self) -> str:
        return "scalar-kl"


def test_kl_warmup_ramps_then_restores_target():
    model = _KLModel(kl_weight=2.0)
    trainer = SimpleTrainer(
        max_epochs=5, lr=1e-3, batch_size=4, steps_per_epoch=2,
        kl_warmup_epochs=4, scheduler="none", log_every=0, seed=0,
    )
    history = trainer.fit(model, _ConstantSource(1.0))
    # Epoch e uses target * (e + 1) / warmup, capped at the target.
    assert history["kl_weight"] == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.0])
    assert model.kl_weight == pytest.approx(2.0)


def test_amp_flag_is_safe_on_cpu():
    model = _ScalarModel(0.0)
    trainer = SimpleTrainer(
        max_epochs=2, lr=1e-3, batch_size=4, steps_per_epoch=2,
        device="cpu", amp=True, scheduler="none", log_every=0, seed=0,
    )
    history = trainer.fit(model, _ConstantSource(1.0))
    assert len(history["loss"]) == 2
    assert trainer._use_amp is False  # AMP is CUDA-only


def test_trainer_ema_runs_and_averages_weights():
    model = _ScalarModel(0.0)
    trainer = SimpleTrainer(
        max_epochs=3, lr=0.1, batch_size=4, steps_per_epoch=2,
        device="cpu", ema_decay=0.9, scheduler="none", log_every=0, seed=0,
    )
    trainer.fit(model, _ConstantSource(1.0))
    assert trainer._ema is not None
    assert torch.isfinite(model.weight).all()
