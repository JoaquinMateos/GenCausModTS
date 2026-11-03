"""Tests for the RCA-VAE regime-shift generator and method (III.B)."""

from __future__ import annotations

import math

import torch

from gcmts.core import SimpleTrainer
from gcmts.data.regime_shift import RegimeShiftPropagationGenerator
from gcmts.structure_discovery.methods.rca_vae import RCAVAE
from gcmts.typing import Batch


def _generator(seed: int = 0) -> RegimeShiftPropagationGenerator:
    return RegimeShiftPropagationGenerator(n_nodes=8, horizon=32, seed=seed)


def _batch(seed: int = 0, n_samples: int = 4) -> Batch:
    return _generator(seed).sample(n_samples)


def test_generator_is_deterministic_given_seed() -> None:
    first = _generator(7).sample(6)
    second = _generator(7).sample(6)
    assert torch.equal(first.x, second.x)
    assert first.context is not None and second.context is not None
    for key in first.context:
        assert torch.equal(first.context[key], second.context[key]), key
    other = _generator(8).sample(6)
    assert not torch.equal(first.x, other.x)


def test_generator_shapes_and_regime_split() -> None:
    generator = _generator(0)
    horizon, nodes = 32, 8
    batch = generator.sample(6, horizon)
    assert batch.x.shape == (6, horizon, nodes)
    assert batch.context is not None
    context = batch.context
    assert context["graph"].shape == (nodes, nodes)
    assert context["root_causes"].shape == (6, nodes)
    assert context["root_causes"].dtype == torch.bool
    assert context["change_point"].shape == (6,)
    assert context["regime"].shape == (6, horizon)
    assert context["source"].shape == (6,)
    assert context["path"].shape == (6, generator.max_path_len + 1)
    counts = context["root_causes"].sum(dim=1)
    assert bool(((counts >= 1) & (counts <= 2)).all())
    low, high = max(1, horizon // 3), max(1, horizon // 3) + 1
    high = max(high, (2 * horizon) // 3)
    assert bool(((context["change_point"] >= low) & (context["change_point"] < high)).all())
    for index in range(6):
        cp = int(context["change_point"][index])
        assert bool((context["regime"][index, :cp] == 0).all())
        assert bool((context["regime"][index, cp:] == 1).all())
        assert bool(context["root_causes"][index, int(context["source"][index])])


def test_generator_matrix_is_stable_and_changes_only_root_rows() -> None:
    generator = _generator(3)
    matrix = generator.pre_change_matrix
    assert torch.allclose(matrix.abs().sum(dim=-1), torch.full((8,), 0.9), atol=1e-6)
    assert float(torch.linalg.eigvals(matrix).abs().max()) <= 0.9 + 1e-5
    batch = generator.sample(1)
    assert batch.context is not None
    roots = batch.context["root_causes"][0].nonzero().flatten()
    post = generator._post_matrix(roots)
    unchanged = (~batch.context["root_causes"][0]).nonzero().flatten()
    assert torch.equal(post[unchanged], matrix[unchanged])
    assert not torch.allclose(post[roots], matrix[roots])


def test_model_forward_loss_backward_and_edge_posteriors() -> None:
    torch.manual_seed(0)
    batch = _batch()
    model = RCAVAE(n_nodes=8, hidden_dim=32)
    outputs = model(batch)
    assert outputs["recon"].shape == batch.x.shape
    assert outputs["edge_logits"].shape == (4, 8, 8, 3)
    assert outputs["edge_probs"].shape == outputs["edge_logits"].shape
    assert outputs["window_probs"].shape[-1] == 3
    totals = outputs["edge_probs"].sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)
    losses = model.loss(outputs, batch)
    losses["loss"].backward()
    for key in ("loss", "recon", "kl", "pcm"):
        assert torch.isfinite(losses[key]), key
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_root_causes_and_latent_adjacency() -> None:
    torch.manual_seed(0)
    batch = _batch()
    model = RCAVAE(n_nodes=8, hidden_dim=32)
    model.eval()
    outputs = model(batch)
    result = model.root_causes(
        outputs["edge_probs_pre"], outputs["edge_probs_post"], parents=batch.x
    )
    assert result["scores"].shape == (4, 8)
    assert torch.isfinite(result["scores"]).all()
    assert torch.isfinite(result["kl"]).all()
    assert (result["scores"] >= 0).all()
    adjacency = model.get_latent_adjacency()
    assert adjacency is not None and adjacency.shape == (8, 8, 1)
    assert set(adjacency.unique().tolist()) <= {0.0, 1.0}


def test_sample_paths_returns_paths_sources_and_scores() -> None:
    torch.manual_seed(0)
    batch = _batch()
    model = RCAVAE(n_nodes=8, hidden_dim=32)
    model.eval()
    result = model.sample_paths(batch)
    assert result["paths"].shape == (4, model.max_path_len + 1)
    assert result["sources"].shape == (4,)
    assert result["root_cause_scores"].shape == (4, 8)
    assert torch.isfinite(result["root_cause_scores"]).all()
    assert bool((result["sources"] >= 0).all())
    assert bool((result["sources"] < 8).all())


def test_two_epoch_cpu_smoke_fit() -> None:
    torch.manual_seed(0)
    generator = _generator(0)
    model = RCAVAE(n_nodes=8, hidden_dim=16)
    trainer = SimpleTrainer(
        max_epochs=2,
        batch_size=4,
        steps_per_epoch=2,
        log_every=0,
        device="cpu",
        seed=0,
        val_size=0,
    )
    history = trainer.fit(model, generator)
    assert len(history["loss"]) == 2
    assert all(math.isfinite(value) for value in history["loss"])
