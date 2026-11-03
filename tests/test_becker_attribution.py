"""Tests for the Becker et al. attribution surrogate and its generator."""

from __future__ import annotations

import pytest
import torch

from gcmts.data.concept_factor_ts import ConceptFactorTimeSeriesGenerator
from gcmts.explanation.methods.becker_attribution import BeckerAttribution
from gcmts.typing import CausalLevel

CONTEXT_KEYS = {
    "z",
    "w",
    "causal_concepts",
    "true_attribution",
    "true_ablation",
    "pair_x",
}


def _generator(seed: int = 0) -> ConceptFactorTimeSeriesGenerator:
    return ConceptFactorTimeSeriesGenerator(seed=seed)


# --- generator --------------------------------------------------------------


def test_generator_determinism_and_shapes() -> None:
    first = _generator(seed=3).sample(8)
    second = _generator(seed=3).sample(8)
    assert first.x.shape == (8, 12, 6)
    assert first.z is not None and first.z.shape == (8, 12, 4)
    assert first.outcome is not None and first.outcome.shape == (8,)
    assert first.context is not None
    assert set(first.context) >= CONTEXT_KEYS
    assert first.context["z"].shape == (8, 4)
    assert first.context["w"].shape == (4,)
    assert first.context["true_attribution"].shape == (8, 4)
    assert first.context["true_ablation"].shape == (8, 4)
    assert first.context["pair_x"].shape == (8, 12, 6)
    assert torch.equal(first.context["causal_concepts"], torch.tensor([0, 2]))

    # Same seed -> identical draw; a different seed -> a different draw.
    for key in ("x", "z"):
        assert torch.allclose(getattr(first, key), getattr(second, key))
    assert torch.equal(first.outcome, second.outcome)
    assert torch.allclose(first.context["pair_x"], second.context["pair_x"])
    assert not torch.allclose(first.x, _generator(seed=4).sample(8).x)


def test_generator_ground_truth_attributions() -> None:
    batch = _generator(seed=1).sample(16)
    assert batch.context is not None
    z = batch.context["z"]
    weights = batch.context["w"]
    tanh = torch.tanh(z)
    assert torch.allclose(batch.context["true_attribution"], weights * (1.0 - tanh**2))
    assert torch.allclose(batch.context["true_ablation"], weights * tanh)
    # Nuisance factors have exactly zero ground-truth attribution.
    nuisance = [1, 3]
    assert torch.equal(batch.context["true_attribution"][:, nuisance], torch.zeros(16, 2))
    assert torch.equal(batch.context["true_ablation"][:, nuisance], torch.zeros(16, 2))


def test_true_ablation_is_monotone_in_each_causal_factor() -> None:
    generator = _generator(seed=0)
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    z = torch.zeros(values.numel(), 4)
    z[:, 0] = values
    z[:, 2] = values
    batch = generator.sample(values.numel(), z=z)
    assert batch.context is not None
    ablation = batch.context["true_ablation"]
    # Factor 0 has w_0 = +2 and factor 2 has w_2 = -1.5: strictly monotone,
    # with opposite signs.
    assert torch.all(ablation[1:, 0] > ablation[:-1, 0])
    assert torch.all(ablation[1:, 2] < ablation[:-1, 2])


def test_true_ablation_ranks_causal_above_nuisance() -> None:
    batch = _generator(seed=2).sample(32)
    assert batch.context is not None
    magnitude = batch.context["true_ablation"].abs()
    causal = batch.context["causal_concepts"]
    nuisance = torch.tensor([1, 3])
    assert torch.all(
        magnitude[:, causal].min(dim=-1).values > magnitude[:, nuisance].max(dim=-1).values
    )


# --- model ------------------------------------------------------------------


def test_forward_loss_backward_is_finite() -> None:
    batch = _generator().sample(16)
    model = BeckerAttribution()
    outputs = model.forward(batch)
    assert outputs["x_recon"].shape == batch.x.shape
    assert outputs["x_recon_pair"].shape == batch.x.shape
    assert outputs["mu"].shape == (16, 4)
    assert outputs["shared"].shape == (16, 4)

    losses = model.loss(outputs, batch)
    assert {"loss", "recon", "kl", "bce"} <= set(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_forward_without_pair_context() -> None:
    batch = _generator(seed=5).sample(6)
    batch.context = {key: value for key, value in batch.context.items() if key != "pair_x"}
    model = BeckerAttribution()
    outputs = model.forward(batch)
    assert "x_recon_pair" not in outputs
    assert torch.isfinite(model.loss(outputs, batch)["loss"])


def test_explain_shapes_and_finiteness() -> None:
    batch = _generator(seed=6).sample(20)
    model = BeckerAttribution()
    explanation = model.explain(batch)
    assert explanation["attributions"].shape == (20, 4)
    assert explanation["attributions_grad"].shape == (20, 4)
    assert explanation["labels"].shape == (20,)
    assert explanation["mu"].shape == (20, 4)
    assert torch.isfinite(explanation["attributions"]).all()
    assert torch.isfinite(explanation["attributions_grad"]).all()
    assert not explanation["mu"].requires_grad
    # The mean-ablation and gradient heads need not have the same scale.
    assert explanation["attributions"].abs().sum() > 0


def test_tiny_training_reduces_loss_and_stays_finite() -> None:
    torch.manual_seed(0)
    generator = _generator(seed=7)
    model = BeckerAttribution()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    last = None
    for _ in range(30):
        batch = generator.sample(64)
        loss = model.loss(model.forward(batch), batch)["loss"]
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    assert first is not None and last is not None
    assert torch.isfinite(torch.tensor(last))
    assert last < first

    explanation = model.explain(generator.sample(16))
    assert torch.isfinite(explanation["attributions"]).all()
    assert torch.isfinite(explanation["attributions_grad"]).all()


def test_manipulate_changes_prediction() -> None:
    batch = _generator(seed=8).sample(12)
    model = BeckerAttribution()
    baseline = model.manipulate(batch, factor=0, delta=0.0)
    moved = model.manipulate(batch, factor=0, delta=1.5)
    assert moved["mu"].shape == (12, 4)
    assert moved["x"].shape == batch.x.shape
    assert moved["prediction"].shape == (12,)
    assert torch.allclose(moved["mu"][:, 0], baseline["mu"][:, 0] + 1.5)
    assert not torch.allclose(moved["prediction"], baseline["prediction"])
    assert not torch.allclose(moved["x"], baseline["x"])
    with pytest.raises(ValueError):
        model.manipulate(batch, factor=4, delta=1.0)


def test_causal_level_and_identifiability_statement() -> None:
    model = BeckerAttribution()
    assert model.causal_level() == CausalLevel.INTERVENTION
    statement = model.identifiability_statement()
    assert "Ada-GVAE" in statement
    assert "no identifiability theorem" in statement.lower()
