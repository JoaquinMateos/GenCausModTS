"""Tests for the compact SCARF sequential-recourse implementation (III.B)."""

from __future__ import annotations

import torch

from gcmts.data.sequential_recourse import SequentialRecourseGenerator
from gcmts.decision_making.methods.scarf import SCARF


def _generator(**kwargs: object) -> SequentialRecourseGenerator:
    defaults = {"state_dim": 6, "horizon": 8, "action_dim": 3, "seed": 0}
    defaults.update(kwargs)
    return SequentialRecourseGenerator(**defaults)  # type: ignore[arg-type]


def test_generator_shapes_and_context():
    generator = _generator()
    batch = generator.sample(12)
    assert batch.x.shape == (12, 8, 6)
    assert batch.context is not None
    context = batch.context
    required = {
        "sensitive",
        "decision",
        "actionable_mask",
        "propagation",
        "budget",
        "step_cap",
        "noise",
        "cost_weights",
    }
    assert required <= set(context)
    assert context["sensitive"].shape == (12,)
    assert context["decision"].shape == (12, 8)
    assert context["noise"].shape == (12, 8, 6)
    assert context["propagation"].shape == (6, 6)
    assert context["cost_weights"].shape == (6,)
    assert context["actionable_mask"].tolist() == [False, True, True, True, False, False]
    # S is binary and immutable along every factual trajectory.
    assert torch.all((batch.x[:, :, 0] == 0) | (batch.x[:, :, 0] == 1))
    assert torch.allclose(batch.x[:, :, 0], context["sensitive"].unsqueeze(1))
    # The group-blind decision rule does not use coordinate 0.
    flipped = batch.x.clone()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    assert torch.equal(generator.decision(batch.x), generator.decision(flipped))


def test_generator_is_deterministic_for_a_fixed_seed():
    first = _generator(seed=7).sample(9)
    second = _generator(seed=7).sample(9)
    assert torch.equal(first.x, second.x)
    assert first.context is not None and second.context is not None
    for key in ("sensitive", "decision", "noise", "propagation"):
        assert torch.equal(first.context[key], second.context[key])


def test_counterfactual_with_zero_delta_reproduces_the_factual_trajectory():
    generator = _generator(seed=3)
    batch = generator.sample(10)
    delta = torch.zeros_like(batch.x)
    counterfactual = generator.counterfactual(batch, delta, project_budget=False)
    assert torch.allclose(counterfactual.x, batch.x, atol=1e-6)
    repeated = generator.counterfactual(batch, delta, project_budget=False)
    assert torch.allclose(counterfactual.x, repeated.x, atol=1e-7)
    assert counterfactual.context is not None
    assert torch.equal(counterfactual.context["decision"], batch.context["decision"])


def test_counterfactual_only_intervenes_on_actionable_coordinates():
    generator = _generator(seed=1, step_cap=0.3)
    batch = generator.sample(6)
    delta = torch.zeros_like(batch.x)
    delta[:, :, 1:4] = 0.05
    # These entries must be ignored (S immutable, M reached only by propagation).
    delta[:, :, 0] = 5.0
    delta[:, :, 4:] = 5.0
    counterfactual = generator.counterfactual(batch, delta, project_budget=False)
    assert counterfactual.context is not None
    applied = counterfactual.context["intervention"]
    assert torch.all(applied[:, :, [0, 4, 5]] == 0)
    assert applied.abs().max() <= 0.3 + 1e-6
    assert torch.allclose(counterfactual.x[:, :, 0], batch.x[:, :, 0])
    # Mediated features move only through the graph-restricted propagation.
    assert not torch.allclose(counterfactual.x[:, :, 4:], batch.x[:, :, 4:])


def test_counterfactual_respects_the_global_budget():
    generator = _generator(seed=2, budget=0.1)
    batch = generator.sample(8)
    delta = torch.ones_like(batch.x)
    counterfactual = generator.counterfactual(batch, delta, project_budget=True)
    assert counterfactual.context is not None
    applied = counterfactual.context["intervention"]
    weights = batch.context["cost_weights"]
    cost = (applied.abs() * weights).sum(dim=(1, 2))
    assert torch.all(cost <= 0.1 + 1e-5)


def test_scarf_forward_loss_and_backward_are_finite():
    generator = _generator()
    batch = generator.sample(8)
    model = SCARF(state_dim=6, action_dim=3, hidden_dim=32, horizon=8)
    outputs = model(batch)
    assert outputs["intervention"].shape == (8, 8, 6)
    assert outputs["counterfactual"].shape == (8, 8, 6)
    assert outputs["probability"].shape == (8, 8)
    assert outputs["decision"].shape == (8, 8)
    assert outputs["next_state"].shape == (8, 7, 6)
    assert outputs["propagation"].shape == (6, 6)
    # Interventions are generated only on the actionable block.
    assert torch.all(outputs["intervention"][:, :, [0, 4, 5]] == 0)
    assert outputs["intervention"].abs().max() <= model.step_cap + 1e-6
    assert torch.allclose(outputs["counterfactual"][:, :, 0], outputs["sensitive"].unsqueeze(1))

    losses = model.loss(outputs, batch)
    assert {
        "loss",
        "improvement",
        "fairness",
        "budget",
        "transition",
        "propagation",
    } <= set(losses)
    for value in losses.values():
        assert torch.isfinite(value)
    losses["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert torch.isfinite(torch.cat([g.reshape(-1) for g in gradients])).all()
    assert model.identifiability_statement()


def test_scarf_budget_projection_keeps_cost_within_the_context_budget():
    generator = _generator(budget=0.15)
    batch = generator.sample(10)
    model = SCARF(state_dim=6, action_dim=3, hidden_dim=32, horizon=8)
    outputs = model(batch)
    cost = (outputs["intervention"].abs() * model.cost_weights).sum(dim=(1, 2))
    assert torch.all(cost <= 0.15 + 1e-5)
    assert torch.allclose(outputs["cost"], cost, atol=1e-6)


def test_scarf_recourse_and_act_shapes_and_actionability():
    generator = _generator()
    batch = generator.sample(5)
    model = SCARF(state_dim=6, action_dim=3, hidden_dim=32, horizon=8)
    recourse = model.recourse(batch)
    assert {"intervention", "counterfactual", "decision"} <= set(recourse)
    assert recourse["intervention"].shape == (5, 8, 6)
    assert recourse["counterfactual"].shape == (5, 8, 6)
    assert recourse["decision"].shape == (5, 8)
    assert torch.all((recourse["decision"] == 0) | (recourse["decision"] == 1))
    assert torch.all(recourse["intervention"][:, :, [0, 4, 5]] == 0)

    trajectory = model.act(batch.x)
    assert trajectory.shape == (5, 8, 6)
    assert torch.allclose(trajectory[:, :, 0], batch.x[:, 0, 0].unsqueeze(1))
    assert model.act(batch.x[:, 0]).shape == (5, 8, 6)
