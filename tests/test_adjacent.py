"""Tests for the newly implemented methods (red taxonomy targets).

Covers CHiLD (I.C.2), iCITRIS (I.D.2), Wu-IPTW (II.A.2), CausalGAN (II.D),
GBTPL (III.A), LacaDM (III.B), CausalConceptTS (III.C), and the adjacent-branch
metrics.
"""

from __future__ import annotations

import torch

from gcmts.causal_representation_learning.methods import CHiLD, iCITRIS
from gcmts.data.binary_scm import BinarySCMGenerator
from gcmts.data.concept_ts import ConceptTimeSeriesGenerator
from gcmts.data.hierarchical import HierarchicalTemporalGenerator
from gcmts.data.multi_objective import MultiObjectiveMDPGenerator
from gcmts.data.propagation import PropagationGenerator
from gcmts.data.synthetic import InterventionalTemporalGenerator
from gcmts.data.synthetic_effect import HarmonicOscillatorGenerator
from gcmts.decision_making.methods import LacaDM
from gcmts.effect_estimation.methods import CausalGAN, WuIPTW
from gcmts.evaluation.metrics import (
    FaithfulnessMetric,
    HypervolumeMetric,
    PathF1Metric,
    SourceAccuracyMetric,
    StructuralViolationMetric,
)
from gcmts.explanation.methods import CausalConceptTS
from gcmts.structure_discovery.methods import GBTPL


def test_child_forward_loss_and_forecast():
    generator = HierarchicalTemporalGenerator(
        width=4, n_layers=2, horizon=12, observed_dim=4, seed=0
    )
    batch = generator.sample(8)
    model = CHiLD(observed_dim=4, latent_dim=4, n_layers=2)
    outputs = model.forward(batch)
    assert outputs.z.shape == (8, 12, 8)
    losses = model.loss(outputs, batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.predict_next_observations(batch.x[:, :-1], 1, None).shape == (8, 1, 4)


def test_icitris_forward_loss_and_adjacency():
    generator = InterventionalTemporalGenerator(n_vars=2, var_dim=2, n_shared=1, horizon=10, seed=0)
    batch = generator.sample(8)
    model = iCITRIS(observed_dim=5, n_vars=2, var_dim=2, n_shared=1, n_flow_layers=3)
    outputs = model.forward(batch)
    assert outputs.z.shape == (8, 10, 5)
    assert outputs.adjacency is not None and outputs.adjacency.shape == (5, 5, 2)
    losses = model.loss(outputs, batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_wu_iptw_forward_loss_and_counterfactual():
    generator = HarmonicOscillatorGenerator(n_units=2, observed_dim=4, horizon=8, seed=0)
    batch = generator.sample(8)
    model = WuIPTW(observed_dim=4, treatment_dim=1, latent_dim=3)
    losses = model.loss(model.forward(batch), batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    weights = model.iptw_weights(batch)
    assert weights.shape == (8,) and (weights > 0).all()
    cf = model.counterfactual(batch, {"A": torch.zeros(8, 1)})
    assert cf.counterfactual.shape == batch.x.shape


def test_causalgan_forward_loss_and_counterfactual():
    generator = BinarySCMGenerator(n_labels=3, observed_dim=6, horizon=8, seed=0)
    batch = generator.sample(16)
    model = CausalGAN(observed_dim=6, n_labels=3)
    losses = model.loss(model.forward(batch), batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    cf = model.counterfactual(batch, {"A": torch.ones(16, 1)})
    assert cf.counterfactual.shape == batch.x.shape


def test_gbtpl_forward_loss_and_paths():
    generator = PropagationGenerator(n_nodes=6, horizon=16, max_path_len=4, seed=0)
    batch = generator.sample(8)
    model = GBTPL(n_nodes=6, horizon=16, max_path_len=3)
    losses = model.loss(model.forward(batch), batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    result = model.sample_paths(batch)
    assert result["paths"].shape == (8, 4)
    assert result["sources"].shape == (8,)


def test_lacadm_forward_loss_and_act():
    generator = MultiObjectiveMDPGenerator(state_dim=4, action_dim=2, horizon=12, seed=0)
    batch = generator.sample(8)
    model = LacaDM(state_dim=4, action_dim=2, n_steps=5)
    losses = model.loss(model.forward(batch), batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.act(batch.x).shape == (8, 2)


def test_causal_concept_ts_forward_loss_and_explain():
    generator = ConceptTimeSeriesGenerator(
        n_channels=4, horizon=16, n_concepts=4, causal_concepts=(0, 2), seed=0
    )
    batch = generator.sample(16)
    model = CausalConceptTS(n_channels=4, horizon=16, n_concepts=4)
    losses = model.loss(model.forward(batch), batch)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    explanation = model.explain(batch)
    assert explanation["attributions"].shape == (16, 4)


def test_adjacent_metrics():
    paths = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]])
    truth = torch.tensor([[0, 1, 2, -1], [3, 5, -1, -1]])
    graph = torch.zeros(6, 6)
    graph[0, 1] = graph[1, 2] = graph[3, 4] = 1.0
    assert PathF1Metric()({"paths": paths}, {"paths": truth}).value > 0
    assert SourceAccuracyMetric()(
        {"sources": paths[:, 0]}, {"sources": truth[:, 0]}
    ).value == 1.0
    assert StructuralViolationMetric()({"paths": paths}, {"graph": graph}).value >= 0
    front = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert HypervolumeMetric(n_samples=500)(front, {"reference": torch.ones(2)}).value > 0
    attributions = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert FaithfulnessMetric()(attributions, {"causal_concepts": torch.tensor([0, 2])}).value > 0
