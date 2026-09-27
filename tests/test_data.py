"""Tests for synthetic generators and representation metrics."""

from __future__ import annotations

import torch

from gcmts.data.synthetic import (
    CartPoleGenerator,
    Causal3DIdentGenerator,
    InterventionalTemporalGenerator,
    LinearSDEGenerator,
    NCTRLGenerator,
    SlowFeatureGenerator,
    TDRLGenerator,
    TemporalNonlinearGenerator,
)
from gcmts.evaluation.metrics import R2DiagMetric


def test_slow_feature_generator_shapes():
    generator = SlowFeatureGenerator(observed_dim=3, horizon=10, seed=0)
    batch = generator.sample(5)
    assert batch.x.shape == (5, 10, 3)
    assert batch.z is not None and batch.z.shape == (5, 10, 3)
    increments = batch.z[:, 1:] - batch.z[:, :-1]
    assert torch.isfinite(increments).all()


def test_temporal_generator_time_regimes():
    generator = TemporalNonlinearGenerator(
        observed_dim=4,
        latent_dim=2,
        horizon=8,
        n_regimes=3,
        regime_mode="time",
        seed=0,
    )
    batch = generator.sample(6)
    assert batch.x.shape == (6, 8, 4)
    assert batch.context is not None
    assert batch.context["u"].shape == (6, 8, 3)


def test_temporal_generator_trajectory_regimes():
    generator = TemporalNonlinearGenerator(
        observed_dim=4, latent_dim=2, horizon=8, n_regimes=3, seed=0
    )
    batch = generator.sample(6)
    assert batch.context is not None
    assert batch.context["u"].shape == (6, 3)


def test_temporal_generator_linear_mixing():
    generator = TemporalNonlinearGenerator(
        observed_dim=5, latent_dim=2, horizon=6, mixing="linear", seed=0
    )
    batch = generator.sample(4)
    assert batch.x.shape == (4, 6, 5)


def test_interventional_generator_targets():
    generator = InterventionalTemporalGenerator(
        n_vars=2, var_dim=2, n_shared=1, horizon=7, seed=0
    )
    batch = generator.sample(5)
    assert batch.context is not None
    targets = batch.context["I"]
    assert targets.shape == (5, 7, 2)
    assert torch.all((targets == 0) | (targets == 1))
    assert torch.all(targets[:, 0] == 0)  # no intervention at t=0


def test_tdrl_generator_blocks_and_context():
    generator = TDRLGenerator(
        observed_dim=6, fix_dim=1, chg_dim=2, obs_dim=1, horizon=10, n_regimes=3, seed=0
    )
    batch = generator.sample(5)
    assert batch.x.shape == (5, 10, 6)
    assert batch.z is not None and batch.z.shape == (5, 10, 4)
    assert batch.context is not None and batch.context["u"].shape == (5, 3)


def test_nctrl_generator_regimes_and_shapes():
    generator = NCTRLGenerator(
        observed_dim=5, latent_dim=3, horizon=12, n_regimes=3, seed=0
    )
    batch = generator.sample(4)
    assert batch.x.shape == (4, 12, 5)
    assert batch.z is not None and batch.z.shape == (4, 12, 3)
    assert batch.context is not None
    regimes = batch.context["r"]
    assert regimes.shape == (4, 12)
    assert int(regimes.min()) >= 0 and int(regimes.max()) < 3


def test_causal3dident_generator_invertible():
    generator = Causal3DIdentGenerator(
        observed_dim=5, latent_dim=5, horizon=9, mixing="invertible", seed=0
    )
    batch = generator.sample(6)
    assert batch.x.shape == (6, 9, 5)
    assert batch.z is not None and batch.z.shape == (6, 9, 5)
    assert batch.adjacency is not None
    assert torch.isfinite(batch.x).all()


def test_causal3dident_generator_nonlinear_non_invertible():
    generator = Causal3DIdentGenerator(
        observed_dim=8, latent_dim=3, horizon=6, mixing="nonlinear", seed=1
    )
    batch = generator.sample(4)
    assert batch.x.shape == (4, 6, 8)
    assert batch.z is not None and batch.z.shape == (4, 6, 3)
    assert torch.isfinite(batch.x).all()


def test_cartpole_generator_state_and_regimes():
    generator = CartPoleGenerator(horizon=20, n_regimes=3, seed=0)
    batch = generator.sample(5)
    assert batch.x.shape == (5, 20, 4)
    assert batch.z is not None and batch.z.shape == (5, 20, 4)
    assert batch.adjacency is None  # no latent causal graph for control systems
    assert batch.context is not None and batch.context["u"].shape == (5, 3)
    assert torch.isfinite(batch.x).all()


def test_cartpole_generator_nonlinear_observations():
    generator = CartPoleGenerator(
        horizon=10, observed_dim=6, mixing="nonlinear", seed=2
    )
    batch = generator.sample(3)
    assert batch.x.shape == (3, 10, 6)
    assert batch.z is not None and batch.z.shape == (3, 10, 4)


def test_linear_sde_generator_stationary_scale():
    generator = LinearSDEGenerator(observed_dim=3, horizon=30, seed=0)
    batch = generator.sample(256)
    assert torch.isfinite(batch.x).all()
    # Stable drift: the process should not blow up.
    assert batch.x.abs().mean() < 50


def test_r2_diag_is_permutation_invariant():
    torch.manual_seed(0)
    prediction = torch.randn(500, 3)
    target = prediction[:, [2, 0, 1]]
    result = R2DiagMetric()(prediction, target)
    assert result.value > 0.99


def test_r2_diag_scale_invariant():
    torch.manual_seed(0)
    prediction = torch.randn(500, 2) * 5.0 + 3.0
    target = prediction / 5.0
    result = R2DiagMetric()(prediction, target)
    assert result.value > 0.99
