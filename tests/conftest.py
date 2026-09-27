"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from gcmts.data.synthetic import (
    LTSCMGenerator,
    NonlinearICAGenerator,
    TemporalNonlinearGenerator,
    VARGenerator,
)
from gcmts.typing import Batch


@pytest.fixture
def simple_var_batch() -> Batch:
    gen = VARGenerator(observed_dim=4, horizon=16, seed=42)
    return gen.sample(8, 16)


@pytest.fixture
def simple_ltscm_batch() -> Batch:
    gen = LTSCMGenerator(observed_dim=5, latent_dim=3, horizon=16, seed=42)
    return gen.sample(8, 16)


@pytest.fixture
def ivae_generator() -> NonlinearICAGenerator:
    return NonlinearICAGenerator(
        observed_dim=5, latent_dim=3, horizon=1, n_regimes=4, seed=0
    )


@pytest.fixture
def ivae_batch(ivae_generator: NonlinearICAGenerator) -> Batch:
    return ivae_generator.sample(32)


@pytest.fixture
def temporal_generator() -> TemporalNonlinearGenerator:
    return TemporalNonlinearGenerator(
        observed_dim=6, latent_dim=3, horizon=12, n_regimes=2, seed=0
    )
