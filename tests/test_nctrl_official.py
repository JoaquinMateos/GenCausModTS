"""Regression tests for the official NCTRL adapter.

The NCTRL adapter is easy to get *almost* right while silently changing the
model: the paper's ARHMM models ``p(x_t | x_{<t}, c_t)`` over observations, the
KLD is ``log q - log p``, and the VAE must not resample latents inside
``loss``.  These tests pin each of those contracts against the vendored
official source.
"""

from __future__ import annotations

import pytest
import torch

from gcmts.causal_representation_learning.methods.nctrl_official import NCTRLOfficial
from gcmts.core import SimpleTrainer
from gcmts.data.synthetic import TemporalNonlinearGenerator
from gcmts.typing import Batch


def _npvar_batch(*, batch: int = 8, horizon: int = 12) -> tuple[object, Batch]:
    generator = TemporalNonlinearGenerator(
        observed_dim=6,
        latent_dim=3,
        horizon=horizon,
        max_lag=1,
        n_regimes=4,
        regime_mode="time",
        mixing="mlp",
        seed=0,
    )
    return generator, generator.sample(batch)


def _model(**kwargs: object) -> NCTRLOfficial:
    return NCTRLOfficial(observed_dim=6, latent_dim=3, n_regimes=4, **kwargs)


# --- ARHMM contract ---------------------------------------------------------


def test_official_nctrl_hmm_models_observations() -> None:
    """The ARHMM is sized for, and fed, observations (paper Eq. 5 / App. B.8).

    The vendored ``NCTRL`` hard-codes ``x_dim=z_dim`` because its synthetic
    experiments use a square mixing function; with ``observed_dim != latent_dim``
    the HMM must still model the observed series.
    """
    model = _model()
    assert model.core.hmm.x_dim == 6
    assert model.core.hmm.lags == 1
    assert model.core.hmm.feat == "H", "official config uses mle_scaled:H"
    assert model.core.hmm.mode == "mle_scaled"

    _, batch = _npvar_batch()
    seen: dict[str, torch.Tensor] = {}
    original = model.core.hmm.forward

    def recorder(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seen["input"] = x.detach().clone()
        return original(x)

    model.core.hmm.forward = recorder  # type: ignore[method-assign]
    outputs = model(batch)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    assert seen["input"].shape == batch.x.shape
    assert torch.equal(seen["input"], batch.x), "HMM must see the observations, not z"


def test_official_nctrl_loss_matches_official_objective() -> None:
    """Adapter loss equals the vendored official NCTRL objective, sign included."""
    model = _model()
    _, batch = _npvar_batch()
    with torch.no_grad():
        outputs = model(batch)
        losses = model.loss(outputs, batch)

        obs = outputs.extras["obs"]
        x_recon = outputs.x_recon
        assert x_recon is not None
        mus, logvars = outputs.extras["mean"], outputs.extras["logvar"]
        z_est = outputs.z
        _, c_est = model.core.hmm(obs)
        embeddings = model.core.c_embeddings(c_est)

        recon = model.core.reconstruction_loss(obs[:, :1], x_recon[:, :1]) + (
            model.core.reconstruction_loss(obs[:, 1:], x_recon[:, 1:])
            / (obs.shape[1] - 1)
        )
        log_qz = torch.distributions.Normal(
            mus, torch.exp(logvars / 2).clamp(min=1e-6, max=1e6)
        ).log_prob(z_est)
        log_pz_normal = torch.sum(
            torch.distributions.Normal(
                torch.zeros_like(mus[:, :1]), torch.ones_like(logvars[:, :1])
            ).log_prob(z_est[:, :1]),
            dim=-1,
        ).sum(-1)
        log_qz_normal = torch.sum(log_qz[:, :1], dim=-1).sum(-1)
        # Official: KLD = E_q[log q - log p], a positive divergence.
        expected_kld_normal = (log_qz_normal - log_pz_normal).mean()
        residuals, logabsdet = model.core.transition_prior(z_est, embeddings)
        log_pz = torch.sum(model.core.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        expected_kld_laplace = (
            torch.sum(log_qz[:, 1:], dim=-1).sum(-1) - log_pz
        ).div(obs.shape[1] - 1).mean()
        expected = (
            recon
            + model.core.beta * expected_kld_normal
            + model.core.gamma * expected_kld_laplace
        )

    assert torch.allclose(losses["loss"], expected, atol=1e-6)
    assert torch.allclose(losses["recon"], recon, atol=1e-6)
    # The pre-fix adapter had this term negated, which made the optimiser
    # *maximise* the divergence from the prior and drove the reported loss
    # negative.  A Gaussian ELBO must not be systematically negative here.
    assert float(losses["loss"]) > 0.0


def test_official_nctrl_loss_does_not_resample_latents() -> None:
    """``loss`` is a pure function of ``forward`` (no second encoder pass)."""
    model = _model()
    _, batch = _npvar_batch()
    with torch.no_grad():
        outputs = model(batch)
        first = model.loss(outputs, batch)
        second = model.loss(outputs, batch)
    assert torch.equal(first["loss"], second["loss"])
    assert torch.equal(first["recon"], second["recon"])

    # Nor may it silently re-encode observations.
    calls = {"n": 0}
    original = model.core.net.forward

    def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(*args, **kwargs)

    model.core.net.forward = counted  # type: ignore[method-assign]
    with torch.no_grad():
        model.loss(outputs, batch)
    assert calls["n"] == 0


def test_official_nctrl_hmm_has_its_own_optimiser() -> None:
    """The shared trainer must not update the ARHMM (official two-optimiser step)."""
    model = _model()
    names = {name for name, _ in model.named_parameters()}
    shared_ids = {id(p) for p in model.parameters()}
    shared = {name for name, p in model.named_parameters() if id(p) in shared_ids}
    hmm_names = {name for name in names if name.startswith("core.hmm.")}
    assert hmm_names
    assert not (shared & hmm_names)
    assert names - hmm_names == shared
    assert {id(p) for p in model.hmm_optimizer.param_groups[0]["params"]} == {
        id(p) for name, p in model.named_parameters() if name.startswith("core.hmm.")
    }


def test_official_nctrl_hmm_updates_each_step() -> None:
    model = _model()
    _, batch = _npvar_batch()
    before = {name: p.detach().clone() for name, p in model.core.hmm.named_parameters()}

    outputs = model(batch)
    losses = model.loss(outputs, batch)
    losses["loss"].backward()

    changed = [
        name
        for name, p in model.core.hmm.named_parameters()
        if not torch.equal(p.detach(), before[name])
    ]
    assert changed, "HMM parameters were not updated by their own optimiser"
    assert model.core.net.encoder[0].weight.grad is not None
    assert torch.isfinite(model.core.net.encoder[0].weight.grad).all()


# --- end-to-end -------------------------------------------------------------


def test_official_nctrl_stationary_regimes_reduce_to_one_class() -> None:
    """Stationary suites pass ``n_regimes=1``; the ARHMM must still be valid."""
    torch.manual_seed(0)
    batch = Batch(x=torch.randn(5, 7, 4), context={"u": torch.ones(5, 7, 1)})
    model = NCTRLOfficial(observed_dim=4, latent_dim=4, n_regimes=1, hidden_dim=32)
    assert model.core.hmm.n_class == 1
    outputs = model(batch)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.core.net.encoder[0].weight.grad is not None


def test_official_nctrl_trainer_smoke() -> None:
    generator, _ = _npvar_batch()
    model = _model(hidden_dim=32)
    trainer = SimpleTrainer(
        max_epochs=20,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=8,
        log_every=0,
        seed=0,
        val_size=64,
        val_every=10,
    )
    history = trainer.fit(model, generator)
    assert all(torch.isfinite(torch.tensor(value)) for value in history["loss"])
    assert all(torch.isfinite(torch.tensor(value)) for value in history["recon"])
    assert all(torch.isfinite(torch.tensor(value)) for value in history["hmm"])
    assert torch.isfinite(torch.tensor(trainer.best_val))


def test_official_nctrl_overfits_a_fixed_batch() -> None:
    _, batch = _npvar_batch(batch=16, horizon=8)
    model = _model(hidden_dim=32)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(80):
        outputs = model(batch)
        losses = model.loss(outputs, batch)
        optimiser.zero_grad()
        losses["loss"].backward()
        optimiser.step()
        if step == 0:
            first = float(losses["recon"].detach())
        last = float(losses["recon"].detach())
    assert first is not None and last is not None
    assert last < first, f"reconstruction rose during overfit ({first} -> {last})"


@pytest.mark.slow
def test_official_nctrl_identifies_npvar_latents() -> None:
    """The shared npvar protocol: the fixed adapter must beat the old 0.62 range."""
    from gcmts.core import move_batch
    from gcmts.core.utils import mcc

    torch.manual_seed(0)
    generator = TemporalNonlinearGenerator(
        observed_dim=6,
        latent_dim=3,
        horizon=16,
        max_lag=1,
        n_regimes=4,
        regime_mode="time",
        mixing="mlp",
        seed=0,
    )
    model = _model()
    trainer = SimpleTrainer(
        max_epochs=150,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=10,
        log_every=0,
        seed=0,
        val_size=128,
        val_every=25,
        ema_decay=0.99,
    )
    trainer.fit(model, generator)
    eval_batch = TemporalNonlinearGenerator(
        observed_dim=6,
        latent_dim=3,
        horizon=16,
        max_lag=1,
        n_regimes=4,
        regime_mode="time",
        mixing="mlp",
        seed=0,
    ).sample(512)
    eval_batch = move_batch(eval_batch, next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        z_pred = model.encode(eval_batch.x, eval_batch.context)
    assert eval_batch.z is not None
    score = mcc(eval_batch.z, z_pred)
    assert score > 0.7, f"official NCTRL failed to identify npvar latents (MCC={score:.3f})"
