"""Regression tests for the official iVAE adapter.

These pin the parts of the adapter that are easy to get silently wrong: the
sign of the ELBO, the auxiliary-regime encoding, the absence of a second
stochastic sampling pass inside ``loss``, the reconstruction/scale path and the
device/dtype handling of the wrapped official module.
"""

from __future__ import annotations

import pytest
import torch

from gcmts.causal_representation_learning.methods.ivae_official import (
    IVAEOfficial,
    _aux,
    _gaussian_logpdf,
)
from gcmts.core import SimpleTrainer, move_batch
from gcmts.core.utils import mcc
from gcmts.data.synthetic import (
    NonlinearICAGenerator,
    TemporalNonlinearGenerator,
)


def _npvar_batch(batch: int = 8, horizon: int = 12) -> tuple[IVAEOfficial, object]:
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


def test_official_ivae_forward_shapes_and_grad() -> None:
    generator, batch = _npvar_batch()
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4)
    outputs = model.forward(batch)
    assert outputs.z.shape == (8, 12, 3)
    assert outputs.x_recon is not None and outputs.x_recon.shape == batch.x.shape
    assert torch.isfinite(outputs.x_recon).all()
    assert outputs.log_prob is not None and outputs.log_prob.shape == (8, 12)
    assert outputs.extras is not None
    # The official encoder head is a variance; the reported extra is a log.
    logvar = outputs.extras["logvar"]
    assert torch.isfinite(logvar).all()

    losses = model.loss(outputs, batch)
    assert set(losses) == {"loss", "recon", "kl"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for core_module in (model.core.g, model.core.logv, model.core.f, model.core.logl):
        assert core_module.fc[-1].weight.grad is not None
        assert torch.isfinite(core_module.fc[-1].weight.grad).all()

    with torch.no_grad():
        context = {"u": batch.context["u"][:, :10]}
        forecast = model.predict_next_observations(batch.x[:, :10], 2, context)
    assert forecast.shape == (8, 2, 6)
    assert torch.isfinite(forecast).all()


def test_official_ivae_loss_is_negative_elbo() -> None:
    _, batch = _npvar_batch()
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4)
    outputs = model(batch)
    extras = outputs.extras
    assert extras is not None
    log_px, log_q, log_p = extras["log_px"], extras["log_q"], extras["log_p"]
    losses = model.loss(outputs, batch)
    # The trainer minimises ``loss``: it must be the *negative* ELBO.
    assert torch.allclose(losses["loss"], -(log_px + log_p - log_q).mean())
    assert torch.allclose(losses["loss"], losses["recon"] + losses["kl"])
    assert torch.allclose(losses["recon"], -log_px.mean())
    # ``loss`` is a pure function of the forward pass: it must not resample z.
    again = model.loss(outputs, batch)
    assert torch.equal(losses["loss"], again["loss"])


def test_official_ivae_objective_matches_official_elbo() -> None:
    """The reported loss must equal the vendored model's own ELBO."""
    from gcmts.vendor.ivaE.nets import Normal

    _, batch = _npvar_batch()
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4)
    with torch.no_grad():
        z_fixed = model.encode(batch.x, batch.context).reshape(-1, 3)
        sample = torch.randn(4, 3)
        reference = Normal().log_pdf(
            sample, torch.zeros(4, 3), 2.0 * torch.ones(4, 3), reduce=False
        )
        assert torch.allclose(
            _gaussian_logpdf(sample, torch.zeros(4, 3), 2.0 * torch.ones(4, 3)),
            reference,
        )
    flat, u, _, _ = model._flat(batch.x, batch.context)
    model.core.encoder_dist.sample = lambda mean, var: z_fixed  # type: ignore[method-assign]
    model._sample = lambda mean, var: z_fixed  # type: ignore[method-assign]
    outputs = model(batch)
    losses = model.loss(outputs, batch)
    official_elbo, _ = model.core.elbo(flat, u)
    assert torch.allclose(-losses["loss"], official_elbo, atol=1e-5)


def test_official_ivae_aux_encodings_are_equivalent() -> None:
    batch, time, regimes = 5, 7, 4
    torch.manual_seed(0)
    labels = torch.randint(0, regimes, (batch, time))
    expected = torch.nn.functional.one_hot(labels, regimes).float().reshape(-1, regimes)
    for context_u in (labels, labels.float(), labels.unsqueeze(-1)):
        encoded = _aux({"u": context_u}, batch, time, regimes, torch.device("cpu"))
        assert torch.equal(encoded, expected)
    one_hot = torch.nn.functional.one_hot(labels, regimes).float()
    assert torch.equal(_aux({"u": one_hot}, batch, time, regimes, torch.device("cpu")), expected)
    per_trajectory = torch.randint(0, regimes, (batch,))
    trajectory_expected = (
        torch.nn.functional.one_hot(per_trajectory, regimes)
        .float()
        .unsqueeze(1)
        .expand(-1, time, -1)
        .reshape(-1, regimes)
    )
    for context_u in (
        per_trajectory,
        per_trajectory.unsqueeze(1),
        torch.nn.functional.one_hot(per_trajectory, regimes).float(),
    ):
        encoded = _aux({"u": context_u}, batch, time, regimes, torch.device("cpu"))
        assert torch.equal(encoded, trajectory_expected)
    assert torch.equal(
        _aux(None, batch, time, regimes, torch.device("cpu")),
        torch.zeros(batch * time, regimes),
    )
    # A constant context is a constant auxiliary variable; the adapter must not
    # try to one-hot encode label 1 with a single class (Causal3DIdent).
    assert torch.equal(
        _aux({"u": torch.ones(batch, 1)}, batch, time, 1, torch.device("cpu")),
        torch.zeros(batch * time, 1),
    )
    with pytest.raises(ValueError):
        _aux(
            {"u": torch.nn.functional.one_hot(per_trajectory, regimes + 1).float()},
            batch,
            time,
            regimes,
            torch.device("cpu"),
        )


def test_official_ivae_accepts_generator_context_styles() -> None:
    """A constant context must not crash and integer labels must be one-hot."""
    torch.manual_seed(0)
    observations = torch.randn(4, 6, 5)
    from gcmts.typing import Batch

    stationary = IVAEOfficial(observed_dim=5, latent_dim=2, u_dim=1)
    constant = Batch(x=observations, context={"u": torch.ones(4, 1)})
    losses = stationary.loss(stationary(constant), constant)
    assert torch.isfinite(losses["loss"])

    labelled = IVAEOfficial(observed_dim=5, latent_dim=2, u_dim=3)
    per_timestep = Batch(x=observations, context={"u": torch.randint(0, 3, (4, 6))})
    losses = labelled.loss(labelled(per_timestep), per_timestep)
    assert torch.isfinite(losses["loss"])


def test_official_ivae_fits_a_fixed_batch() -> None:
    """Catches ELBO sign errors: fitting one batch must lower the NLL."""
    torch.manual_seed(0)
    generator, batch = _npvar_batch(batch=16, horizon=8)
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(150):
        outputs = model(batch)
        losses = model.loss(outputs, batch)
        optimiser.zero_grad()
        losses["loss"].backward()
        optimiser.step()
        if step == 0:
            first = float(losses["recon"].detach())
        last = float(losses["recon"].detach())
    assert first is not None and last is not None
    assert last < first, f"reconstruction NLL rose during training ({first} -> {last})"


def test_official_ivae_decode_matches_forward_and_scale() -> None:
    _, batch = _npvar_batch()
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4, x_scale=3.0)
    outputs = model(batch)
    with torch.no_grad():
        mean = outputs.extras["mean"]
        decoded = model.decode(mean)
    assert torch.allclose(model.decode(outputs.z), outputs.x_recon, atol=1e-6)
    reference = (
        model.core.f(mean.reshape(-1, 3)).clamp(-model.decoder_clip, model.decoder_clip) / 3.0
    ).reshape(8, 12, 6)
    assert torch.allclose(decoded, reference, atol=1e-6)


def test_official_ivae_buffers_follow_device_and_dtype() -> None:
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4, decoder_var=0.04)
    state = model.state_dict()
    assert "core.prior_mean" in state
    assert "core.decoder_var" in state
    assert torch.allclose(model.core.decoder_var, torch.tensor([0.04]))
    model.double()
    assert model.core.prior_mean.dtype == torch.float64
    assert model.core.decoder_var.dtype == torch.float64
    _, batch = _npvar_batch()
    outputs = model(batch.to(torch.float64) if batch.x.dtype == torch.float64 else batch)
    losses = model.loss(outputs, batch)
    assert losses["loss"].dtype == torch.float64
    assert torch.isfinite(losses["loss"])


def test_official_ivae_obs_noise_sets_decoder_variance() -> None:
    model = IVAEOfficial(observed_dim=4, latent_dim=2, u_dim=2, obs_noise=0.3)
    assert torch.allclose(model.core.decoder_var, torch.tensor([0.09]))
    with pytest.raises(ValueError):
        IVAEOfficial(observed_dim=4, latent_dim=2, obs_noise=0.0)


def test_official_ivae_trainer_smoke() -> None:
    generator, _ = _npvar_batch()
    model = IVAEOfficial(observed_dim=6, latent_dim=3, u_dim=4)
    trainer = SimpleTrainer(
        max_epochs=25,
        lr=1e-3,
        batch_size=64,
        steps_per_epoch=10,
        log_every=0,
        seed=0,
        val_size=64,
        val_every=5,
    )
    history = trainer.fit(model, generator)
    assert history["loss"][-1] < history["loss"][0]
    assert all(torch.isfinite(torch.tensor(value)) for value in history["loss"])
    assert all(torch.isfinite(torch.tensor(value)) for value in history["recon"])
    assert all(torch.isfinite(torch.tensor(value)) for value in history["kl"])
    assert torch.isfinite(torch.tensor(trainer.best_val))


@pytest.mark.slow
def test_official_ivae_recovers_latents() -> None:
    """The official i.i.d. setting: nonlinear mixing, indexed by regime."""
    torch.manual_seed(0)
    generator = NonlinearICAGenerator(
        observed_dim=5,
        latent_dim=3,
        horizon=1,
        n_regimes=4,
        mixing="nonlinear",
        seed=0,
    )
    model = IVAEOfficial(observed_dim=5, latent_dim=3, u_dim=4)
    trainer = SimpleTrainer(
        max_epochs=400,
        lr=1e-3,
        batch_size=128,
        steps_per_epoch=10,
        log_every=0,
        seed=0,
    )
    trainer.fit(model, generator)
    batch = move_batch(generator.sample(2048), next(model.parameters()).device)
    z_pred = model.encode(batch.x, batch.context)
    assert batch.z is not None
    score = mcc(batch.z, z_pred)
    assert score > 0.5, f"official iVAE failed to identify latents (MCC={score:.3f})"
