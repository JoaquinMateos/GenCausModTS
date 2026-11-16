"""Regression tests for the official MOSAIC adapter.

The adapter has three contracts that are easy to break silently:

* the regime context reaches the temporal prior as an integer label, from
  either of the encodings the benchmark generators emit (per-window one-hot or
  per-timestep integer labels);
* the additive decoder and the fixed unit-variance latent prior see the
  per-variable standardised observations the released pipeline prepares, which
  is what makes the ELBO (and hence the fit) independent of the physical units
  of a suite; and
* ``encode``/``decode``/``forward`` still speak the input units, so
  reconstruction and forecast metrics remain comparable across methods.
"""

from __future__ import annotations

import math

import pytest
import torch

from gcmts.causal_representation_learning.methods.mosaic_official import (
    MOSAICOfficial,
    _ct,
)
from gcmts.data.synthetic import CartPoleGenerator, TemporalNonlinearGenerator
from gcmts.typing import Batch


def _cartpole_batch(*, batch: int = 8, horizon: int = 12) -> tuple[object, Batch]:
    generator = CartPoleGenerator(horizon=horizon, observed_dim=4, n_regimes=3, seed=0)
    return generator, generator.sample(batch)


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


def _model(**kwargs: object) -> MOSAICOfficial:
    return MOSAICOfficial(observed_dim=4, latent_dim=4, u_dim=3, **kwargs)


# --- regime context ---------------------------------------------------------


def test_regime_context_accepts_both_generator_encodings() -> None:
    """One-hot windows and per-timestep integer labels decode to the same label."""
    one_hot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        _ct({"u": one_hot}, 3, 3, torch.device("cpu")), torch.tensor([0, 1, 2])
    )

    # CartPole/latent-SCM: (B, n_classes) indicator repeated over the window.
    labels_per_step = torch.tensor([[2, 2, 1], [0, 1, 0], [1, 1, 1]])
    assert _ct({"u": labels_per_step}, 3, 3, torch.device("cpu")).tolist() == [1, 0, 1]

    # Physics/control: (B, T) integer labels; the final step labels the window.
    assert _ct({"u": labels_per_step.long()}, 3, 3, torch.device("cpu")).tolist() == [1, 0, 1]

    # Per-timestep indicator (B, T, n_classes).
    stacked = torch.nn.functional.one_hot(labels_per_step.long(), 3).float()
    assert _ct({"u": stacked}, 3, 3, torch.device("cpu")).tolist() == [1, 0, 1]

    # Per-window integer label, with or without the trailing singleton.
    per_window = _ct({"u": torch.tensor([2, 0, 1])}, 3, 3, torch.device("cpu"))
    singleton = _ct({"u": torch.tensor([[2], [0], [1]])}, 3, 3, torch.device("cpu"))
    assert per_window.tolist() == singleton.tolist() == [2, 0, 1]


def test_regime_context_defaults_to_one_class_and_rejects_wrong_width() -> None:
    """Stationary suites have no regimes; a misdeclared width is reported."""
    assert _ct(None, 3, 1, torch.device("cpu")).tolist() == [0, 0, 0]
    assert _ct({"u": torch.ones(3, 1)}, 3, 1, torch.device("cpu")).tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="width 4"):
        _ct({"u": torch.zeros(3, 4)}, 3, 3, torch.device("cpu"))


def test_context_reaches_the_vendored_prior_as_labels() -> None:
    """The value the vendored model sees equals the generator's regime."""
    _, batch = _cartpole_batch(batch=5)
    model = _model()
    seen: dict[str, torch.Tensor] = {}
    original = model.core.embed_func.forward

    def recorder(index: torch.Tensor) -> torch.Tensor:
        seen["ct"] = index.detach().clone()
        return original(index)

    model.core.embed_func.forward = recorder  # type: ignore[method-assign]
    model.loss(model(batch), batch)
    assert seen["ct"].tolist() == batch.context["u"].argmax(dim=-1).tolist()
    assert seen["ct"].dtype == torch.long


# --- standardisation --------------------------------------------------------


def test_elbo_is_invariant_to_observation_units() -> None:
    """With per-variable standardisation, the units of a suite cancel out.

    The additive decoder is initialised for unit-variance inputs and the
    reconstruction term is an unweighted MSE, so without standardisation a
    suite's choice of units silently reweights both.  Here two identically
    initialised models see the same data up to a per-variable affine map and
    report the same ELBO.
    """
    _, batch = _cartpole_batch(batch=16)
    torch.manual_seed(0)
    first = _model()
    torch.manual_seed(0)
    second = _model()

    scale = torch.tensor([0.1, 1.0, 0.5, 3.0])
    shift = torch.tensor([0.5, -2.0, 0.0, 10.0])
    rescaled = Batch(
        x=batch.x * scale + shift,
        z=batch.z,
        context=batch.context,
        adjacency=batch.adjacency,
    )

    torch.manual_seed(1)
    loss_a = first.loss(first(batch), batch)["loss"]
    torch.manual_seed(1)
    loss_b = second.loss(second(rescaled), rescaled)["loss"]
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-5)
    # The two models see identical encoder-space inputs, so the fit is
    # independent of the units the suite happens to use (float32 rounding only).
    torch.testing.assert_close(first._obs(batch.x), second._obs(rescaled.x), rtol=1e-4, atol=1e-3)

    # Control: the ablation path is exactly the scale-sensitive official ELBO.
    torch.manual_seed(0)
    raw_first = _model(standardise=False)
    torch.manual_seed(0)
    raw_second = _model(standardise=False)
    torch.manual_seed(1)
    raw_a = float(raw_first.loss(raw_first(batch), batch)["loss"].detach())
    torch.manual_seed(1)
    raw_b = float(raw_second.loss(raw_second(rescaled), rescaled)["loss"].detach())
    assert not math.isnan(raw_b)
    assert abs(raw_a - raw_b) / abs(raw_a) > 0.05


def test_running_statistics_are_batch_moments_on_first_step() -> None:
    _, batch = _cartpole_batch(batch=16)
    model = _model()
    model.loss(model(batch), batch)
    flat = batch.x.reshape(-1, batch.x.shape[-1])
    torch.testing.assert_close(model.obs_mean, flat.mean(dim=0, keepdim=True))
    torch.testing.assert_close(model.obs_var, flat.var(dim=0, unbiased=False, keepdim=True))
    assert int(model.obs_count) == flat.shape[0]


def test_standardisation_is_skipped_for_rendered_images() -> None:
    """CNN features are not prepared variables, so the transform stays off."""
    model = MOSAICOfficial(
        observed_dim=3 * 16 * 16, latent_dim=4, image_shape=(3, 16, 16), image_feature_dim=8
    )
    assert model.standardise is False
    assert not hasattr(model, "obs_mean")


# --- input units of the public interface ------------------------------------


def test_outputs_are_returned_in_input_units() -> None:
    _, batch = _cartpole_batch(batch=16)
    model = _model()
    model.loss(model(batch), batch)

    # encode maps into standardised space, decode maps back.
    with torch.no_grad():
        z = model.encode(batch.x, batch.context)
        decoded = model.decode(z)
    assert decoded.shape == batch.x.shape
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(
        model._denormalise(model._obs(batch.x)), batch.x, rtol=1e-5, atol=1e-5
    )

    # Reconstruction and rollout are in the same units as the batch.
    out = model(batch)
    assert out.x_recon is not None and out.x_recon.shape == batch.x.shape
    per_dim_scale = out.x_recon.std(dim=(0, 1)) / batch.x.std(dim=(0, 1))
    assert torch.all(per_dim_scale < 50), "reconstruction is not in input units"
    with torch.no_grad():
        forecast = model.predict_next_observations(batch.x[:, :8], 2, batch.context)
    assert forecast.shape == (batch.x.shape[0], 2, batch.x.shape[-1])
    assert torch.isfinite(forecast).all()


def test_forward_loss_backward_smoke() -> None:
    _, batch = _npvar_batch()
    model = MOSAICOfficial(observed_dim=6, latent_dim=3, u_dim=4)
    outputs = model(batch)
    assert outputs.z.shape == (8, 12, 3)
    losses = model.loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.core.net.encoder[0].weight.grad is not None
    assert model.core.net.decoder.W2.grad is not None  # type: ignore[union-attr]
