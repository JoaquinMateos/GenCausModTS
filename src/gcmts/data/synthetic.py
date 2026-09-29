"""Synthetic data generators used across the review benchmarks."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from gcmts.core.backbones import RealNVP
from gcmts.core.tscm import LTSCM, TSCM
from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = [
    "VARGenerator",
    "LTSCMGenerator",
    "NonlinearICAGenerator",
    "TemporalNonlinearGenerator",
    "InterventionalTemporalGenerator",
    "SlowFeatureGenerator",
    "LinearSDEGenerator",
    "TDRLGenerator",
    "NCTRLGenerator",
    "Causal3DIdentGenerator",
    "CartPoleGenerator",
]


def random_invertible_mixing(
    dim: int,
    *,
    n_layers: int = 4,
    hidden: int = 64,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Return a fixed, randomly initialised invertible map ``R^dim -> R^dim``.

    Implemented as a :class:`RealNVP` with frozen (non-trainable) parameters so
    it can serve as ground-truth mixing for flow-based CRL methods.
    """
    with torch.random.fork_rng(devices=[]):
        if generator is not None:
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
        flow = RealNVP(dim, n_layers=n_layers, hidden_dims=(hidden, hidden))
    for param in flow.parameters():
        param.requires_grad_(False)
    flow.eval()
    return _ForwardOnly(flow)


class _ForwardOnly(nn.Module):
    """Wrap a flow so ``forward`` returns only the transformed tensor."""

    def __init__(self, flow: nn.Module) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, z: Tensor) -> Tensor:
        return cast(Tensor, self.flow(z)[0])


class VARGenerator(BaseDataGenerator):
    r"""Vector-autoregressive process with random DAG coefficients.

    .. math::
        X_t = \Sigma_{\tau=1}^{p} A^{(\tau)} X_{t-\tau} + \epsilon_t,
        \quad \epsilon_t \sim \mathcal N(0, \Sigma)

    The coefficient matrices ``A^{(\tau)}`` are masked by a random DAG so that
    ``X_{i,t}`` only depends on a subset of past components.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        horizon: int,
        max_lag: int = 1,
        adjacency_sparsity: float = 0.3,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.max_lag = max_lag
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.adjacency = self._sample_dag(adjacency_sparsity)

    def _sample_dag(self, sparsity: float) -> Tensor:
        # Upper-triangular adjacency ensures acyclicity for contemporaneous.
        adj = torch.rand(self.observed_dim, self.observed_dim, generator=self.rng)
        mask = torch.rand_like(adj) < sparsity
        adj = adj * mask * torch.triu(torch.ones_like(adj), diagonal=1)
        return adj.unsqueeze(-1).expand(-1, -1, self.max_lag)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        tscm = TSCM(
            observed_dim=self.observed_dim,
            max_lag=self.max_lag,
            mechanisms=[
                lambda p, e, i=i: self._var_component(p, e, i)  # type: ignore[misc]
                for i in range(self.observed_dim)
            ],
            noise_sampler=lambda _n, _h, _d: torch.randn(
                _n, _h, _d, generator=self.rng
            )
            * self.noise_std,
            adjacency=self.adjacency,
        )
        x = tscm.sample(n_samples, horizon)
        return Batch(x=x)

    def _var_component(self, parents: Tensor, eps: Tensor, i: int) -> Tensor:
        # parents: (B, p, D); project through lagged adjacency.
        contrib = torch.zeros(parents.shape[0])
        for lag in range(self.max_lag):
            contrib = contrib + (parents[:, lag] @ self.adjacency[:, :, lag].T)[:, i]
        return contrib + eps[:, i]


class LTSCMGenerator(BaseDataGenerator):
    """Latent TSCM with a linear mixing decoder.

    Generates latent causal variables from a random latent DAG and mixes them
    with a learnable/invertible linear decoder ``x = W z + η``.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        horizon: int,
        max_lag: int = 1,
        adjacency_sparsity: float = 0.3,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.latent_dim = latent_dim
        self.max_lag = max_lag
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.latent_adjacency = self._sample_dag(adjacency_sparsity)
        self.decoder_weight = torch.randn(
            observed_dim, latent_dim, generator=self.rng
        )

    def _sample_dag(self, sparsity: float) -> Tensor:
        adj = torch.rand(self.latent_dim, self.latent_dim, generator=self.rng)
        mask = torch.rand_like(adj) < sparsity
        adj = adj * mask * torch.triu(torch.ones_like(adj), diagonal=1)
        return adj.unsqueeze(-1).expand(-1, -1, self.max_lag)

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        return_latents: bool = True,
        **kwargs: Any,
    ) -> Batch:
        horizon = horizon or self.horizon

        def mech(parents: Tensor, eps: Tensor, _context: Tensor | None, i: int) -> Tensor:
            out = torch.zeros(parents.shape[0])
            for lag in range(self.max_lag):
                out = out + (parents[:, lag] @ self.latent_adjacency[:, :, lag].T)[:, i]
            return out + eps[:, i]

        def decoder(z: Tensor, eta: Tensor) -> Tensor:
            return z @ self.decoder_weight.T + eta

        ltscm = LTSCM(
            latent_dim=self.latent_dim,
            observed_dim=self.observed_dim,
            max_lag=self.max_lag,
            mechanisms=[
                lambda p, e, c, i=i: mech(p, e, c, i)  # type: ignore[misc]
                for i in range(self.latent_dim)
            ],
            decoder=decoder,
            noise_sampler=lambda _n, _h, _d: torch.randn(
                _n, _h, _d, generator=self.rng
            )
            * self.noise_std,
            adjacency=self.latent_adjacency,
        )
        x, z = ltscm.sample(n_samples, horizon, return_latents=True)
        return Batch(x=x, z=z, adjacency=self.latent_adjacency)


def _frozen_mlp(
    in_dim: int,
    out_dim: int,
    *,
    hidden: int = 64,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """A fixed random MLP used as a ground-truth nonlinear mixing function."""
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
    for param in net.parameters():
        param.requires_grad_(False)
    net.eval()
    return net


class NonlinearICAGenerator(BaseDataGenerator):
    r"""Non-stationary nonlinear ICA data for iVAE-style methods.

    Latent factors follow regime-conditioned Gaussians
    ``z_i | u ~ N(mu_{u,i}, exp(logvar_{u,i}))`` and are mixed through a fixed
    nonlinear MLP ``x = g(z) + eta``. This is the canonical iVAE benchmark:
    the regime index ``u`` modulates the prior, which breaks the rotation
    ambiguity.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        horizon: int = 1,
        n_regimes: int = 3,
        hidden: int = 64,
        noise_std: float = 0.05,
        mixing: str = "nonlinear",
        shift_mean: bool = True,
        standardize: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if mixing not in {"linear", "nonlinear"}:
            raise ValueError("mixing must be 'linear' or 'nonlinear'.")
        self.latent_dim = latent_dim
        self.n_regimes = n_regimes
        self.noise_std = noise_std
        self.mixing_type = mixing
        self.standardize = standardize
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.regime_means = torch.randn(n_regimes, latent_dim, generator=self.rng)
        if not shift_mean:
            self.regime_means = torch.zeros_like(self.regime_means)
        logvars = torch.empty(n_regimes, latent_dim)
        for r in range(n_regimes):
            logvars[r] = -1.0 + 2.0 * (r / max(n_regimes - 1, 1))
        self.regime_logvars = logvars
        self.linear_weight = torch.randn(observed_dim, latent_dim, generator=self.rng)
        self.mixing = _frozen_mlp(
            latent_dim, observed_dim, hidden=hidden, generator=self.rng
        )

    def _mix(self, z: Tensor) -> Tensor:
        if self.mixing_type == "linear":
            return z @ self.linear_weight.T
        return cast(Tensor, self.mixing(z))

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        u = torch.randint(0, self.n_regimes, (n_samples,), generator=self.rng)
        mean = self.regime_means[u].unsqueeze(1)
        logvar = self.regime_logvars[u].unsqueeze(1)
        eps = torch.randn(n_samples, horizon, self.latent_dim, generator=self.rng)
        z = mean + eps * torch.exp(0.5 * logvar)
        with torch.no_grad():
            x = self._mix(z.reshape(-1, self.latent_dim)).reshape(
                n_samples, horizon, self.observed_dim
            )
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        if self.standardize:
            x = (x - x.mean(dim=(0, 1), keepdim=True)) / (
                x.std(dim=(0, 1), keepdim=True) + 1e-6
            )
        context = {"u": torch.nn.functional.one_hot(u, self.n_regimes).float()}
        return Batch(x=x, z=z, context=context)


class TemporalNonlinearGenerator(BaseDataGenerator):
    r"""Temporal nonlinear ICA with optional regime-dependent noise.

    Latent dynamics follow a lagged linear mechanism on an upper-triangular
    (acyclic) adjacency, mixed through a fixed nonlinear MLP. When
    ``n_regimes > 1`` the process noise scale depends on the regime, giving
    the non-stationarity LEAP exploits.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        horizon: int,
        max_lag: int = 1,
        n_regimes: int = 1,
        regime_mode: str = "trajectory",
        mixing: str = "mlp",
        obs_shift: bool = False,
        adjacency_sparsity: float = 0.4,
        mechanism_scale: float = 0.7,
        noise_std: float = 0.1,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if regime_mode not in {"trajectory", "time"}:
            raise ValueError("regime_mode must be 'trajectory' or 'time'.")
        if mixing not in {"mlp", "linear"}:
            raise ValueError("mixing must be 'mlp' or 'linear'.")
        self.latent_dim = latent_dim
        self.max_lag = max_lag
        self.n_regimes = n_regimes
        self.regime_mode = regime_mode
        self.mixing_type = mixing
        self.obs_shift = obs_shift
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        adj = torch.rand(latent_dim, latent_dim, generator=self.rng)
        mask = (torch.rand_like(adj) < adjacency_sparsity).float()
        adj = adj * mask * torch.triu(torch.ones_like(adj), diagonal=1)
        self.latent_adjacency = (adj * mechanism_scale).unsqueeze(-1).expand(
            -1, -1, max_lag
        )
        self.regime_noise = torch.linspace(0.5, 1.5, max(n_regimes, 1))
        self.linear_weight = torch.randn(observed_dim, latent_dim, generator=self.rng)
        self.obs_scale = 0.5 + torch.rand(n_regimes, observed_dim, generator=self.rng)
        self.obs_bias = torch.randn(n_regimes, observed_dim, generator=self.rng)
        self.mixing = _frozen_mlp(latent_dim, observed_dim, hidden=hidden, generator=self.rng)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        p = self.max_lag
        z = torch.zeros(n_samples, horizon + p, self.latent_dim)
        z[:, :p] = torch.randn(n_samples, p, self.latent_dim, generator=self.rng)
        if self.n_regimes > 1:
            if self.regime_mode == "time":
                u = torch.randint(0, self.n_regimes, (n_samples, horizon), generator=self.rng)
                scale = self.regime_noise[u] * self.noise_std  # (N, T)
            else:
                u = torch.randint(0, self.n_regimes, (n_samples,), generator=self.rng)
                scale = (self.regime_noise[u] * self.noise_std).unsqueeze(1)
        else:
            u = torch.zeros(n_samples, dtype=torch.long)
            scale = torch.full((n_samples, 1), self.noise_std)
        for t in range(p, horizon + p):
            parents = z[:, t - p : t]
            contrib = torch.zeros(n_samples, self.latent_dim)
            for lag in range(p):
                contrib = contrib + parents[:, lag] @ self.latent_adjacency[:, :, lag].T
            eps = torch.randn(n_samples, self.latent_dim, generator=self.rng)
            step_scale = (
                scale[:, 0] if scale.shape[1] == 1 else scale[:, t - p]
            ).unsqueeze(-1)
            z[:, t] = contrib + eps * step_scale
        z_obs = z[:, p:]
        # Standardise each latent dimension (a component-wise invertible transform
        # that preserves identifiability) so the benchmark is well-conditioned:
        # unit-variance latents and observation noise that is small relative to
        # the signal, matching the controlled synthetic setups of the papers.
        z_obs = (z_obs - z_obs.mean(dim=(0, 1), keepdim=True)) / (
            z_obs.std(dim=(0, 1), keepdim=True) + 1e-6
        )
        with torch.no_grad():
            if self.mixing_type == "linear":
                x = z_obs @ self.linear_weight.T
            else:
                x = cast(
                    Tensor, self.mixing(z_obs.reshape(-1, self.latent_dim))
                ).reshape(n_samples, horizon, self.observed_dim)
        if self.obs_shift:
            regime_index = u if u.ndim == 1 else u[:, 0]
            x = x * self.obs_scale[regime_index].unsqueeze(1) + self.obs_bias[
                regime_index
            ].unsqueeze(1)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        context = {"u": torch.nn.functional.one_hot(u, max(self.n_regimes, 1)).float()}
        return Batch(x=x, z=z_obs, context=context, adjacency=self.latent_adjacency)


class InterventionalTemporalGenerator(BaseDataGenerator):
    r"""Temporal causal process with recorded intervention targets (CITRIS-style).

    The latent space is partitioned into ``n_vars`` causal variable groups
    (each of size ``var_dim``) plus one shared group. At each time step a
    variable is intervened with probability ``intervene_prob``; intervened
    groups are resampled independently of their parents and the binary target
    vector ``I_t`` is recorded in ``context["I"]``. Observations are produced
    by a fixed invertible mixing (when ``observed_dim == latent_dim``).
    """

    def __init__(
        self,
        *,
        n_vars: int,
        var_dim: int,
        n_shared: int,
        horizon: int,
        observed_dim: int | None = None,
        intervene_prob: float = 0.3,
        mechanism_scale: float = 0.7,
        noise_std: float = 0.1,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        latent_dim = n_vars * var_dim + n_shared
        observed_dim = observed_dim or latent_dim
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.n_vars = n_vars
        self.var_dim = var_dim
        self.n_shared = n_shared
        self.latent_dim = latent_dim
        self.intervene_prob = intervene_prob
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.mechanisms = nn.ModuleList(
            [
                _frozen_mlp(latent_dim, var_dim, hidden=hidden, generator=self.rng)
                for _ in range(n_vars)
            ]
        )
        self.shared_mechanism = _frozen_mlp(
            latent_dim, n_shared, hidden=hidden, generator=self.rng
        )
        self.mechanism_scale = mechanism_scale
        if observed_dim == latent_dim:
            self.mixing = random_invertible_mixing(latent_dim, generator=self.rng)
        else:
            self.mixing = _frozen_mlp(
                latent_dim, observed_dim, hidden=hidden, generator=self.rng
            )

    def _groups(self) -> list[slice]:
        groups = [
            slice(i * self.var_dim, (i + 1) * self.var_dim) for i in range(self.n_vars)
        ]
        groups.append(slice(self.n_vars * self.var_dim, self.latent_dim))
        return groups

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        groups = self._groups()
        z = torch.zeros(n_samples, horizon, self.latent_dim)
        z[:, 0] = torch.randn(n_samples, self.latent_dim, generator=self.rng)
        targets = torch.zeros(n_samples, horizon, self.n_vars)
        for t in range(1, horizon):
            draw = torch.rand(n_samples, self.n_vars, generator=self.rng)
            targets[:, t] = (draw < self.intervene_prob).float()
            parents = z[:, t - 1]
            for i, group in enumerate(groups[:-1]):
                intervened = targets[:, t, i].bool()
                mean = self.mechanism_scale * self.mechanisms[i](parents)
                noise = self.noise_std * torch.randn(n_samples, self.var_dim, generator=self.rng)
                fresh = self.noise_std * torch.randn(n_samples, self.var_dim, generator=self.rng)
                updated = mean + noise
                updated[intervened] = fresh[intervened]
                z[:, t, group] = updated
            shared_group = groups[-1]
            mean_shared = self.mechanism_scale * self.shared_mechanism(parents)
            z[:, t, shared_group] = mean_shared + self.noise_std * torch.randn(
                n_samples, self.n_shared, generator=self.rng
            )
        with torch.no_grad():
            x = cast(Tensor, self.mixing(z.reshape(-1, self.latent_dim))).reshape(
                n_samples, horizon, self.observed_dim
            )
        x = x + 0.5 * self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, context={"I": targets})


class SlowFeatureGenerator(BaseDataGenerator):
    r"""Slow-feature source-separation data for Slow Flows.

    Latent sources follow a random walk with i.i.d. Gaussian increments
    ``z_t = z_{t-1} + eps_t`` (so the temporal-difference prior of Slow Flows is
    correctly specified), and are observed through a component-wise monotone
    mixing ``x_i = z_i + a * tanh(z_i)``. The mixing is invertible, so a
    well-trained flow recovers the sources up to permutation and scaling.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        horizon: int,
        increment_std: float = 1.0,
        nonlinearity: float = 0.5,
        observation_noise: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.increment_std = increment_std
        self.nonlinearity = nonlinearity
        self.observation_noise = observation_noise
        self.rng = torch.Generator().manual_seed(seed or 0)
        # z_t = z_{t-1} + eps_t: each source depends only on its own past.
        self.latent_adjacency = torch.eye(observed_dim).unsqueeze(-1)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        increments = self.increment_std * torch.randn(
            n_samples, horizon, self.observed_dim, generator=self.rng
        )
        increments[:, 0] = torch.randn(
            n_samples, self.observed_dim, generator=self.rng
        )
        z = torch.cumsum(increments, dim=1)
        x = z + self.nonlinearity * torch.tanh(z)
        if self.observation_noise:
            x = x + self.observation_noise * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, adjacency=self.latent_adjacency)


class LinearSDEGenerator(BaseDataGenerator):
    r"""Discrete-time sampling of a stable linear SDE (Ornstein–Uhlenbeck).

    ``x_{t+1} = x_t + A x_t \Delta t + G \varepsilon_t`` with a stable drift
    matrix ``A`` (negative-definite symmetric part) and Gaussian diffusion
    ``G``. ``z = x`` because the process is directly observed; used by CEGEN and
    other observed-space SDE methods.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        horizon: int,
        drift_scale: float = 0.3,
        noise_std: float = 0.5,
        dt: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.dt = dt
        self.rng = torch.Generator().manual_seed(seed or 0)
        skew = torch.randn(observed_dim, observed_dim, generator=self.rng)
        skew = 0.1 * (skew - skew.T)
        self.drift = -drift_scale * torch.eye(observed_dim) + skew
        self.diffusion = noise_std * torch.randn(
            observed_dim, observed_dim, generator=self.rng
        )

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        x = torch.zeros(n_samples, horizon, self.observed_dim)
        x[:, 0] = torch.randn(n_samples, self.observed_dim, generator=self.rng)
        for t in range(1, horizon):
            noise = torch.randn(n_samples, self.observed_dim, generator=self.rng)
            drift = x[:, t - 1] @ self.drift.T
            diffusion = noise @ self.diffusion.T
            x[:, t] = x[:, t - 1] + drift * self.dt + diffusion
        return Batch(x=x, z=x)


class TDRLGenerator(BaseDataGenerator):
    r"""Multivariate time series matching the TDRL generative model (Eq. 1).

    The latent state has three blocks: ``z^fix`` (fixed transition shared across
    domains), ``z^chg`` (transition modulated by a domain factor
    :math:`\theta^{dyn}_r`), and ``z^obs`` (observation-shift block generated
    from :math:`\theta^{obs}_r`, independent of the past). Observations are a
    fixed nonlinear mixture ``x = g(z)``.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        fix_dim: int,
        chg_dim: int,
        obs_dim: int,
        horizon: int,
        n_regimes: int,
        hidden: int = 64,
        mechanism_scale: float = 0.7,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        latent_dim = fix_dim + chg_dim + obs_dim
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.fix_dim = fix_dim
        self.chg_dim = chg_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.n_regimes = n_regimes
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.w_fix = mechanism_scale * torch.randn(fix_dim, latent_dim, generator=self.rng)
        self.w_chg = mechanism_scale * torch.randn(
            n_regimes, chg_dim, latent_dim, generator=self.rng
        )
        self.obs_mean = torch.randn(n_regimes, obs_dim, generator=self.rng)
        self.obs_logvar = torch.linspace(-1.0, 1.0, n_regimes).unsqueeze(1).expand(
            n_regimes, obs_dim
        ).clone()
        self.mixing = _frozen_mlp(latent_dim, observed_dim, hidden=hidden, generator=self.rng)
        # fix/changing blocks depend on all past latents; observation block does not.
        block = torch.zeros(latent_dim, latent_dim)
        block[: fix_dim + chg_dim, :] = 1.0
        self.latent_adjacency = block.unsqueeze(-1)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        u = torch.randint(0, self.n_regimes, (n_samples,), generator=self.rng)
        z = torch.zeros(n_samples, horizon, self.latent_dim)
        z[:, 0] = torch.randn(n_samples, self.latent_dim, generator=self.rng)
        for t in range(1, horizon):
            previous = z[:, t - 1]
            z_fix = previous @ self.w_fix.T
            z_chg = torch.einsum("bij,bj->bi", self.w_chg[u], previous)
            z_obs = self.obs_mean[u] + torch.exp(0.5 * self.obs_logvar[u]) * torch.randn(
                n_samples, self.obs_dim, generator=self.rng
            )
            noise = self.noise_std * torch.randn(n_samples, self.latent_dim, generator=self.rng)
            z[:, t] = torch.cat([z_fix, z_chg, z_obs], dim=-1) + noise
        # Standardise latents: unit-variance factors make the benchmark
        # well-conditioned without affecting identifiability.
        z = (z - z.mean(dim=(0, 1), keepdim=True)) / (z.std(dim=(0, 1), keepdim=True) + 1e-6)
        with torch.no_grad():
            x = cast(Tensor, self.mixing(z.reshape(-1, self.latent_dim))).reshape(
                n_samples, horizon, self.observed_dim
            )
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        context = {"u": torch.nn.functional.one_hot(u, self.n_regimes).float()}
        return Batch(x=x, z=z, context=context, adjacency=self.latent_adjacency)


class NCTRLGenerator(BaseDataGenerator):
    r"""Temporal process with a first-order Markov regime process (NCTRL).

    Regimes ``r_t`` follow a Markov chain with transition matrix ``P`` and the
    latent transition is regime-dependent:
    :math:`z_t = A_{r_t} z_{t-1} + \epsilon_t`, mixed by a fixed nonlinear map.
    Regimes are latent; ``context["r"]`` is provided only for reference.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        horizon: int,
        n_regimes: int,
        mechanism_scale: float = 0.7,
        regime_skew: float = 0.7,
        noise_std: float = 0.1,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.latent_dim = latent_dim
        self.n_regimes = n_regimes
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.transition_matrices = mechanism_scale * torch.randn(
            n_regimes, latent_dim, latent_dim, generator=self.rng
        )
        logits = 0.5 * torch.randn(n_regimes, n_regimes, generator=self.rng)
        self.regime_transition = torch.softmax(logits, dim=-1)
        self.regime_skew = regime_skew
        self.mixing = _frozen_mlp(latent_dim, observed_dim, hidden=hidden, generator=self.rng)

    def _sample_regimes(self, n_samples: int, horizon: int) -> Tensor:
        regimes = torch.zeros(n_samples, horizon, dtype=torch.long)
        regimes[:, 0] = torch.randint(0, self.n_regimes, (n_samples,), generator=self.rng)
        for t in range(1, horizon):
            probs = self.regime_transition[regimes[:, t - 1]]
            regimes[:, t] = torch.multinomial(probs, 1, generator=self.rng).squeeze(-1)
        return regimes

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        regimes = self._sample_regimes(n_samples, horizon)
        z = torch.zeros(n_samples, horizon, self.latent_dim)
        z[:, 0] = torch.randn(n_samples, self.latent_dim, generator=self.rng)
        for t in range(1, horizon):
            matrices = self.transition_matrices[regimes[:, t]]
            drift = torch.einsum("bij,bj->bi", matrices, z[:, t - 1])
            noise = self.noise_std * torch.randn(n_samples, self.latent_dim, generator=self.rng)
            z[:, t] = drift + noise
        # Unit-variance latents for a well-conditioned benchmark.
        z = (z - z.mean(dim=(0, 1), keepdim=True)) / (z.std(dim=(0, 1), keepdim=True) + 1e-6)
        with torch.no_grad():
            x = cast(Tensor, self.mixing(z.reshape(-1, self.latent_dim))).reshape(
                n_samples, horizon, self.observed_dim
            )
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z, context={"r": regimes})


class Causal3DIdentGenerator(BaseDataGenerator):
    r"""Temporal Causal3DIdent-like benchmark with known visual factors.

    A low-dimensional surrogate for the Causal3DIdent family: ``latent_dim``
    continuous factors (object position, rotation, hue, spotlight, \dots) evolve
    through a lagged acyclic mechanism and are rendered through a non-linear
    observation map. When ``observed_dim == latent_dim`` and ``mixing ==
    "invertible"`` the render is a fixed invertible flow, so latent recovery is
    well posed; with ``observed_dim > latent_dim`` the render is non-invertible
    (for methods that relax invertibility). Ground-truth factors ``z`` and the
    latent adjacency are returned for MCC/R2/SHD.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        horizon: int,
        max_lag: int = 1,
        adjacency_sparsity: float = 0.4,
        mixing: str = "invertible",
        mechanism_scale: float = 0.7,
        noise_std: float = 0.1,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if mixing not in {"invertible", "nonlinear"}:
            raise ValueError("mixing must be 'invertible' or 'nonlinear'.")
        if mixing == "invertible" and observed_dim != latent_dim:
            raise ValueError("invertible mixing requires observed_dim == latent_dim.")
        self.latent_dim = latent_dim
        self.max_lag = max_lag
        self.mixing_type = mixing
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        adj = torch.rand(latent_dim, latent_dim, generator=self.rng)
        mask = (torch.rand_like(adj) < adjacency_sparsity).float()
        adj = adj * mask * torch.triu(torch.ones_like(adj), diagonal=1)
        self.latent_adjacency = (adj * mechanism_scale).unsqueeze(-1).expand(
            -1, -1, max_lag
        )
        if mixing == "invertible":
            self.mixing: nn.Module = random_invertible_mixing(
                latent_dim, generator=self.rng
            )
        else:
            self.mixing = _frozen_mlp(
                latent_dim, observed_dim, hidden=hidden, generator=self.rng
            )

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        p = self.max_lag
        z = torch.zeros(n_samples, horizon + p, self.latent_dim)
        z[:, :p] = torch.randn(n_samples, p, self.latent_dim, generator=self.rng)
        for t in range(p, horizon + p):
            parents = z[:, t - p : t]
            contrib = torch.zeros(n_samples, self.latent_dim)
            for lag in range(p):
                contrib = contrib + parents[:, lag] @ self.latent_adjacency[:, :, lag].T
            eps = torch.randn(n_samples, self.latent_dim, generator=self.rng)
            z[:, t] = contrib + eps * self.noise_std
        z_obs = z[:, p:]
        with torch.no_grad():
            x = cast(Tensor, self.mixing(z_obs.reshape(-1, self.latent_dim))).reshape(
                n_samples, horizon, self.observed_dim
            )
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        return Batch(x=x, z=z_obs, adjacency=self.latent_adjacency)


class CartPoleGenerator(BaseDataGenerator):
    r"""Regime-changing CartPole trajectories with known latent physical state.

    Integrates the standard cart-pole dynamics under random actions, with
    per-trajectory regime parameters (force magnitude, pole length and masses).
    The latent state is the physical state
    :math:`(x, \dot x, \theta, \dot\theta)`; observations are either the state
    itself or a fixed non-linear render. No latent causal graph is defined, so
    ``adjacency`` is ``None`` and SHD metrics do not apply.
    """

    def __init__(
        self,
        *,
        horizon: int,
        observed_dim: int = 4,
        n_regimes: int = 3,
        dt: float = 0.02,
        gravity: float = 9.8,
        mixing: str = "state",
        noise_std: float = 0.05,
        hidden: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        if mixing not in {"state", "nonlinear"}:
            raise ValueError("mixing must be 'state' or 'nonlinear'.")
        if mixing == "state" and observed_dim != 4:
            raise ValueError("state observations require observed_dim == 4.")
        self.n_regimes = n_regimes
        self.dt = dt
        self.gravity = gravity
        self.noise_std = noise_std
        self.mixing_type = mixing
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.force_mag = 5.0 + 5.0 * torch.rand(n_regimes, generator=self.rng)
        self.pole_length = 0.5 + 0.5 * torch.rand(n_regimes, generator=self.rng)
        self.mass_cart = 0.5 + 0.5 * torch.rand(n_regimes, generator=self.rng)
        self.mass_pole = 0.05 + 0.1 * torch.rand(n_regimes, generator=self.rng)
        if mixing == "nonlinear":
            self.mixing: nn.Module | None = _frozen_mlp(
                4, observed_dim, hidden=hidden, generator=self.rng
            )
        else:
            self.mixing = None

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        u = torch.randint(0, self.n_regimes, (n_samples,), generator=self.rng)
        force_mag = self.force_mag[u]
        length = self.pole_length[u]
        mass_cart = self.mass_cart[u]
        mass_pole = self.mass_pole[u]
        state = torch.zeros(n_samples, horizon, 4)
        state[:, 0] = 0.05 * torch.randn(n_samples, 4, generator=self.rng)
        actions = torch.where(
            torch.rand(n_samples, horizon, generator=self.rng) < 0.5, 1.0, -1.0
        )
        for t in range(1, horizon):
            x = state[:, t - 1, 0]
            x_dot = state[:, t - 1, 1]
            theta = state[:, t - 1, 2]
            theta_dot = state[:, t - 1, 3]
            cos = torch.cos(theta)
            sin = torch.sin(theta)
            force = actions[:, t - 1] * force_mag
            temp = (force + mass_pole * length * theta_dot**2 * sin) / (
                mass_cart + mass_pole
            )
            theta_acc = (self.gravity * sin - cos * temp) / (
                length * (4.0 / 3.0 - mass_pole * cos**2 / (mass_cart + mass_pole))
            )
            x_acc = temp - mass_pole * length * theta_acc * cos / (
                mass_cart + mass_pole
            )
            state[:, t, 0] = (x + self.dt * x_dot).clamp(-5.0, 5.0)
            state[:, t, 1] = (x_dot + self.dt * x_acc).clamp(-10.0, 10.0)
            state[:, t, 2] = (theta + self.dt * theta_dot).clamp(-math.pi, math.pi)
            state[:, t, 3] = (theta_dot + self.dt * theta_acc).clamp(-15.0, 15.0)
        with torch.no_grad():
            if self.mixing is None:
                x = state
            else:
                x = cast(
                    Tensor, self.mixing(state.reshape(-1, 4))
                ).reshape(n_samples, horizon, self.observed_dim)
        x = x + self.noise_std * torch.randn(x.shape, generator=self.rng)
        context = {
            "u": torch.nn.functional.one_hot(u, self.n_regimes).float(),
        }
        return Batch(x=x, z=state, context=context)
