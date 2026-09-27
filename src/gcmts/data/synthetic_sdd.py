r"""Synthetic benchmarks for static-dynamic disentanglement.

Both generators realise the time-aware SCM of Section~\ref{sec:sdd}: a latent
vector split into static causal ``z_stc``, dynamic causal ``z_dyc``, static
spurious ``z_sts`` and dynamic spurious ``z_dys`` factors. The target ``Y_t``
depends only on the causal pair, while a domain-varying confounder makes the
spurious factors predictive within a domain but harmful out of distribution.
Held-out domains therefore isolate whether a representation kept the causal
factors and discarded the spurious ones.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["MultiDomainGenerator", "EvolvingDomainGenerator"]


class MultiDomainGenerator(BaseDataGenerator):
    """Cross-domain SCM with discrete domains and held-out regimes.

    Latent groups: ``z_stc`` (``static_dim``), ``z_dyc`` (``dynamic_dim``),
    ``z_sts`` (``spurious_static_dim``), ``z_dys`` (``spurious_dynamic_dim``).
    The causal target ``Y_t`` depends on ``(z_stc, z_dyc,t)``; the spurious
    factors enter ``Y_t`` with a domain sign that flips on the held-out domain.
    """

    def __init__(
        self,
        *,
        static_dim: int = 2,
        dynamic_dim: int = 2,
        spurious_static_dim: int = 1,
        spurious_dynamic_dim: int = 1,
        observed_dim: int = 8,
        horizon: int = 12,
        n_domains: int = 5,
        noise_std: float = 0.05,
        seed: int | None = None,
    ) -> None:
        latent_dim = (
            static_dim + dynamic_dim + spurious_static_dim + spurious_dynamic_dim
        )
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.spurious_static_dim = spurious_static_dim
        self.spurious_dynamic_dim = spurious_dynamic_dim
        self.latent_dim = latent_dim
        self.n_domains = n_domains
        self.noise_std = noise_std
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.mixing = torch.randn(observed_dim, latent_dim, generator=self.rng)
        self.causal_coeff = 0.7 * torch.randn(
            dynamic_dim, static_dim + dynamic_dim, generator=self.rng
        )
        # Causal dynamics are domain-invariant, so the causal factors transfer
        # exactly; only the spurious correlation flips on the held-out domain.
        self.dynamic_scale = torch.ones(n_domains)
        # The spurious correlation is positive in most training domains but
        # flips sign on the last training domain and on the held-out domain, so
        # an average-risk predictor leans on it while an invariant one discards it.
        self.spurious_sign = torch.ones(n_domains)
        self.spurious_sign[-1] = -1.0
        self.spurious_sign[-2] = -1.0
        self.static_causal_weights = torch.randn(static_dim, generator=self.rng)

    def _roll(
        self,
        static: Tensor,
        spurious_static: Tensor,
        dynamic: Tensor,
        spurious_dynamic: Tensor,
        domain: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, horizon, _ = dynamic.shape
        dyc = [dynamic[:, 0]]
        dys = [spurious_dynamic[:, 0]]
        for _ in range(1, horizon):
            scale = self.dynamic_scale[domain].view(-1, 1)
            parents = torch.cat([static, dyc[-1]], dim=-1)
            dyc.append(scale * (parents @ self.causal_coeff.T) + 0.1 * torch.randn(
                batch, self.dynamic_dim, generator=self.rng
            ))
            dys.append(0.5 * dys[-1] + 0.2 * torch.randn(
                batch, self.spurious_dynamic_dim, generator=self.rng
            ))
        dynamic_seq = torch.stack(dyc, dim=1)
        spurious_dynamic_seq = torch.stack(dys, dim=1)
        latent = torch.cat(
            [static.unsqueeze(1).expand(-1, horizon, -1),
             dynamic_seq,
             spurious_static.unsqueeze(1).expand(-1, horizon, -1),
             spurious_dynamic_seq],
            dim=-1,
        )
        causal = static @ self.static_causal_weights + dynamic_seq.mean(dim=-1).sum(-1)
        spurious = spurious_dynamic_seq.mean(dim=(1, 2)) + spurious_static.sum(-1)
        outcome = causal + self.spurious_sign[domain] * 0.8 * spurious
        return latent, outcome

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        domains: list[int] | Tensor | None = None,
        **kwargs: Any,
    ) -> Batch:
        horizon = horizon or self.horizon
        if domains is None:
            domain = torch.randint(0, self.n_domains, (n_samples,), generator=self.rng)
        elif isinstance(domains, Tensor):
            domain = domains
        else:
            domain = torch.tensor(domains)[
                torch.randint(0, len(domains), (n_samples,), generator=self.rng)
            ]
        static = torch.randn(n_samples, self.static_dim, generator=self.rng)
        spurious_static = torch.randn(
            n_samples, self.spurious_static_dim, generator=self.rng
        )
        dynamic = torch.randn(n_samples, horizon, self.dynamic_dim, generator=self.rng)
        spurious_dynamic = torch.randn(
            n_samples, horizon, self.spurious_dynamic_dim, generator=self.rng
        )
        latent, outcome = self._roll(
            static, spurious_static, dynamic, spurious_dynamic, domain
        )
        observed = latent.reshape(-1, self.latent_dim) @ self.mixing.T
        observed = observed.reshape(n_samples, horizon, self.observed_dim)
        observed = observed + self.noise_std * torch.randn(
            observed.shape, generator=self.rng
        )
        return Batch(
            x=observed,
            z=latent,
            outcome=outcome,
            domain=domain,
            context={"domain": domain},
        )


class EvolvingDomainGenerator(MultiDomainGenerator):
    """Continuous-time analogue: the domain parameter drifts over the sequence.

    The spurious mechanism strength ramps with time, so the training window
    ``t < t_split`` and the held-out window ``t >= t_split`` have different
    spurious correlations; a causal representation transfers, a spurious one
    does not.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("n_domains", 6)
        super().__init__(**kwargs)

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        domains: list[int] | Tensor | None = None,
        **kwargs: Any,
    ) -> Batch:
        batch = super().sample(n_samples, horizon, domains=domains, **kwargs)
        if domains is None:
            ramp = torch.linspace(0.0, 1.0, batch.x.shape[1])
            if batch.outcome is not None:
                batch.outcome = batch.outcome * (0.5 + ramp.mean())
        return batch
