"""Counterfactual and treatment-effect estimation for time series.

Methods here target Pearl levels L2 (interventions) and L3 (counterfactuals)
under time-varying treatments and latent confounders.

Key references:
* CaTSG — ``yutongxiaCausalTimeSeries2025``
* Wu-IPTW — ``wuCounterfactualGenerativeModels2024``
* LCD / CLIPR — ``larslorchLatentCausalDiffusions2026``
* PFD-BDCM — ``xinwenliuPartiallyFunctionalDynamic2025``
* CaPaint — ``yifanduanCausalDecipheringInpainting2024``
* LacaDM — ``xuemingyanLacaDMLatentCausal2025``
* CRN — ``bicaEstimatingCounterfactualTreatment2020``
* CausalTransformer — ``melnychukCausalTransformerEstimating2022``
* GANITE — ``yoonGANITEEstimationIndividualized2018``
* CEPAE — ``tomasgarrigaCEPAEConditionalEntropyPenalized2026``
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor

from gcmts.core.base import BaseModel
from gcmts.typing import Batch, CounterfactualOutput

__all__ = ["BaseEffectEstimator"]


class BaseEffectEstimator(BaseModel):
    """Abstract base for treatment-effect and counterfactual estimators.

    The canonical workflow follows Pearl's three steps:

    1. **Abduction** — infer exogenous noise / confounders from the factual
       trajectory and factual action.
    2. **Action** — replace the mechanisms of the treatment variables with the
       counterfactual action values.
    3. **Prediction** — roll the model forward holding the inferred noise fixed.

    Subclasses must implement the ``counterfactual`` method. Intervention-only
    methods (L2) may raise :class:`NotImplementedError` for ``counterfactual``
    and override :meth:`causal_level` accordingly.
    """

    def __init__(self, *, observed_dim: int, treatment_dim: int = 1) -> None:
        super().__init__(observed_dim=observed_dim, latent_dim=None)
        self.treatment_dim = treatment_dim

    @abstractmethod
    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        """Infer exogenous noise/confounders from ``batch``.

        Returns a dictionary of inferred noise tensors.
        """

    @abstractmethod
    def action(
        self,
        noise: dict[str, Tensor],
        intervention: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Apply the intervention to the inferred noise/mechanism context."""

    @abstractmethod
    def prediction(
        self,
        context: dict[str, Tensor],
        horizon: int,
    ) -> Tensor:
        """Generate a trajectory of length ``horizon`` under the new action."""

    @abstractmethod
    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        """Answer an L2 query: ``P(X_{1:T} | do(A_{1:T}=a_{1:T}))``."""

    @abstractmethod
    def counterfactual(
        self,
        batch: Batch,
        counterfactual_action: dict[str, Tensor],
    ) -> CounterfactualOutput:
        """Answer an L3 query using abduction–action–prediction."""

    def estimate_effect(
        self,
        batch: Batch,
        actions: list[dict[str, Tensor]],
        *,
        outcome_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Estimate causal effects across a list of actions.

        For each action ``a`` the model produces either an intervention sample
        (L2) or a counterfactual sample (L3, if ``batch`` contains the factual
        trajectory). Returns a tensor of expected outcomes.
        """
        outcomes = []
        for action in actions:
            cf = self.counterfactual(batch, action)
            y = cf.counterfactual if outcome_fn is None else outcome_fn(cf.counterfactual)
            outcomes.append(y.mean(0))
        return torch.stack(outcomes, dim=0)

    def causal_level(self) -> str:
        return "L3_counterfactual"
