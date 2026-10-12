"""CausalConceptTS: causal concept-based explanations of time-series classifiers.

Reference: Lopez Alcaraz & Strodthoff, *Explaining Time Series Classification
Predictions via Causal Attributions*, 2025 — BibTeX
``j.m.lopezalcarazExplainingTimeSeries2025``.

Concepts are predefined, non-overlapping time segments. A diffusion (denoising)
imputer fills in a concept under a target class state, and the concept's causal
individual treatment effect is the difference in the classifier's log-odds when
the concept is imputed under the factual versus the reference class state,

    ITE(X, f, c, CS*, CS0) =
        log2 E[f(X | do(CS = CS*))] - log2 E[f(X | do(CS = CS0))].

The returned object is an explanation (per-concept attributions). There is no
identifiability theorem; attributions are external-validity evidence.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.core.backbones import MLP
from gcmts.explanation.base import BaseExplainer
from gcmts.typing import Batch

logger = logging.getLogger(__name__)

__all__ = ["CausalConceptTS"]


class CausalConceptTS(BaseExplainer):
    """Concept-based causal attributions via masked diffusion imputation.

    Args:
        n_channels: number of channels ``D``.
        horizon: sequence length ``T``.
        n_concepts: number ``C`` of predefined concept segments.
        n_classes: number of classifier classes.
        hidden_dims: network widths.
        cls_weight / impute_weight: loss weights.
        n_imputations: Monte-Carlo imputations per concept at explanation time.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        horizon: int,
        n_concepts: int = 4,
        n_classes: int = 2,
        hidden_dims: tuple[int, ...] = (128, 128),
        cls_weight: float = 1.0,
        impute_weight: float = 1.0,
        n_imputations: int = 8,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            horizon=horizon,
            n_concepts=n_concepts,
            n_classes=n_classes,
        )
        self.cls_weight = cls_weight
        self.impute_weight = impute_weight
        self.n_imputations = n_imputations
        flat = n_channels * horizon
        self.classifier = MLP(flat, n_classes, hidden_dims=hidden_dims)
        self.imputer = MLP(
            flat + n_classes + n_concepts, flat, hidden_dims=hidden_dims
        )
        self.register_buffer("concept_ids", self._concept_ids(horizon, n_concepts))
        self.concept_ids: Tensor
        logger.debug(
            "Initialised CausalConceptTS (D=%d, T=%d, C=%d)",
            n_channels,
            horizon,
            n_concepts,
        )

    @staticmethod
    def _concept_ids(horizon: int, n_concepts: int) -> Tensor:
        width = max(1, horizon // n_concepts)
        ids = torch.arange(horizon) // width
        ids = ids.clamp(max=n_concepts - 1)
        return ids.long()

    def _mask_channel(self, x: Tensor, concept: Tensor) -> Tensor:
        """Set the time steps of ``concept`` to noise (per-sample concept id)."""
        mask = (self.concept_ids.unsqueeze(0) == concept.unsqueeze(1))  # (B, T)
        mask = mask.unsqueeze(-1)
        noise = torch.randn_like(x)
        return torch.where(mask, noise, x)

    def _impute(self, masked: Tensor, class_label: Tensor, concept: Tensor) -> Tensor:
        batch = masked.shape[0]
        class_onehot = F.one_hot(class_label.long(), self.n_classes).float()
        concept_onehot = F.one_hot(concept.long(), self.n_concepts).float()
        features = torch.cat([masked.reshape(batch, -1), class_onehot, concept_onehot], dim=-1)
        return self.imputer(features).reshape_as(masked)

    # -- forward / loss ------------------------------------------------------

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        x = batch.x
        batch_size = x.shape[0]
        labels = self._labels(batch)
        logits = self.classifier(x.reshape(batch_size, -1))
        concept = torch.randint(0, self.n_concepts, (batch_size,), device=x.device)
        masked = self._mask_channel(x, concept)
        reconstruction = self._impute(masked, labels, concept)
        return {
            "logits": logits,
            "reconstruction": reconstruction,
            "target": x,
            "concept": concept,
            "labels": labels,
        }

    def _labels(self, batch: Batch) -> Tensor:
        if batch.context is not None and "y" in batch.context:
            value = batch.context["y"]
            return value.reshape(-1)
        return torch.zeros(batch.x.shape[0], device=batch.x.device)

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        labels = outputs["labels"].long()
        classification = F.cross_entropy(outputs["logits"], labels)
        impute = F.mse_loss(outputs["reconstruction"], outputs["target"])
        total = self.cls_weight * classification + self.impute_weight * impute
        return {"loss": total, "classification": classification, "impute": impute}

    # -- explanation ---------------------------------------------------------

    @torch.no_grad()
    def explain(self, batch: Batch) -> dict[str, Tensor]:
        """Per-concept causal individual treatment effects ``(B, C)``."""
        x = batch.x
        batch_size = x.shape[0]
        labels = self._labels(batch)
        attributions = torch.zeros(batch_size, self.n_concepts, device=x.device)
        for concept in range(self.n_concepts):
            concept_ids = torch.full((batch_size,), concept, device=x.device, dtype=torch.long)
            masked = self._mask_channel(x, concept_ids)
            factual = torch.zeros(batch_size, device=x.device)
            reference = torch.zeros(batch_size, device=x.device)
            for _ in range(self.n_imputations):
                imputed_f = self._impute(masked, labels, concept_ids)
                imputed_r = self._impute(masked, 1 - labels, concept_ids)
                logits_f = self.classifier(imputed_f.reshape(batch_size, -1))
                logits_r = self.classifier(imputed_r.reshape(batch_size, -1))
                factual = factual + F.log_softmax(logits_f, dim=-1)[
                    torch.arange(batch_size), labels
                ]
                reference = reference + F.log_softmax(logits_r, dim=-1)[
                    torch.arange(batch_size), labels
                ]
            attributions[:, concept] = (factual - reference) / self.n_imputations
        return {"attributions": attributions, "labels": labels}

    def identifiability_statement(self) -> str:
        return (
            "CausalConceptTS: concept attributions are concept-based do(CS) "
            "individual treatment effects estimated by class-conditional "
            "imputation; there is no identifiability theorem, and the explanation "
            "is external-validity evidence relative to the classifier."
        )
