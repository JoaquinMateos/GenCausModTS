"""Evaluation metrics organised by model family and causal level."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import Tensor

from gcmts.core.graph_ops import aligned_shd, edge_auprc
from gcmts.core.solvers import bures_w2
from gcmts.core.utils import hungarian_match, mcc

__all__ = [
    "Metric",
    "MetricResult",
    "MetricRegistry",
    "MCCMetric",
    "R2DiagMetric",
    "R2SepMetric",
    "DCIDisentanglementMetric",
    "DCICompletenessMetric",
    "DCIInformativenessMetric",
    "SHDMetric",
    "WSHDMetric",
    "EdgeAUPRCMetric",
    "TripletDistanceMetric",
    "TwoSampleDiscriminatorMetric",
    "PEHEMetric",
    "ATEMetric",
    "PolicyRiskMetric",
    "CFMAEMetric",
    "MBEMetric",
    "MMD2Metric",
    "JFTSDMetric",
    "OODMSEMetric",
    "DomainAccuracyMetric",
    "MIGMetric",
    "CDSMetric",
]


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    level: str  # L1/L2/L3 or family tag
    higher_is_better: bool


class Metric(ABC):
    """A single evaluation metric with a ground-truth contract."""

    def __init__(self, name: str, level: str, *, higher_is_better: bool) -> None:
        self.name = name
        self.level = level
        self.higher_is_better = higher_is_better

    @abstractmethod
    def __call__(self, prediction: Any, target: Any) -> MetricResult:
        """Compute the metric from a prediction and ground-truth target."""

    def direction(self) -> int:
        return 1 if self.higher_is_better else -1


class MetricRegistry:
    """Central registry of metrics, indexed by family and causal level."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Metric '{metric.name}' already registered.")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        return self._metrics[name]

    def list_metrics(self, level: str | None = None) -> list[str]:
        if level is None:
            return list(self._metrics.keys())
        return [n for n, m in self._metrics.items() if m.level == level]

    def evaluate(
        self,
        names: list[str],
        prediction: Any,
        target: Any,
    ) -> list[MetricResult]:
        return [self._metrics[n](prediction, target) for n in names]


def _default_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register(MCCMetric())
    reg.register(R2DiagMetric())
    reg.register(R2SepMetric())
    reg.register(DCIDisentanglementMetric())
    reg.register(DCICompletenessMetric())
    reg.register(DCIInformativenessMetric())
    reg.register(SHDMetric())
    reg.register(WSHDMetric())
    reg.register(EdgeAUPRCMetric())
    reg.register(TripletDistanceMetric())
    reg.register(TwoSampleDiscriminatorMetric())
    reg.register(PEHEMetric())
    reg.register(ATEMetric())
    reg.register(PolicyRiskMetric())
    reg.register(CFMAEMetric())
    reg.register(MBEMetric())
    reg.register(MMD2Metric())
    reg.register(JFTSDMetric())
    reg.register(OODMSEMetric())
    reg.register(DomainAccuracyMetric())
    reg.register(MIGMetric())
    reg.register(CDSMetric())
    return reg


# --- Identifiable CRL metrics ------------------------------------------------

class MCCMetric(Metric):
    """Mean correlation coefficient after Hungarian matching."""

    def __init__(self) -> None:
        super().__init__("mcc", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        return MetricResult(self.name, mcc(target, prediction), self.level, self.higher_is_better)


def _nonlinear_r2(features: Tensor, target: Tensor, *, seed: int = 0) -> float:
    """Test-split R² predicting ``target`` from ``features``, as in CITRIS.

    Mirrors the coefficient-of-determination protocol of Lippe et al. (2022): a
    non-linear regressor predicts the true factor and R² is measured on held-out
    samples. Implemented with gradient boosting (which, unlike extremely
    randomised trees, is never systematically worse than a linear fit here); the
    reported value is the best of the linear and boosted fits so that it is a
    valid lower-bound-consistent R².
    """
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy().reshape(-1)
    if x.shape[0] < 40:
        return float("nan")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    linear = LinearRegression().fit(x_train, y_train)
    boosted = GradientBoostingRegressor(random_state=seed).fit(x_train, y_train)
    r2_linear = r2_score(y_test, linear.predict(x_test))
    r2_boosted = r2_score(y_test, boosted.predict(x_test))
    return float(max(0.0, r2_linear, r2_boosted))


def _r2_matrix(pred: Tensor, tgt: Tensor, *, seed: int = 0) -> tuple[np.ndarray, Tensor]:
    """Return the ``(K, K)`` matrix ``R²_ij`` (true ``i`` from estimated ``j``) and match."""
    pred = pred.reshape(-1, pred.shape[-1])
    tgt = tgt.reshape(-1, tgt.shape[-1])
    pred_std = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
    tgt_std = (tgt - tgt.mean(0)) / (tgt.std(0) + 1e-8)
    corr = (tgt_std.T @ pred_std) / tgt.shape[0]
    _, col = hungarian_match(-corr.abs())
    k = tgt.shape[-1]
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            matrix[i, j] = _nonlinear_r2(pred[:, j : j + 1], tgt[:, i], seed=seed)
    return matrix, col


class R2DiagMetric(Metric):
    """Diagonal R²: each true factor predicted from its matched latent block.

    ``R²_diag = (1/K) Σ_i R²(true_i, matched latent)`` with a non-linear
    regressor, as defined for multi-dimensional causal factors (Lippe et al.,
    2022). Matching uses the Hungarian assignment on correlations.
    """

    def __init__(self) -> None:
        super().__init__("r2_diag", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix, col = _r2_matrix(prediction, target)
        value = float(
            np.mean([matrix[i, int(col[i].item())] for i in range(matrix.shape[0])])
        )
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class R2SepMetric(Metric):
    """Separation R²: leakage of each true factor into non-matched latents.

    ``R²_sep = (1/K) Σ_i max_{j ≠ matched(i)} R²(true_i, latent_j)``; values near
    zero indicate that factors do not leak across latent dimensions.
    """

    def __init__(self) -> None:
        super().__init__("r2_sep", "L1_representation", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix, col = _r2_matrix(prediction, target)
        leakage = []
        for i in range(matrix.shape[0]):
            others = [matrix[i, j] for j in range(matrix.shape[1]) if j != col[i].item()]
            leakage.append(max(others) if others else 0.0)
        return MetricResult(self.name, float(np.mean(leakage)), self.level, self.higher_is_better)


def _importance_matrix(pred: Tensor, tgt: Tensor, *, seed: int = 0) -> np.ndarray:
    """Feature-importance matrix ``R_ij`` (importance of latent ``j`` for factor ``i``)."""
    x = pred.reshape(-1, pred.shape[-1]).detach().cpu().numpy()
    y = tgt.reshape(-1, tgt.shape[-1]).detach().cpu().numpy()
    k = x.shape[1]
    matrix = np.zeros((k, k))
    for i in range(k):
        regressor = GradientBoostingRegressor(random_state=seed)
        regressor.fit(x, y[:, i])
        matrix[i] = regressor.feature_importances_
    return matrix


def _normalised_entropy(row: np.ndarray) -> float:
    total = row.sum()
    if total <= 0:
        return 1.0
    probabilities = row / total
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return entropy / np.log(len(row)) if len(row) > 1 else 0.0


class DCIDisentanglementMetric(Metric):
    """DCI disentanglement: each latent depends on a single ground-truth factor."""

    def __init__(self) -> None:
        super().__init__("dci_disentanglement", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix = _importance_matrix(prediction, target)
        scores = [1.0 - _normalised_entropy(matrix[:, j]) for j in range(matrix.shape[1])]
        return MetricResult(self.name, float(np.mean(scores)), self.level, self.higher_is_better)


class DCICompletenessMetric(Metric):
    """DCI completeness: each ground-truth factor is captured by a single latent."""

    def __init__(self) -> None:
        super().__init__("dci_completeness", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix = _importance_matrix(prediction, target)
        scores = [1.0 - _normalised_entropy(matrix[i]) for i in range(matrix.shape[0])]
        return MetricResult(self.name, float(np.mean(scores)), self.level, self.higher_is_better)


class DCIInformativenessMetric(Metric):
    """DCI informativeness: predictive information the latents retain about factors."""

    def __init__(self) -> None:
        super().__init__("dci_informativeness", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        tgt = target.reshape(-1, target.shape[-1])
        pred = prediction.reshape(-1, prediction.shape[-1])
        scores = [_nonlinear_r2(pred, tgt[:, i]) for i in range(tgt.shape[-1])]
        return MetricResult(
            self.name, float(np.nanmean(scores)), self.level, self.higher_is_better
        )


class SHDMetric(Metric):
    """Structural Hamming Distance between estimated and true adjacency.

    The estimated graph is aligned to the ground truth over latent permutations
    before counting edge additions/deletions, since latents are identified only
    up to permutation. Accepts adjacency of shape ``(d, d)`` or ``(d, d, p)``.
    """

    def __init__(self) -> None:
        super().__init__("shd", "L1_structure", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        value = aligned_shd(prediction, target)
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class WSHDMetric(Metric):
    """Weighted SHD: structural differences weighted by edge magnitude.

    Uses the same permutation alignment as :class:`SHDMetric`; see
    :func:`gcmts.core.graph_ops.weighted_shd`.
    """

    def __init__(self) -> None:
        super().__init__("wshd", "L1_structure", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        value = aligned_shd(prediction, target, weighted=True)
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class EdgeAUPRCMetric(Metric):
    """Threshold-agnostic structural recovery: average precision of edge scores.

    Ranks the absolute estimated edge coefficients against the binarised
    ground-truth edges and returns the area under the precision--recall curve
    (CaDRe's threshold-agnostic structural metric). Higher is better and
    ``nan`` is returned when the ground truth has no edges or is complete.
    Unlike :class:`SHDMetric` this metric does not search over permutations;
    callers should apply the Hungarian assignment from
    :func:`gcmts.core.graph_ops.latent_permutation` first, as the shared CRL
    suites do.
    """

    def __init__(self) -> None:
        super().__init__("edge_auprc", "L1_structure", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        value = edge_auprc(prediction, target)
        return MetricResult(self.name, value, self.level, self.higher_is_better)


def _flatten_leading(tensor: Tensor) -> Tensor:
    """Flatten every leading dimension into a single sample axis."""
    if tensor.ndim < 2:
        raise ValueError("Expected at least a sample and a feature dimension.")
    return tensor.reshape(-1, tensor.shape[-1])


def _covariance(x: Tensor) -> Tensor:
    """Unbiased covariance of a ``(N, e)`` sample matrix."""
    centered = x - x.mean(0, keepdim=True)
    return (centered.T @ centered) / max(x.shape[0] - 1, 1)


class TripletDistanceMetric(Metric):
    r"""Triplet distance: isolation of an intervened factor.

    For an intervention on factor ``a`` the *matched* latent coordinate
    ``pi(a)`` should move, while every other latent coordinate stays fixed.
    With ``d_self = |z_before_{pi(a)} - z_after_{pi(a)}|`` and
    ``d_other = mean_{k != pi(a)} |z_before_k - z_after_k|`` the per-sample
    score is ``d_other / (d_self + d_other)``: zero when the intervention is
    perfectly isolated, about ``0.5`` when the change is spread uniformly
    (chance). It is reported as a distance, matching the direction of the
    source evaluation (CITRIS reports distances with optimum zero), and not as
    the isolation ratio of the earlier implementation.

    Inputs follow the CITRIS convention: ``prediction`` is a dictionary with
    ``before`` and ``after`` latent tensors and an intervention-target tensor
    ``targets`` (either an index vector ``(N,)`` or a binary multi-hot matrix
    ``(N, K)``); ``target`` is the true factor tensor of the *factual* samples,
    used only to match latent coordinates to causal variables (Hungarian
    assignment on absolute correlations, valid up to permutation).
    """

    def __init__(self) -> None:
        super().__init__("triplet", "L2_intervention", higher_is_better=False)

    def __call__(self, prediction: dict[str, Tensor], target: Tensor) -> MetricResult:
        before = _flatten_leading(prediction["before"])
        after = _flatten_leading(prediction["after"])
        targets = prediction["targets"].reshape(-1)
        truth = _flatten_leading(target)
        if before.shape != after.shape:
            raise ValueError("'before' and 'after' latents must share a shape.")
        if truth.shape[0] != before.shape[0]:
            raise ValueError("Target factors and latents must share a sample axis.")

        truth_std = (truth - truth.mean(0)) / (truth.std(0) + 1e-8)
        pred_std = (before - before.mean(0)) / (before.std(0) + 1e-8)
        corr = (truth_std.T @ pred_std) / truth.shape[0]
        _, matched = hungarian_match(-corr.abs())
        matched = matched.cpu()

        if targets.ndim == 1 and targets.numel() == before.shape[0]:
            index_tensor = targets.long()
        else:
            binary = prediction["targets"].reshape(before.shape[0], -1)
            counts = binary.sum(dim=-1)
            valid = counts == 1
            index_tensor = binary.argmax(dim=-1)[valid]
            before = before[valid]
            after = after[valid]

        delta = (after - before).abs()
        scores = []
        for row, variable in zip(delta, index_tensor.tolist(), strict=True):
            coordinate = int(matched[variable].item())
            d_self = float(row[coordinate].item())
            others = [float(row[k].item()) for k in range(row.numel()) if k != coordinate]
            d_other = float(np.mean(others)) if others else 0.0
            scores.append(d_other / (d_self + d_other + 1e-8))
        value = float(np.mean(scores)) if scores else float("nan")
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class TwoSampleDiscriminatorMetric(Metric):
    r"""Two-sample discriminator accuracy between real and generated samples.

    A binary classifier (standardised logistic regression) is trained on a
    random split of real interventional samples (``target``) versus generated
    ones (``prediction``); the reported value is ``|accuracy - 0.5|``, which is
    ``0`` when the two distributions are indistinguishable (the ideal, i.e.
    accuracy at chance) and ``0.5`` when perfectly separable. Lower is better.
    The paper reports this separability as the deviation of the discriminator
    accuracy from ``50\%``.
    """

    def __init__(self, *, test_size: float = 0.5, seed: int = 0) -> None:
        super().__init__("two_sample", "L2_intervention", higher_is_better=False)
        self.test_size = test_size
        self.seed = seed

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        from sklearn.linear_model import LogisticRegression

        generated = _flatten_leading(prediction).detach().cpu().numpy()
        real = _flatten_leading(target).detach().cpu().numpy()
        n = min(generated.shape[0], real.shape[0])
        if n < 20:
            return MetricResult(self.name, float("nan"), self.level, self.higher_is_better)
        generated, real = generated[:n], real[:n]
        features = np.concatenate([real, generated], axis=0)
        labels = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
        features = (features - features.mean(0)) / (features.std(0) + 1e-8)
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.seed, stratify=labels
        )
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(x_train, y_train)
        accuracy = float(classifier.score(x_test, y_test))
        value = abs(accuracy - 0.5)
        return MetricResult(self.name, value, self.level, self.higher_is_better)


# --- Effect-estimation metrics -----------------------------------------------

class PEHEMetric(Metric):
    """Precision in Estimation of Heterogeneous Effect."""

    def __init__(self) -> None:
        super().__init__("pehe", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        diff = (prediction - target).reshape(-1)
        pehe = torch.sqrt((diff**2).mean()).item()
        return MetricResult(self.name, pehe, self.level, self.higher_is_better)


class ATEMetric(Metric):
    """Absolute error on the average treatment effect."""

    def __init__(self) -> None:
        super().__init__("ate_error", "L2_intervention", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        ate_pred = prediction.mean().item()
        ate_true = target.mean().item()
        return MetricResult(
            self.name, abs(ate_pred - ate_true), self.level, self.higher_is_better
        )


class PolicyRiskMetric(Metric):
    r"""Policy risk of the treatment rule induced by an estimated effect.

    For a threshold rule :math:`\pi(x) = 1\{\hat\tau(x) > 0\}` the policy risk is
    the expected outcome under that rule,
    :math:`\mathbb E[Y(\pi(X))] = \frac1N\sum_i [\pi_i y_i(1) + (1-\pi_i)
    y_i(0)]`, evaluated with the potential outcomes that the factual outcome and
    its counterfactual expose. ``prediction`` is the estimated effect
    :math:`\hat\tau` and ``target`` a dictionary with ``yf``, ``t`` and ``ycf``.
    Lower is better (the objective is to minimise the expected outcome).
    """

    def __init__(self) -> None:
        super().__init__("policy_risk", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: dict[str, Tensor]) -> MetricResult:
        tau = prediction.reshape(-1)
        factual = target["yf"].reshape(-1)
        counterfactual = target["ycf"].reshape(-1)
        treatment = target["t"].reshape(-1)
        decision = (tau > 0).float()
        effect_one = torch.where(treatment > 0, factual, counterfactual)
        effect_zero = torch.where(treatment > 0, counterfactual, factual)
        outcome = decision * effect_one + (1.0 - decision) * effect_zero
        return MetricResult(
            self.name, float(outcome.mean().item()), self.level, self.higher_is_better
        )


class CFMAEMetric(Metric):
    """Counterfactual mean absolute error."""

    def __init__(self) -> None:
        super().__init__("cf_mae", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        mae = (prediction - target).abs().mean().item()
        return MetricResult(self.name, mae, self.level, self.higher_is_better)


# --- Static-dynamic metrics --------------------------------------------------

class MBEMetric(Metric):
    """Mean bias error: the signed counterpart of the counterfactual MAE."""

    def __init__(self) -> None:
        super().__init__("mbe", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        bias = (prediction - target).mean().item()
        return MetricResult(self.name, float(bias), self.level, self.higher_is_better)


class MMD2Metric(Metric):
    r"""Squared maximum mean discrepancy with an RBF kernel.

    ``MMD^2(P, Q) = E[k(x,x')] - 2 E[k(x,y)] + E[k(y,y')]`` on flattened
    samples. Lower is better; zero means the two empirical distributions agree.
    """

    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__("mmd2", "L3_counterfactual", higher_is_better=False)
        self.bandwidth = bandwidth

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        x = prediction.reshape(prediction.shape[0], -1)
        y = target.reshape(target.shape[0], -1)
        scale = 2.0 * self.bandwidth**2

        def kernel(a: Tensor, b: Tensor) -> Tensor:
            diff = a.unsqueeze(1) - b.unsqueeze(0)
            return torch.exp(-(diff**2).sum(-1) / scale)

        value = (
            kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
        ).item()
        return MetricResult(self.name, float(value), self.level, self.higher_is_better)


class JFTSDMetric(Metric):
    r"""Joint Fr\'echet time-series distance (J-FTSD).

    ``||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})``
    between the Gaussian fits of the real and generated joint
    trajectory--context embeddings (introduced by Narasimhan et al., 2024 and
    adopted by CaTSG). Inputs are ``(N, e)`` joint embeddings; lower is better.
    """

    def __init__(self) -> None:
        super().__init__("jftsd", "L3_counterfactual", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        real = _flatten_leading(target).detach().float()
        generated = _flatten_leading(prediction).detach().float()
        n = min(real.shape[0], generated.shape[0])
        real, generated = real[:n], generated[:n]
        if n < 4:
            return MetricResult(self.name, float("nan"), self.level, self.higher_is_better)
        value = bures_w2(
            real.mean(0, keepdim=True),
            _covariance(real).unsqueeze(0),
            generated.mean(0, keepdim=True),
            _covariance(generated).unsqueeze(0),
        )
        return MetricResult(self.name, float(value.item()), self.level, self.higher_is_better)


class OODMSEMetric(Metric):
    """Out-of-distribution mean squared error."""
    def __init__(self) -> None:
        super().__init__("ood_mse", "L1_generalisation", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        mse = ((prediction - target) ** 2).mean().item()
        return MetricResult(self.name, mse, self.level, self.higher_is_better)


class DomainAccuracyMetric(Metric):
    """Accuracy of inferred discrete domain/regime labels."""

    def __init__(self) -> None:
        super().__init__("domain_accuracy", "L1_generalisation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        pred = prediction.reshape(-1).long()
        true = target.reshape(-1).long()
        accuracy = (pred == true).float().mean().item()
        return MetricResult(self.name, float(accuracy), self.level, self.higher_is_better)


class CDSMetric(Metric):
    """Causal Disentanglement Score.

    For each ground-truth factor, the fraction of its total feature importance
    carried by its single best-explaining latent; averaged over factors. A value
    near one means each factor is concentrated in one latent, near zero means it
    is spread across many.
    """

    def __init__(self) -> None:
        super().__init__("cds", "L1_generalisation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix = _importance_matrix(prediction, target)
        scores = []
        for i in range(matrix.shape[0]):
            total = matrix[i].sum()
            scores.append(float(matrix[i].max() / total) if total > 0 else 0.0)
        return MetricResult(self.name, float(np.mean(scores)), self.level, self.higher_is_better)


class MIGMetric(Metric):
    """Mutual-Information-Gap-style score from the latent/factor importance matrix.

    For each ground-truth factor we rank the normalised importances of the
    latents and average the gap between the top two, so that a factor explained
    by a single latent scores high and a factor spread over many scores low.
    """

    def __init__(self) -> None:
        super().__init__("mig", "L1_generalisation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix = _importance_matrix(prediction, target)
        gaps = []
        for i in range(matrix.shape[0]):
            row = matrix[i]
            total = row.sum()
            if total <= 0:
                gaps.append(0.0)
                continue
            normalised = np.sort(row / total)[::-1]
            gaps.append(float(normalised[0] - normalised[1]) if normalised.size > 1 else 0.0)
        return MetricResult(self.name, float(np.mean(gaps)), self.level, self.higher_is_better)


DEFAULT_REGISTRY = _default_registry()
