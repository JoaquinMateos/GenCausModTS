"""Evaluation metrics organised by model family and causal level."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
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
    "PathF1Metric",
    "SourceAccuracyMetric",
    "StructuralViolationMetric",
    "HypervolumeMetric",
    "SparsityMetric",
    "FaithfulnessMetric",
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
    reg.register(PathF1Metric())
    reg.register(SourceAccuracyMetric())
    reg.register(StructuralViolationMetric())
    reg.register(HypervolumeMetric())
    reg.register(SparsityMetric())
    reg.register(FaithfulnessMetric())
    return reg


# --- Identifiable CRL metrics ------------------------------------------------

class MCCMetric(Metric):
    """Mean correlation coefficient after Hungarian matching."""

    def __init__(self) -> None:
        super().__init__("mcc", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        return MetricResult(self.name, mcc(target, prediction), self.level, self.higher_is_better)


def _nonlinear_r2(features: Tensor, target: Tensor, *, seed: int = 0) -> float:
    """Held-out R² of a nonlinear regressor predicting ``target`` from ``features``.

    Mirrors the coefficient-of-determination protocol of Lippe et al. (2022):
    a nonlinear regressor predicts the true factor and R² is measured on held-out
    samples.  R² is reported as defined (it may be negative for an
    uninformative representation); it is *not* clipped at zero, because clipping
    turns a failed factor into a zero-leakage factor and biases both
    ``R²_diag`` and ``R²_sep`` upwards.  A single pre-specified regressor is
    used; selecting the better of a linear and a nonlinear fit on the test split
    would leak the test labels into model selection.
    """
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy().reshape(-1)
    if x.shape[0] < 40:
        return float("nan")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    boosted = GradientBoostingRegressor(random_state=seed).fit(x_train, y_train)
    return float(r2_score(y_test, boosted.predict(x_test)))


def _r2_matrix(
    pred: Tensor, tgt: Tensor, *, seed: int = 0
) -> tuple[np.ndarray, dict[int, int]]:
    """Return the ``(K_true, K_pred)`` matrix ``R²_ij`` and the matched column.

    The estimated and true latent dimensions need not agree: a method may
    over- or under-allocate its latent budget. The Hungarian assignment is
    rectangular and is returned as ``{true_index: predicted_index}`` so callers
    never index a matrix with a column that does not exist.
    """
    pred = pred.reshape(-1, pred.shape[-1])
    tgt = tgt.reshape(-1, tgt.shape[-1])
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(tgt).all()):
        return np.full((tgt.shape[-1], pred.shape[-1]), np.nan), {}
    pred_std = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
    tgt_std = (tgt - tgt.mean(0)) / (tgt.std(0) + 1e-8)
    corr = (tgt_std.T @ pred_std) / tgt.shape[0]
    row, col = hungarian_match(-corr.abs())
    assignment = {
        int(i): int(j) for i, j in zip(row.tolist(), col.tolist(), strict=True)
    }
    k_true, k_pred = tgt.shape[-1], pred.shape[-1]
    matrix = np.zeros((k_true, k_pred))
    for i in range(k_true):
        for j in range(k_pred):
            matrix[i, j] = _nonlinear_r2(pred[:, j : j + 1], tgt[:, i], seed=seed)
    return matrix, assignment


class R2DiagMetric(Metric):
    """Diagonal R²: each true factor predicted from its matched latent block.

    ``R²_diag = (1/K) Σ_i R²(true_i, matched latent)`` with a non-linear
    regressor, as defined for multi-dimensional causal factors (Lippe et al.,
    2022). Matching uses the Hungarian assignment on correlations.
    """

    def __init__(self) -> None:
        super().__init__("r2_diag", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix, assignment = _r2_matrix(prediction, target)
        matched = [matrix[i, j] for i, j in assignment.items()]
        value = float(np.mean(matched)) if matched else float("nan")
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class R2SepMetric(Metric):
    """Separation R²: leakage of each true factor into non-matched latents.

    ``R²_sep = (1/K) Σ_i max_{j ≠ matched(i)} R²(true_i, latent_j)``; values near
    zero indicate that factors do not leak across latent dimensions.
    """

    def __init__(self) -> None:
        super().__init__("r2_sep", "L1_representation", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix, assignment = _r2_matrix(prediction, target)
        leakage = []
        for i in range(matrix.shape[0]):
            others = [
                matrix[i, j] for j in range(matrix.shape[1]) if j != assignment.get(i)
            ]
            leakage.append(max(others) if others else 0.0)
        value = float(np.mean(leakage)) if leakage else float("nan")
        return MetricResult(self.name, value, self.level, self.higher_is_better)


def _importance_matrix(pred: Tensor, tgt: Tensor, *, seed: int = 0) -> np.ndarray:
    """Feature-importance matrix ``R_ij`` (importance of latent ``j`` for factor ``i``).

    The matrix is ``(K_true, K_pred)`` so it remains well defined when a method
    over- or under-allocates its latent budget.
    """
    x = pred.reshape(-1, pred.shape[-1]).detach().cpu().numpy()
    y = tgt.reshape(-1, tgt.shape[-1]).detach().cpu().numpy()
    k_true, k_pred = y.shape[1], x.shape[1]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.full((k_true, k_pred), np.nan)
    matrix = np.zeros((k_true, k_pred))
    for i in range(k_true):
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
        if not np.isfinite(matrix).all():
            return MetricResult(self.name, float("nan"), self.level, self.higher_is_better)
        scores = [1.0 - _normalised_entropy(matrix[:, j]) for j in range(matrix.shape[1])]
        return MetricResult(self.name, float(np.mean(scores)), self.level, self.higher_is_better)


class DCICompletenessMetric(Metric):
    """DCI completeness: each ground-truth factor is captured by a single latent."""

    def __init__(self) -> None:
        super().__init__("dci_completeness", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        matrix = _importance_matrix(prediction, target)
        if not np.isfinite(matrix).all():
            return MetricResult(self.name, float("nan"), self.level, self.higher_is_better)
        scores = [1.0 - _normalised_entropy(matrix[i]) for i in range(matrix.shape[0])]
        return MetricResult(self.name, float(np.mean(scores)), self.level, self.higher_is_better)


class DCIInformativenessMetric(Metric):
    """DCI informativeness: predictive information the latents retain about factors."""

    def __init__(self) -> None:
        super().__init__("dci_informativeness", "L1_representation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        tgt = target.reshape(-1, target.shape[-1])
        pred = prediction.reshape(-1, prediction.shape[-1])
        if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(tgt).all()):
            return MetricResult(self.name, float("nan"), self.level, self.higher_is_better)
        scores = [_nonlinear_r2(pred, tgt[:, i]) for i in range(tgt.shape[-1])]
        return MetricResult(
            self.name, float(np.nanmean(scores)), self.level, self.higher_is_better
        )


class SHDMetric(Metric):
    """Structural Hamming Distance between estimated and true adjacency.

    The estimated graph is aligned to the ground truth over latent permutations
    before counting edge additions/deletions/reversals, since latents are
    identified only up to permutation. Accepts adjacency of shape ``(d, d)`` or
    ``(d, d, p)``.

    Set ``aligned=False`` when the caller has already aligned the latents with
    :func:`gcmts.core.graph_ops.latent_permutation`. Re-optimising the
    permutation inside the metric would otherwise use the ground-truth graph to
    choose the alignment and make the score optimistic relative to the
    correlation-based protocol used by the source methods.
    """

    def __init__(self, *, aligned: bool = True) -> None:
        super().__init__("shd", "L1_structure", higher_is_better=False)
        self.aligned = aligned

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        from gcmts.core.graph_ops import shd as plain_shd

        value = (
            aligned_shd(prediction, target)
            if self.aligned
            else float(plain_shd(prediction, target))
        )
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class WSHDMetric(Metric):
    """Weighted SHD: structural differences weighted by edge magnitude.

    Uses the same permutation alignment as :class:`SHDMetric`; see
    :func:`gcmts.core.graph_ops.weighted_shd`.
    """

    def __init__(self, *, aligned: bool = True) -> None:
        super().__init__("wshd", "L1_structure", higher_is_better=False)
        self.aligned = aligned

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        if self.aligned:
            value = aligned_shd(prediction, target, weighted=True)
        else:
            from gcmts.core.graph_ops import weighted_shd

            value = weighted_shd(prediction, target)
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
    r"""Intervention-isolation distance (a derived diagnostic, not the CITRIS triplet).

    For an intervention on factor ``a`` the matched latent coordinate
    ``pi(a)`` should move, while the other coordinates should not.  Samples are
    grouped by intervention target so that the natural transition dynamics
    average out; for target ``a``

    ``d_self = mean_{n: I_n=a} |z_after_{pi(a)} - z_before_{pi(a)}|`` and
    ``d_other = mean_{n: I_n=a} mean_{k != pi(a)} |z_after_k - z_before_k|``,

    and the score is ``d_other / (d_self + d_other)`` (zero when the
    intervention is isolated, about ``0.5`` at chance).  This is deliberately
    *not* the CITRIS "triplet" evaluation, which composes two images in latent
    space, decodes the composition and scores the decoded factors with a
    supervised CNN; that protocol requires a rendered-image benchmark and a
    factor regressor, neither of which exists for the low-dimensional SCMs
    here.  Inputs follow the CITRIS convention: ``prediction`` is a dictionary
    with ``before``/``after`` latents and an intervention-target vector;
    ``target`` is the factual factor tensor used only for Hungarian matching.
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
        scores: list[float] = []
        for variable in torch.unique(index_tensor).tolist():
            mask = index_tensor == int(variable)
            if int(mask.sum()) == 0:
                continue
            coordinate = int(matched[int(variable)].item())
            d_self = float(delta[mask, coordinate].mean().item())
            others = [k for k in range(delta.shape[1]) if k != coordinate]
            if not others:
                continue
            d_other = float(delta[mask][:, others].mean().item())
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

    The kernel bandwidth is chosen by the median heuristic on the pooled
    squared distances, which makes the score invariant to a global rescaling of
    the trajectories. A fixed bandwidth (the previous default) makes the value
    depend on the outcome scale and is not comparable across datasets.
    """

    def __init__(self, bandwidth: float | str = "median") -> None:
        super().__init__("mmd2", "L3_counterfactual", higher_is_better=False)
        self.bandwidth = bandwidth

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        x = prediction.reshape(prediction.shape[0], -1).detach().float()
        y = target.reshape(target.shape[0], -1).detach().float()
        pooled = torch.cat([x, y], dim=0)
        if self.bandwidth == "median":
            if pooled.shape[0] > 512:
                generator = torch.Generator().manual_seed(0)
                index = torch.randperm(pooled.shape[0], generator=generator)[:512]
                pooled = pooled[index]
            pairwise = torch.cdist(pooled, pooled).pow(2)
            positive = pairwise[pairwise > 0]
            scale = float(positive.median()) if positive.numel() else 1.0
            scale = max(scale, 1e-12)
        else:
            scale = 2.0 * float(self.bandwidth) ** 2

        def kernel(a: Tensor, b: Tensor) -> Tensor:
            return torch.exp(-torch.cdist(a, b).pow(2) / scale)

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
    """Mutual-Information Gap (MIG) for continuous latents and factors.

    For each ground-truth factor ``v_i`` the metric compares the mutual
    information carried by the best latent ``z_{j*}`` with that of the
    second-best latent, normalised by the differential entropy of ``v_i``:

    ``MIG = (1/K) sum_i [I(z_{j*}; v_i) - max_{j != j*} I(z_j; v_i)] / H(v_i)``.

    Mutual information is estimated with a k-nearest-neighbour regressor
    (``sklearn.feature_selection.mutual_info_regression``); the entropy is the
    Gaussian differential entropy ``0.5 log(2 pi e Var(v_i))``.  The previous
    implementation used the gap between the top two normalised feature
    importances, which is not MIG and is not comparable across dimensions.
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__("mig", "L1_generalisation", higher_is_better=True)
        self.seed = seed

    def __call__(self, prediction: Tensor, target: Tensor) -> MetricResult:
        from sklearn.feature_selection import mutual_info_regression

        x = prediction.reshape(-1, prediction.shape[-1]).detach().cpu().numpy()
        y = target.reshape(-1, target.shape[-1]).detach().cpu().numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return MetricResult(self.name, float("nan"), self.level, self.higher_is_better)
        scores: list[float] = []
        for i in range(y.shape[1]):
            entropy = 0.5 * float(np.log(2.0 * np.pi * np.e * (np.var(y[:, i]) + 1e-12)))
            if entropy <= 1e-8:
                continue
            information = np.array(
                [
                    max(
                        0.0,
                        float(
                            mutual_info_regression(
                                x[:, j : j + 1], y[:, i], random_state=self.seed
                            )[0]
                        ),
                    )
                    for j in range(x.shape[1])
                ]
            )
            if information.size == 0:
                continue
            best = float(information.max())
            second = (
                float(np.partition(information, -2)[-2])
                if information.size > 1
                else 0.0
            )
            scores.append((best - second) / entropy)
        value = float(np.mean(scores)) if scores else float("nan")
        return MetricResult(self.name, value, self.level, self.higher_is_better)


# --- Adjacent-branch metrics (III.A structure discovery) ---------------------

def _valid_nodes(row: Tensor) -> set[int]:
    return {int(v) for v in row.reshape(-1).tolist() if int(v) >= 0}


class PathF1Metric(Metric):
    """Node-set F1 between a predicted and a ground-truth propagation path.

    ``prediction`` and ``target`` are dictionaries with a ``paths`` tensor of
    shape ``(N, L)`` (``-1`` marks padding). The score is the mean over
    instances of the F1 between the predicted and true node sets.
    """

    def __init__(self) -> None:
        super().__init__("path_f1", "L1_structure", higher_is_better=True)

    def __call__(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> MetricResult:
        predicted = prediction["paths"]
        truth = target["paths"]
        scores = []
        for row_pred, row_true in zip(predicted, truth, strict=True):
            pred_set, true_set = _valid_nodes(row_pred), _valid_nodes(row_true)
            if not pred_set and not true_set:
                scores.append(1.0)
                continue
            overlap = len(pred_set & true_set)
            precision = overlap / len(pred_set) if pred_set else 0.0
            recall = overlap / len(true_set) if true_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores.append(f1)
        return MetricResult(self.name, float(np.mean(scores)), self.level, self.higher_is_better)


class SourceAccuracyMetric(Metric):
    """Accuracy of the localised source node; inputs are dictionary ``sources``."""

    def __init__(self) -> None:
        super().__init__("source_accuracy", "L1_structure", higher_is_better=True)

    def __call__(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> MetricResult:
        pred = prediction["sources"].reshape(-1)
        true = target["sources"].reshape(-1)
        accuracy = (pred == true).float().mean().item()
        return MetricResult(self.name, float(accuracy), self.level, self.higher_is_better)


class StructuralViolationMetric(Metric):
    """Fraction of predicted path transitions that violate the given graph.

    ``prediction`` is a dictionary with ``paths`` ``(N, L)``; ``target`` is a
    dictionary with the ``graph`` ``(D, D)``. Lower is better.
    """

    def __init__(self) -> None:
        super().__init__("structural_violation", "L1_structure", higher_is_better=False)

    def __call__(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> MetricResult:
        paths = prediction["paths"]
        graph = target["graph"]
        allowed = (graph + graph.T) > 0
        violations, total = 0, 0
        for row in paths:
            nodes = [int(v) for v in row.tolist() if int(v) >= 0]
            for a, b in zip(nodes[:-1], nodes[1:], strict=True):
                total += 1
                if not bool(allowed[a, b]):
                    violations += 1
        value = violations / total if total else 0.0
        return MetricResult(self.name, float(value), self.level, self.higher_is_better)


# --- Adjacent-branch metrics (III.B decision-making) -------------------------

class HypervolumeMetric(Metric):
    r"""Hypervolume of a Pareto front relative to a reference point.

    ``prediction`` is an ``(N, m)`` set of objective vectors; ``target`` is a
    dictionary with a ``reference`` vector ``(m,)``.  For a maximisation front
    (the default) the reference must be the *worst* corner (a lower bound); the
    dominated region is the set of points ``x`` with ``x <= p`` componentwise
    for some front point ``p``.  The volume is estimated by Monte-Carlo sampling
    in the bounding box between the reference and the front's upper corner.
    Set ``maximize=False`` for a minimisation front.  Higher is better.
    """

    def __init__(self, *, n_samples: int = 20000, seed: int = 0, maximize: bool = True) -> None:
        super().__init__("hypervolume", "L2_decision", higher_is_better=True)
        self.n_points = n_samples
        self.seed = seed
        self.maximize = maximize

    def __call__(self, prediction: Tensor, target: dict[str, Tensor]) -> MetricResult:
        front = prediction.reshape(prediction.shape[0], -1).detach().cpu()
        if not self.maximize:
            front = -front
        reference = target["reference"].reshape(-1).detach().cpu()
        lower = reference
        upper = front.max(dim=0).values
        scale = (upper - lower).clamp(min=1e-8)
        generator = torch.Generator().manual_seed(self.seed)
        samples = lower + torch.rand(self.n_points, front.shape[1], generator=generator) * scale
        dominated = torch.zeros(self.n_points, dtype=torch.bool)
        for point in front:
            dominated |= (samples <= point).all(dim=1)
        box = float(torch.prod(scale))
        value = box * float(dominated.float().mean())
        return MetricResult(self.name, value, self.level, self.higher_is_better)


class SparsityMetric(Metric):
    """Evenness of a Pareto front across objectives (lower is better)."""

    def __init__(self) -> None:
        super().__init__("sparsity", "L2_decision", higher_is_better=False)

    def __call__(self, prediction: Tensor, target: Any = None) -> MetricResult:
        front = prediction.reshape(prediction.shape[0], -1).detach()
        ordered = front.sort(dim=0).values
        gaps = (ordered[1:] - ordered[:-1]).abs()
        value = float(gaps.std(dim=0).mean().item()) if front.shape[0] > 1 else 0.0
        return MetricResult(self.name, value, self.level, self.higher_is_better)


# --- Adjacent-branch metrics (III.C explanation) -----------------------------

class FaithfulnessMetric(Metric):
    """Alignment of attributions with ground-truth causal concepts.

    ``prediction`` is an ``(N, C)`` attribution matrix; ``target`` is a
    dictionary with ``causal_concepts`` (indices). The score is the mean
    attribution on causal concepts minus that on non-causal ones (higher is
    better; a faithful explanation concentrates attribution on the causal
    concepts).
    """

    def __init__(self) -> None:
        super().__init__("faithfulness", "L3_explanation", higher_is_better=True)

    def __call__(self, prediction: Tensor, target: dict[str, Tensor]) -> MetricResult:
        attributions = prediction.reshape(prediction.shape[0], -1).detach()
        causal = [int(c) for c in target["causal_concepts"].reshape(-1).tolist()]
        all_concepts = set(range(attributions.shape[1]))
        non_causal = sorted(all_concepts - set(causal))
        causal_mean = attributions[:, causal].mean().item() if causal else 0.0
        non_causal_mean = (
            attributions[:, non_causal].mean().item() if non_causal else 0.0
        )
        value = causal_mean - non_causal_mean
        return MetricResult(self.name, float(value), self.level, self.higher_is_better)


DEFAULT_REGISTRY = _default_registry()
