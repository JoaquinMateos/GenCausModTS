# ruff: noqa: E501 -- LaTeX templates are single long lines by design.
"""Generate the LaTeX experiment tables from benchmark JSON rows.

Reads the raw per-seed rows written by ``evaluation.report.save_rows`` and
writes ``latex/pic/experiments/crl.tex``, ``ctee.tex`` and ``stats.tex`` with
mean/std aggregation and the Friedman/Nemenyi summary. Missing suites leave the
corresponding rows as ``--`` (pending) so the manuscript always compiles.

Usage::

    uv run python scripts/make_paper_tables.py \
        --shared-dir outputs_shared --outputs-dir outputs \
        --latex-dir ../latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcmts.evaluation.report import aggregate_rows
from gcmts.evaluation.stats import friedman_test, metric_matrix, nemenyi, plot_cd_diagram

__all__ = ["main"]


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def _fmt(
    aggregated: dict[str, Any], case: str, metric: str, *, bold: bool = False, digits: int = 2
) -> str:
    if case not in aggregated or metric not in aggregated[case]:
        return "--"
    entry = aggregated[case][metric]
    mean, std = entry["mean"], entry["std"]
    if np.isnan(mean):
        return "--"
    text = f"{mean:.{digits}f}"
    body = f"{text}\\tiny{{$\\pm${std:.{digits}f}}}"
    return f"\\textbf{{{body}}}" if bold else body


def _best_cases(aggregated: dict[str, Any], cases: list[str], metric: str, *, higher: bool) -> set[str]:
    values = {}
    for case in cases:
        if case in aggregated and metric in aggregated[case]:
            value = aggregated[case][metric]["mean"]
            if not np.isnan(value):
                values[case] = value
    if not values:
        return set()
    best = max(values.values()) if higher else min(values.values())
    tolerance = 1e-9
    return {case for case, value in values.items() if abs(value - best) <= tolerance}


def _suite_block(
    rows: list[dict[str, Any]],
    suite: str,
    methods: list[str],
    *,
    shd_available: bool,
    seeds: int = 10,
    sde: bool = False,
) -> list[str]:
    subset = [row for row in rows if row.get("suite") == suite]
    if not subset:
        return []
    # Complete blocks only: every method must have all seeds before the block
    # is published, so partial runs never masquerade as ten-seed means.
    for method in methods:
        if sum(1 for row in subset if row.get("case") == method and "seed" in row) < seeds:
            return []
    aggregated = aggregate_rows(subset)
    if sde:
        # CEGEN operates in observation space: no mixing model, so MCC and the
        # latent-graph columns are undefined and its conditional W2 is the
        # primary structural score (reported in the MCC column).
        best_w2 = _best_cases(aggregated, methods, "w2", higher=False)
        lines = []
        for method in methods:
            if method not in aggregated:
                continue
            w2 = _fmt(aggregated, method, "w2", bold=method in best_w2, digits=4)
            f1 = _fmt(aggregated, method, "forecast1")
            f4 = _fmt(aggregated, method, "forecast4")
            lines.append(
                f"    {method} & {suite} & {w2} & n/a & n/a & n/a & n/a & n/a & {f1} & {f4} \\\\"
            )
        return lines
    best_mcc = _best_cases(aggregated, methods, "mcc", higher=True)
    best_r2d = _best_cases(aggregated, methods, "r2_diag", higher=True)
    best_r2s = _best_cases(aggregated, methods, "r2_sep", higher=False)
    best_auprc = _best_cases(aggregated, methods, "auprc", higher=True)
    lines = []
    for method in methods:
        if method not in aggregated:
            continue
        mcc = _fmt(aggregated, method, "mcc", bold=method in best_mcc)
        r2d = _fmt(aggregated, method, "r2_diag", bold=method in best_r2d)
        r2s = _fmt(aggregated, method, "r2_sep", bold=method in best_r2s)
        shd = _fmt(aggregated, method, "shd") if shd_available else "n/a"
        auprc = _fmt(aggregated, method, "auprc", bold=method in best_auprc)
        triplet = _fmt(aggregated, method, "triplet")
        f1 = _fmt(aggregated, method, "forecast1")
        f4 = _fmt(aggregated, method, "forecast4")
        lines.append(
            f"    {method} & {suite} & {mcc} & {r2d} & {r2s} & {shd} & {auprc} & {triplet} & {f1} & {f4} \\\\"
        )
    return lines


def _baselines(suite_name: str) -> list[str]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_crl_shared import SUITES, _persistence_mse
    from sklearn.linear_model import Ridge

    from gcmts.core import move_batch

    if suite_name not in SUITES:
        return []
    suite = SUITES[suite_name]
    import torch

    eval_batch = move_batch(suite.make_generator(0).sample(4096), torch.device("cpu"))
    train_batch = suite.make_generator(12345).sample(8192)
    flat_x = train_batch.x[:, :-1].reshape(-1, train_batch.x.shape[-1]).numpy()
    flat_y = train_batch.x[:, 1:].reshape(-1, train_batch.x.shape[-1]).numpy()
    ar = Ridge(alpha=1e-3).fit(flat_x, flat_y)
    test_x = eval_batch.x[:, :-1].reshape(-1, eval_batch.x.shape[-1]).numpy()
    test_y = eval_batch.x[:, 1:].reshape(-1, eval_batch.x.shape[-1]).numpy()
    ar_mse = float(((ar.predict(test_x) - test_y) ** 2).mean())
    p1 = _persistence_mse(eval_batch, 1)
    p4 = _persistence_mse(eval_batch, 4)
    label = suite_name
    return [
        f"    Persistence & {label} & n/a & n/a & n/a & n/a & n/a & n/a & {p1:.2f} & {p4:.2f} \\\\",
        f"    AR(1) & {label} & n/a & n/a & n/a & n/a & n/a & n/a & {ar_mse:.2f} & -- \\\\",
    ]


def write_crl_table(shared_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    rows: list[dict[str, Any]] = []
    for stem in (
        "crl_shared_npvar",
        "crl_shared_cartpole",
        "crl_shared_causal3d",
        "crl_shared_voronoi",
        "crl_shared_sde",
    ):
        rows.extend(_load(shared_dir / f"{stem}.json"))
    temporal = ["iVAE", "LEAP", "TDRL", "NCTRL", "MOSAIC"]

    header = (
        r"""\begin{table}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.12}
  \caption{Head-to-head comparison of representative Identifiable CRL methods across the shared benchmark suites. All methods in a block share the same data, latent dimension and seed schedule, so methods from different branches (I.A--I.E) are directly comparable within a block. Methods are listed in a fixed order (iVAE, LEAP, TDRL, NCTRL, MOSAIC), not by performance. Mean$\pm$std over """
        + str(seeds)
        + r""" seeds. Lower is better for $R^2_{\text{sep}}$, SHD, the triplet distance and both forecasting-MSE columns; higher is better for MCC, $R^2_{\text{diag}}$ and edge AUPRC. ``n/a'' marks a metric the benchmark cannot expose (control systems define no latent graph; the SDE suite has no mixing model, so CEGEN's conditional $W_2$ is reported in the MCC column); ``--'' a result still to be produced. The forecast columns are a secondary \emph{downstream-utility} diagnostic---the future is generated by the same known mechanism---not a soundness metric; persistence and AR(1) reference rows anchor them.}
  \label{tab:exp_crl}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} p{0.09\textwidth} p{0.11\textwidth} r r r r r r r r @{}}
    \toprule
    \textbf{Method} & \textbf{Benchmark} & \textbf{MCC $\uparrow$} & \textbf{$R^2_{\text{diag}}$ $\uparrow$} & \textbf{$R^2_{\text{sep}}$ $\downarrow$} & \textbf{SHD $\downarrow$} & \textbf{Edge AUPRC $\uparrow$} & \textbf{Triplet $\downarrow$} & \textbf{1-step MSE $\downarrow$} & \textbf{4-step MSE $\downarrow$} \\
    \midrule
"""
    )
    blocks: list[str] = []
    npvar_lines = _suite_block(rows, "npvar", temporal, shd_available=True, seeds=seeds)
    if npvar_lines:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite A: discrete-time L-TSCM process (NP/VAR)}}\\\\\n"
            + "\n".join(npvar_lines)
            + "\n"
            + "\n".join(_baselines("npvar"))
            + "\n    \\cmidrule(l){1-10}\n"
        )
    cartpole_lines = _suite_block(rows, "cartpole", temporal, shd_available=False, seeds=seeds)
    if cartpole_lines:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite A: non-stationary interactive process (CartPole/Pendulum)}}\\\\\n"
            + "\n".join(cartpole_lines)
            + "\n"
            + "\n".join(_baselines("cartpole"))
            + "\n    \\cmidrule(l){1-10}\n"
        )
    else:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite A: non-stationary interactive process (CartPole/Pendulum) --- ten-seed run in progress}}\\\\\n"
        )
    causal_lines = _suite_block(rows, "causal3d", temporal, shd_available=True, seeds=seeds)
    if causal_lines:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite A$'$: Temporal Causal3DIdent factor process}}\\\\\n"
            + "\n".join(causal_lines)
            + "\n"
            + "\n".join(_baselines("causal3d"))
            + "\n    \\cmidrule(l){1-10}\n"
        )
    else:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite A$'$: Temporal Causal3DIdent factor process --- ten-seed run in progress}}\\\\\n"
        )
    voronoi_lines = _suite_block(rows, "voronoi", ["CITRIS"], shd_available=True, seeds=seeds)
    if voronoi_lines:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite B: instantaneous $+$ lagged SCM (Voronoi / Causal Pinball) --- CITRIS}}\\\\\n"
            + "\n".join(voronoi_lines)
            + "\n    \\cmidrule(l){1-10}\n"
        )
    else:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite B: instantaneous $+$ lagged SCM (Voronoi / Causal Pinball) --- pending}}\\\\\n"
            "    CITRIS & Voronoi / Pinball & -- & -- & -- & -- & -- & -- & -- & -- \\\\\n"
        )
    sde_lines = _suite_block(rows, "sde", ["CEGEN"], shd_available=False, seeds=seeds, sde=True)
    if sde_lines:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite C: continuous-time SDE process --- CEGEN}}\\\\\n"
            + "\n".join(sde_lines)
            + "\n    \\cmidrule(l){1-10}\n"
        )
    else:
        blocks.append(
            "    \\multicolumn{10}{@{}l}{\\emph{Suite C: continuous-time SDE process --- pending}}\\\\\n"
            "    CEGEN & Synthetic SDE & -- & -- & -- & -- & -- & -- & -- & -- \\\\\n"
        )

    missing = []
    if not npvar_lines:
        missing.append("NP/VAR")
    if not cartpole_lines:
        missing.append("CartPole")
    if not causal_lines:
        missing.append("Causal3DIdent")
    if not voronoi_lines:
        missing.append("Voronoi")
    if not sde_lines:
        missing.append("SDE")
    if missing:
        pending = (
            "\\textcolor{red}{Blocks are added as their ten-seed runs complete; pending: "
            + ", ".join(missing)
            + ". "
        )
        close = "}}"
    else:
        pending = ""
        close = "}"
    footer = (
        r"""    \bottomrule
    \addlinespace
    \multicolumn{10}{@{}p{\textwidth}@{}}{"""
        + pending
        + r"""The image-render variant of Causal3DIdent remains pending. Structural recovery is scored with the post-hoc protocol of the source methods (per-target sparse regression, Hungarian permutation alignment) and SHD/edge-AUPRC are only defined where the benchmark exposes a latent graph. Real-world series are deliberately excluded: they expose no causal ground truth, so they cannot populate any soundness metric (rule R1)."""
        + close
        + r""" \\
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "crl.tex"
    out.write_text(header + "".join(blocks) + footer, encoding="utf-8")
    print(f"wrote {out}")


def _case_complete(rows: list[dict[str, Any]], case: str, seeds: int) -> bool:
    """True when ``case`` has at least ``seeds`` independent runs."""
    return sum(1 for row in rows if row.get("case") == case) >= seeds


def write_ctee_table(outputs_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    rows = _load(outputs_dir / "ctee_benchmarks.json")
    if not rows:
        print("no CTEE rows; leaving ctee.tex unchanged")
        return
    aggregated = aggregate_rows(rows)
    order = [
        ("ganite_harmonic", "GANITE", "II.A adjust", "Harmonic"),
        ("crn_harmonic", "CRN", "II.B repr.", "Harmonic"),
        ("cepae_harmonic", "CEPAE", "II.B repr.", "Harmonic"),
        ("catsg_harmonic", "CaTSG", "II.C latent", "Harmonic"),
        ("ganite_tumour", "GANITE", "II.A adjust", "Tumour"),
        ("crn_tumour", "CRN", "II.B repr.", "Tumour"),
        ("cepae_tumour", "CEPAE", "II.B repr.", "Tumour"),
        ("catsg_tumour", "CaTSG", "II.C latent", "Tumour"),
    ]

    def row(case: str, method: str, branch: str, dataset: str) -> str:
        cells = [
            _fmt(aggregated, case, "pehe", digits=2),
            _fmt(aggregated, case, "ate_error", digits=2),
            _fmt(aggregated, case, "cf_mae", digits=2),
            _fmt(aggregated, case, "cf_mbe", digits=2),
            _fmt(aggregated, case, "mmd2", digits=4),
            _fmt(aggregated, case, "recon_axiom", digits=2),
            _fmt(aggregated, case, "effectiveness_ok", digits=1),
            _fmt(aggregated, case, "reversibility", digits=2),
        ]
        return f"    {method} & {branch} & {dataset} & " + " & ".join(cells) + " \\\\"

    available = [item for item in order if _case_complete(rows, item[0], seeds)]
    pending_datasets = sorted({item[3] for item in order if item not in available})
    lines = [row(*item) for item in available]
    caption = (
        "Counterfactual and treatment-effect estimators (Branches II.A observed-data adjustment, II.B representation-based identification, II.C latent-confounder \\& environment models) on two benchmarks with exact counterfactuals: the damped harmonic oscillator and the tumour-growth simulator (time-varying chemo/radio treatment). Methods are grouped by branch, not by performance. Mean $\\pm$ standard deviation over "
        + str(seeds)
        + " seeds; lower is better except ``Eff.''. ``Rec.'' is the reconstruction axiom $\\text{MAE}(\\hat X(\\mathrm{do}(a_{\\mathrm{f}})),X^{\\mathrm{f}})$, ``Eff.'' the fraction of seeds whose average estimated effect has the correct sign, and ``Rev.'' the reversibility MAE after applying the reverse treatment. See Section~\\ref{sec:exp_ctee}."
    )
    footer = ""
    if pending_datasets:
        footer = (
            "\n    \\addlinespace\n    \\multicolumn{11}{@{}p{\\textwidth}@{}"
            "}{\\textcolor{red}{Incomplete blocks suppressed until all ten seeds of every method are available: "
            + ", ".join(pending_datasets)
            + ".}}\\\\"
        )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_ctee}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l l l r r r r r r r r}
    \toprule
    \textbf{Method} & \textbf{Branch} & \textbf{Benchmark} & \textbf{PEHE $\downarrow$} & \textbf{$\epsilon_{\text{ATE}}$ $\downarrow$} & \textbf{CF MAE $\downarrow$} & \textbf{MBE $\downarrow$} & \textbf{MMD$^2$ $\downarrow$} & \textbf{Rec. $\downarrow$} & \textbf{Eff. $\uparrow$} & \textbf{Rev. $\downarrow$} \\
    \midrule
"""
        + "\n".join(lines)
        + footer
        + r"""
    \bottomrule
  \end{tabular}}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "ctee.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_sdd_table(outputs_dir: Path, latex_dir: Path, *, seeds: int) -> None:
    rows = _load(outputs_dir / "sdd_benchmarks.json")
    if not rows:
        print("no SDD rows; leaving sdd.tex unchanged")
        return
    aggregated = aggregate_rows(rows)
    order = [
        ("dann_multidomain", "DANN"),
        ("vrex_multidomain", "VREx"),
        ("sync_multidomain", "SYNC"),
        ("erm_multidomain", "ERM (baseline)"),
    ]
    lines = []
    missing_cases = []
    for case, label in order:
        if case not in aggregated:
            missing_cases.append(label)
            continue
        if not _case_complete(rows, case, seeds):
            missing_cases.append(label)
            continue
        lines.append(
            f"    {label} & Multi-domain synthetic & "
            + " & ".join(
                _fmt(aggregated, case, metric)
                for metric in ("ood_mse", "ood_mae", "domain_accuracy", "mig", "cds")
            )
            + " \\\\"
        )
    oracle = aggregated.get("oracle_causal")
    if oracle is not None:
        lines.append(
            "    \\cmidrule(l){1-7}\n    Oracle (true causal factors) & Multi-domain synthetic & "
            + _fmt(aggregated, "oracle_causal", "ood_mse")
            + " & "
            + _fmt(aggregated, "oracle_causal", "ood_mae")
            + " & -- & -- & -- \\\\"
        )
    caption = (
        "Static--dynamic (invariance) sub-branch of non-stationarity \\& environment shift (Branch I.B.2) "
        "on the multi-domain SCM of Section~\\ref{sec:exp_sdd}. Models train on domains $0$--$3$ "
        "and are tested on held-out domain $4$, whose spurious correlation has the opposite sign. "
        "Methods are listed by sub-branch (invariant prediction, then four-way split), not by "
        "performance. Mean $\\pm$ standard deviation over " + str(seeds) + " seeds; lower is better "
        "for OOD error and higher for domain-identification accuracy, MIG and CDS. The last row is a "
        "ridge regression on the \\emph{true} causal factors and is the achievable floor."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_sdd}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} p{0.15\textwidth} p{0.20\textwidth} r r r r r @{}}
    \toprule
    \textbf{Method} & \textbf{Benchmark} & \textbf{OOD MSE $\downarrow$} & \textbf{OOD MAE $\downarrow$} & \textbf{Dom.-ID Acc. $\uparrow$} & \textbf{MIG $\uparrow$} & \textbf{CDS $\uparrow$} \\
    \midrule
"""
        + "\n".join(lines)
        + (
            "\n    \\addlinespace\n    \\multicolumn{7}{@{}p{\\textwidth}@{}"
            "}{\\textcolor{red}{Incomplete blocks suppressed until all ten seeds of every method are available: "
            + ", ".join(missing_cases)
            + ".}}\\\\"
            if missing_cases
            else ""
        )
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "sdd.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def write_stats_section(shared_dir: Path, latex_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for stem in (
        "crl_shared_grid_a",
        "crl_shared_grid_b",
        "crl_shared_grid",
    ):
        rows.extend(_load(shared_dir / f"{stem}.json"))
    if not rows:
        print("no grid rows; leaving stats.tex unchanged")
        return

    lines: list[str] = []
    figure_names: list[str] = []
    for metric, higher in (
        ("mcc", True),
        ("r2_diag", True),
        ("r2_sep", False),
        ("forecast1", False),
    ):
        matrix, cases, problems = metric_matrix(rows, metric)
        complete = matrix[~np.isnan(matrix).any(axis=1)]
        if complete.shape[0] < 2 or complete.shape[1] < 3:
            continue
        friedman = friedman_test(matrix)
        result = nemenyi(matrix, higher_is_better=higher)
        ranks = ", ".join(
            f"{case}: {rank:.2f}"
            for case, rank in zip(cases, result["average_ranks"].tolist(), strict=True)
        )
        lines.append(
            f"\\textbf{{{metric}}} & {friedman['n_problems']:.0f} & "
            f"{friedman['statistic']:.2f} & {friedman['p_value']:.3g} & "
            f"{result['critical_difference']:.2f} & {ranks} \\\\"
        )
        if metric in {"mcc", "r2_diag", "forecast1"}:
            name = f"cd_{metric}.png"
            plot_cd_diagram(
                {
                    case: float(rank)
                    for case, rank in zip(cases, result["average_ranks"].tolist(), strict=True)
                },
                float(result["critical_difference"]),
                path=latex_dir / "pic" / "experiments" / name,
                title=f"{metric} ({'higher' if higher else 'lower'} is better)",
            )
            figure_names.append(name)

    n_problems = len({str(row.get("dataset", "")) for row in rows})
    section = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{Statistical comparison of the five temporal CRL methods over the """
        + f"{n_problems}"
        + r"""-problem configuration grid (Section~\ref{sec:exp_protocol}). The Friedman test is the omnibus test; the Nemenyi critical difference (CD) is the smallest average-rank gap significant at $\alpha=0.05$ for this number of problems and methods. Average ranks use one as best (lower is better).}
  \label{tab:exp_stats}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l r r r r p{0.42\textwidth} @{}}
    \toprule
    \textbf{Metric} & \textbf{N} & \textbf{Friedman $\chi^2$} & \textbf{$p$} & \textbf{CD} & \textbf{Average ranks} \\
    \midrule
"""
        + "\n".join(lines)
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    if figure_names:
        includes = "\n".join(
            f"  \\includegraphics[width=.32\\textwidth]{{pic/experiments/{name}}}"
            for name in figure_names
        )
        section += (
            "\n\\begin{figure*}[t]\n  \\centering\n"
            + includes
            + "\n  \\caption{Critical-difference diagrams for the eighteen-problem CRL configuration grid: methods whose average ranks are joined by a bar are not significantly different at $\\alpha=0.05$ (Nemenyi). Lower ranks are better.}\n"
            "  \\label{fig:cd_diagrams}\n\\end{figure*}\n"
        )
    out = latex_dir / "pic" / "experiments" / "stats.tex"
    out.write_text(section, encoding="utf-8")
    print(f"wrote {out}")


def write_ctee_static_table(
    outputs_dir: Path, latex_dir: Path, *, replications: int = 100
) -> None:
    """IHDP static benchmark table (canonical 100-replication protocol)."""
    rows = _load(outputs_dir / "ctee_static_ihdp.json")
    if not rows:
        print("no static IHDP rows; leaving ctee_static.tex unchanged")
        return
    aggregated = aggregate_rows(rows)
    order = [
        ("ganite_ihdp", "GANITE", "II.A adjust"),
        ("cepae_ihdp", "CEPAE", "II.B repr."),
        ("crn_ihdp", "CRN", "II.B repr."),
        ("catsg_ihdp", "CaTSG", "II.C latent"),
        ("ridge_t_ihdp", "Ridge T-learner", "baseline"),
        ("nearest_neighbour_ihdp", "Nearest neighbour", "baseline"),
    ]
    completed = sum(1 for row in rows if row.get("case") == "ganite_ihdp")
    complete = completed >= replications and all(
        _case_complete(rows, case, replications) for case, _, _ in order
    )
    lines = []
    if complete:
        for case, method, branch in order:
            lines.append(
                f"    {method} & {branch} & "
                + " & ".join(
                    _fmt(aggregated, case, metric, digits=2)
                    for metric in ("pehe", "ate_error", "policy_risk")
                )
                + " \\\\"
            )
    body = (
        "\n".join(lines)
        if complete
        else (
            "    \\multicolumn{5}{@{}l}{\\textcolor{red}{"
            + str(replications)
            + "-replication run in progress ("
            + str(completed)
            + "/"
            + str(replications)
            + "); the table is generated by \\texttt{scripts/make\\_paper\\_tables.py}}} \\\\"
        )
    )
    caption = (
        "Static counterfactual benchmark: the canonical IHDP protocol (Hill, 2011). Each of the "
        + str(replications)
        + " replications is a separate benchmark instance; estimators train on the standard train "
        "split and are scored on the held-out split against the noiseless potential means $\\mu_0,\\mu_1$, "
        "with PEHE and $\\epsilon_{\\text{ATE}}$ for individual and marginal effects and the policy risk "
        "of the induced threshold rule (lower is better throughout). Ridge T-learner and nearest-neighbour "
        "matching are the standard non-parametric references. Mean $\\pm$ standard deviation over replications."
    )
    table = (
        r"""\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.15}
  \caption{"""
        + caption
        + r"""}
  \label{tab:exp_ctee_static}
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l r r r @{}}
    \toprule
    \textbf{Method} & \textbf{Branch} & \textbf{PEHE $\downarrow$} & \textbf{$\epsilon_{\text{ATE}}$ $\downarrow$} & \textbf{Policy risk $\downarrow$} \\
    \midrule
"""
        + body
        + r"""
    \bottomrule
  \end{tabular*}
\end{table}
"""
    )
    out = latex_dir / "pic" / "experiments" / "ctee_static.tex"
    out.write_text(table, encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX experiment tables.")
    parser.add_argument("--shared-dir", type=Path, default=Path("outputs_shared"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--latex-dir", type=Path, default=Path("../latex"))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    write_crl_table(args.shared_dir, args.latex_dir, seeds=args.seeds)
    write_ctee_table(args.outputs_dir, args.latex_dir, seeds=args.seeds)
    write_ctee_static_table(args.outputs_dir, args.latex_dir)
    write_sdd_table(args.outputs_dir, args.latex_dir, seeds=args.seeds)
    write_stats_section(args.shared_dir, args.latex_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
