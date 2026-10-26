"""Canonical, paper-facing names for the benchmarks of the comparative section.

A *benchmark* is one data-generating process (a synthetic generator with a fixed
configuration) together with the methods compared on it under the single
protocol of the review. The shared CRL benchmark suite is organised as:

* **A.1--A.20** — the temporal-family benchmarks, on which iVAE, LEAP, TDRL,
  NCTRL and MOSAIC are compared head-to-head (a broad set of linear, nonlinear
  and physical processes);
* **B** — Voronoi / Causal Pinball (recorded interventions; CITRIS, iCITRIS);
* **C** — continuous-time SDE (CEGEN);
* **D** — hierarchical temporal process (CHiLD).

The effect-estimation suite uses three temporal simulators (harmonic, tumour,
time-varying-treatment SEM) and, for static effects, the IHDP/ACIC/Twins/Jobs/
News/CPS protocols.

This module is the single source of truth: the CRL runner logs the labels, the
full-benchmark orchestrator names its steps with them, and
``scripts/make_paper_tables.py`` uses them (and the ``kind``/``shd`` metadata) to
emit one table block per benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CrlBenchmark",
    "CRL_BENCHMARKS",
    "CTEE_BENCHMARKS",
    "CTEE_STATIC_BENCHMARKS",
    "CTEE_OPTIONAL_STATIC_BENCHMARKS",
    "CTEE_REAL_BENCHMARKS",
    "METHODS_FOR_KIND",
    "crl_label",
    "ctee_label",
    "EMA_DECAY",
]

# Single EMA setting for every learned run in the benchmark.  The pre-audit run
# used 0.99 for six benchmarks and no EMA for sixteen; the audited protocol
# applies it uniformly (and the trainer makes the averaged weights the final
# state, so every reported metric is computed from the EMA weights).
EMA_DECAY = 0.99


@dataclass(frozen=True)
class CrlBenchmark:
    """Paper label, description and evaluation metadata for a CRL benchmark."""

    label: str
    description: str
    kind: str  # "temporal" | "voronoi" | "sde" | "child"
    shd_available: bool
    summary: bool = True  # include in the compact main-text MCC matrix


METHODS_FOR_KIND: dict[str, list[str]] = {
    "temporal": ["iVAE", "LEAP", "TDRL", "NCTRL", "MOSAIC"],
    "voronoi": ["CITRIS", "iCITRIS"],
    "sde": ["CEGEN"],
    "child": ["CHiLD"],
}

# key -> CrlBenchmark
CRL_BENCHMARKS: dict[str, CrlBenchmark] = {
    "npvar": CrlBenchmark(
        "A.1",
        "nonlinear persistent L-TSCM process (NP/VAR)",
        "temporal",
        True,
    ),
    "npvar_linear": CrlBenchmark(
        "A.1b",
        "linear-mixing NP/VAR (linear-Gaussian baseline)",
        "temporal",
        True,
        summary=False,
    ),
    "npvar_hard": CrlBenchmark(
        "A.2", "non-stationary nonlinear NP/VAR (hard)", "temporal", True
    ),
    "cartpole": CrlBenchmark(
        "A.3", "non-stationary interactive process (CartPole)", "temporal", False
    ),
    "causal3d": CrlBenchmark("A.4", "Temporal Causal3DIdent factor process", "temporal", True),
    "lorenz": CrlBenchmark("A.5", "Lorenz-63 physical SCM", "temporal", False),
    "lorenz96": CrlBenchmark("A.6", "Lorenz-96 ring SCM", "temporal", False),
    "pcl": CrlBenchmark("A.7", "physics-coupled Lorenz (PCL)", "temporal", False),
    "kuramoto": CrlBenchmark("A.8", "Kuramoto oscillator network", "temporal", False),
    "lotka_volterra": CrlBenchmark("A.9", "Lotka-Volterra ecosystem", "temporal", False),
    "duffing": CrlBenchmark("A.10", "Duffing oscillator", "temporal", False),
    "fitzhugh_nagumo": CrlBenchmark("A.11", "FitzHugh-Nagumo neuron", "temporal", False),
    "chua": CrlBenchmark("A.12", "Chua's circuit", "temporal", False),
    "van_der_pol": CrlBenchmark("A.13", "van der Pol oscillator", "temporal", False),
    "pendulum": CrlBenchmark("A.14", "torque-limited pendulum", "temporal", False),
    "mountain_car": CrlBenchmark("A.15", "mountain-car control", "temporal", False),
    "varma": CrlBenchmark("A.16", "VARMA process", "temporal", True),
    "ou": CrlBenchmark("A.17", "Ornstein-Uhlenbeck process", "temporal", True),
    "voronoi": CrlBenchmark(
        "B", "instantaneous + lagged SCM (Voronoi / Causal Pinball)", "voronoi", True
    ),
    "sde": CrlBenchmark("C", "continuous-time SDE process", "sde", False),
    "child": CrlBenchmark("D", "hierarchical temporal process (CHiLD)", "child", True),
}

CTEE_BENCHMARKS: dict[str, str] = {
    "harmonic": "Harmonic oscillator",
    "tumour": "Tumour growth",
    "linear_sem": "Time-varying-treatment SEM",
}

CTEE_STATIC_BENCHMARKS: dict[str, str] = {
    "ihdp": "IHDP",
    "acic": "ACIC 2018",
    "twins": "Twins",
    "news": "News",
}

# Documented optional panel: the exact CFRNet CPS file is not publicly
# mirrored.  Supply data/static/cps.csv to enable it.
CTEE_OPTIONAL_STATIC_BENCHMARKS: dict[str, str] = {
    "cps": "CPS (manual)",
}

CTEE_REAL_BENCHMARKS: dict[str, str] = {
    "lalonde_rct": "LaLonde RCT (NSW)",
    "jobs_rct": "Jobs (LaLonde, binary)",
    "oregon_rct": "Oregon Health Insurance",
    "nhefs": "NHEFS",
}


def crl_label(key: str) -> str:
    """Return ``Benchmark <label>: <description>`` for a CRL benchmark key."""
    benchmark = CRL_BENCHMARKS[key]
    return f"Benchmark {benchmark.label}: {benchmark.description}"


def ctee_label(key: str) -> str:
    """Return the paper-facing name of a CTEE temporal benchmark key."""
    return CTEE_BENCHMARKS.get(key, key)
