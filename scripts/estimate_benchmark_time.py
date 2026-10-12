"""Estimate the wall-clock time of the full benchmark.

Per-run times were measured on a single NVIDIA RTX 3050 (8 GB); pass
``--speedup`` for another GPU (an RTX 2080 Ti is roughly 1.3x faster, so use
``--speedup 1.3``). Run counts come from the benchmark registries, so the
estimate tracks the suite as it grows. Usage::

    uv run python scripts/estimate_benchmark_time.py --seeds 5 --replications 100
    uv run python scripts/estimate_benchmark_time.py --seeds 5 --with-grid --speedup 1.3
"""

from __future__ import annotations

import argparse

from gcmts.evaluation.benchmarks import CRL_BENCHMARKS, METHODS_FOR_KIND

# Measured per-run seconds on the RTX 3050 (see AGENTS.md section 14).
SECONDS = {
    "crl_lowdim": 350.0,
    "crl_causal3d": 964.0,
    "crl_sanity": 300.0,
    "ctee_temporal": 190.0,
    "ctee_static_per_rep": 409.0,
    "ctee_real": 20.0,
    "sdd": 73.0,
    "adjacent": 22.0,
    "real_panel_per_seed": 750.0,
    "grid_per_run": 350.0,
}

HIGH_DIM = {"causal3d"}
SANITY_CASES = 11
REAL_PANELS = 11
SDD_METHODS = 5
ADJACENT_METHODS = 4
CTEE_TEMPORAL = 3
CTEE_TEMPORAL_METHODS = 4
CTEE_STATIC = 5
CTEE_STATIC_METHODS = 4
CTEE_REAL = 3
GRID_PROBLEMS = 18
TEMPORAL_METHODS = 5


def estimate(
    *, seeds: int, replications: int, with_grid: bool, speedup: float
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []

    def add(name: str, seconds: float) -> None:
        rows.append((name, seconds / speedup))

    lowdim = sum(
        1 for k, b in CRL_BENCHMARKS.items() if b.kind == "temporal" and k not in HIGH_DIM
    )
    highdim = sum(
        1 for k, b in CRL_BENCHMARKS.items() if b.kind == "temporal" and k in HIGH_DIM
    )
    single = sum(
        len(METHODS_FOR_KIND[b.kind]) for b in CRL_BENCHMARKS.values() if b.kind != "temporal"
    )
    add(
        "CRL shared (low-dim temporal)",
        lowdim * TEMPORAL_METHODS * seeds * SECONDS["crl_lowdim"],
    )
    add(
        "CRL shared (Causal3DIdent)",
        highdim * TEMPORAL_METHODS * seeds * SECONDS["crl_causal3d"],
    )
    add("CRL shared (Voronoi/SDE/child)", single * seeds * SECONDS["crl_lowdim"])
    add("CRL sanity", SANITY_CASES * seeds * SECONDS["crl_sanity"])
    add(
        "CTEE temporal",
        CTEE_TEMPORAL * CTEE_TEMPORAL_METHODS * seeds * SECONDS["ctee_temporal"],
    )
    add(
        "CTEE static (IHDP/Twins/Jobs/News/CPS)",
        CTEE_STATIC * replications * SECONDS["ctee_static_per_rep"],
    )
    add("CTEE real (LaLonde/Oregon/NHEFS)", CTEE_REAL * CTEE_STATIC_METHODS * SECONDS["ctee_real"])
    add("SDD invariance", SDD_METHODS * seeds * SECONDS["sdd"])
    add("Adjacent branches", ADJACENT_METHODS * seeds * SECONDS["adjacent"])
    add("Real external validity", REAL_PANELS * seeds * SECONDS["real_panel_per_seed"])
    if with_grid:
        add(
            "CRL 18-problem grid",
            GRID_PROBLEMS * TEMPORAL_METHODS * seeds * SECONDS["grid_per_run"],
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--with-grid", action="store_true")
    parser.add_argument("--speedup", type=float, default=1.0, help="GPU speedup vs the RTX 3050")
    args = parser.parse_args()
    rows = estimate(
        seeds=args.seeds,
        replications=args.replications,
        with_grid=args.with_grid,
        speedup=args.speedup,
    )
    total = sum(seconds for _, seconds in rows)
    print(f"{'Step':42s} {'hours':>8s}")
    print("-" * 52)
    for name, seconds in rows:
        print(f"{name:42s} {seconds / 3600:8.1f}")
    print("-" * 52)
    print(f"{'TOTAL':42s} {total / 3600:8.1f}  (~{total / 86400:.1f} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
