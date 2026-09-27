"""Simple, reproducible CRL benchmarks.

Trains each implemented CRL method on a matching synthetic benchmark, checks
that inputs/outputs are well-formed, evaluates representation-recovery metrics
with the library's :class:`CrlBenchmarkRunner`, and writes logs + a results
table.

Usage::

    uv run python scripts/run_crl_benchmarks.py
    uv run python scripts/run_crl_benchmarks.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.causal_representation_learning.methods import (
    CEGEN,
    CITRIS,
    IVAE,
    LEAP,
    MOSAIC,
    NCTRL,
    TDRL,
    SlowFlows,
)
from gcmts.core import SimpleTrainer, move_batch
from gcmts.core.graph_ops import estimate_latent_adjacency
from gcmts.data.synthetic import (
    InterventionalTemporalGenerator,
    LinearSDEGenerator,
    NCTRLGenerator,
    NonlinearICAGenerator,
    SlowFeatureGenerator,
    TDRLGenerator,
    TemporalNonlinearGenerator,
)
from gcmts.evaluation import BenchmarkTask, CrlBenchmarkRunner
from gcmts.evaluation.metrics import SHDMetric, WSHDMetric

logger = logging.getLogger("gcmts.benchmarks")


@dataclass
class CaseConfig:
    name: str
    model_factory: Callable[[], BaseCausalRepresentationLearner]
    generator_factory: Callable[[], Any]
    epochs: int
    lr: float = 1e-3
    kind: str = "crl"  # "crl" (mixing model) or "sde" (observed-space process)
    note: str = ""


@dataclass
class CaseResult:
    case: str
    method: str
    epochs: int
    n_parameters: int
    final_loss: float
    recon_mse: float
    mcc: float
    r2_matched: float
    r2_sep: float | None
    dci_disentanglement: float | None
    dci_completeness: float | None
    dci_informativeness: float | None
    shd: float | None
    wshd: float | None
    w2: float | None
    z_shape_ok: bool
    roundtrip_ok: bool
    finite: bool
    seconds: float
    device: str = "auto"
    note: str = ""


def build_cases(quick: bool) -> list[CaseConfig]:
    scale = 0.15 if quick else 1.0
    cases = [
        CaseConfig(
            name="ivae_linear",
            model_factory=lambda: IVAE(observed_dim=5, latent_dim=3, u_dim=10),
            generator_factory=lambda: NonlinearICAGenerator(
                observed_dim=5, latent_dim=3, horizon=1, n_regimes=10, mixing="linear", seed=0
            ),
            epochs=int(2000 * scale),
            note="iVAE, 10 regimes >= nk+1 required for identifiability",
        ),
        CaseConfig(
            name="ivae_nonlinear",
            model_factory=lambda: IVAE(observed_dim=5, latent_dim=3, u_dim=10),
            generator_factory=lambda: NonlinearICAGenerator(
                observed_dim=5, latent_dim=3, horizon=1, n_regimes=10, mixing="nonlinear", seed=0
            ),
            epochs=int(1500 * scale),
            note="nonlinear tanh mixing, 10 regimes (harder)",
        ),
        CaseConfig(
            name="leap_temporal",
            model_factory=lambda: LEAP(observed_dim=6, latent_dim=3, max_lag=1, u_dim=4),
            generator_factory=lambda: TemporalNonlinearGenerator(
                observed_dim=6,
                latent_dim=3,
                horizon=16,
                max_lag=1,
                n_regimes=4,
                regime_mode="time",
                mixing="linear",
                seed=0,
            ),
            epochs=int(800 * scale),
            note="time-varying regime, linear mixing, regime-dependent noise",
        ),
        CaseConfig(
            name="slow_flows",
            model_factory=lambda: SlowFlows(observed_dim=4, n_flow_layers=6),
            generator_factory=lambda: SlowFeatureGenerator(observed_dim=4, horizon=16, seed=0),
            epochs=int(600 * scale),
            note="random-walk slow features, component-wise mixing",
        ),
        CaseConfig(
            name="citris_interventional",
            model_factory=lambda: CITRIS(
                observed_dim=5, n_vars=2, var_dim=2, n_shared=1, n_flow_layers=4
            ),
            generator_factory=lambda: InterventionalTemporalGenerator(
                n_vars=2, var_dim=2, n_shared=1, horizon=12, seed=0
            ),
            epochs=int(400 * scale),
            note="intervention-conditioned temporal process",
        ),
        CaseConfig(
            name="tdrl_modular",
            model_factory=lambda: TDRL(
                observed_dim=6,
                latent_fix_dim=1,
                latent_dyn_dim=2,
                latent_obs_dim=1,
                u_dim=3,
            ),
            generator_factory=lambda: TDRLGenerator(
                observed_dim=6,
                fix_dim=1,
                chg_dim=2,
                obs_dim=1,
                horizon=16,
                n_regimes=3,
                seed=0,
            ),
            epochs=int(1000 * scale),
            note="TDRL generative model: fixed/changing/observation blocks",
        ),
        CaseConfig(
            name="nctrl_regimes",
            model_factory=lambda: NCTRL(observed_dim=6, latent_dim=3, n_regimes=3),
            generator_factory=lambda: NCTRLGenerator(
                observed_dim=6, latent_dim=3, horizon=16, n_regimes=3, seed=0
            ),
            epochs=int(1000 * scale),
            note="unknown HMM regimes inferred from observations only",
        ),
        CaseConfig(
            name="mosaic_modules",
            model_factory=lambda: MOSAIC(observed_dim=5, latent_dim=3, u_dim=2),
            generator_factory=lambda: TemporalNonlinearGenerator(
                observed_dim=5,
                latent_dim=3,
                horizon=8,
                n_regimes=2,
                mixing="linear",
                seed=0,
            ),
            epochs=int(1000 * scale),
            note="sparse additive module-support recovery",
        ),
        CaseConfig(
            name="cegen_sde",
            model_factory=lambda: CEGEN(observed_dim=4, n_regions=4),
            generator_factory=lambda: LinearSDEGenerator(observed_dim=4, horizon=20, seed=0),
            epochs=int(800 * scale),
            kind="sde",
            note="linear SDE drift/diffusion via conditional W2",
        ),
    ]
    return cases


def _check_outputs(
    model: BaseCausalRepresentationLearner, batch: Any
) -> tuple[bool, bool, bool]:
    """Verify forward/encode/decode shapes and finiteness."""
    outputs = model.forward(batch)
    z = outputs.z
    x_recon = outputs.x_recon
    z_shape_ok = z.shape == batch.z.shape and z.shape[-1] == model.latent_dim
    finite = bool(torch.isfinite(z).all())
    z_enc = model.encode(batch.x, batch.context)
    finite = finite and bool(torch.isfinite(z_enc).all())
    roundtrip_ok = True
    if x_recon is not None:
        roundtrip_ok = x_recon.shape == batch.x.shape
        finite = finite and bool(torch.isfinite(x_recon).all())
    logger.info(
        "shape checks | z=%s (true %s) | x_recon=%s | z_shape_ok=%s roundtrip_ok=%s finite=%s",
        tuple(z.shape),
        tuple(batch.z.shape),
        None if x_recon is None else tuple(x_recon.shape),
        z_shape_ok,
        roundtrip_ok,
        finite,
    )
    return z_shape_ok, roundtrip_ok, finite


def run_case(cfg: CaseConfig, device: str = "auto", batch_size: int = 256) -> CaseResult:
    logger.info("=" * 72)
    logger.info("CASE %s | %s", cfg.name, cfg.note)
    torch.manual_seed(0)

    generator = cfg.generator_factory()
    model = cfg.model_factory()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "model=%s | params=%d | epochs=%d | batch=%d",
        type(model).__name__,
        n_params,
        cfg.epochs,
        batch_size,
    )

    probe = generator.sample(16)
    z_shape_ok, roundtrip_ok, finite = _check_outputs(model, probe)

    trainer = SimpleTrainer(
        max_epochs=max(cfg.epochs, 1),
        lr=cfg.lr,
        batch_size=batch_size,
        steps_per_epoch=15,
        log_every=max(cfg.epochs // 10, 1),
        device=device,
        seed=0,
    )
    start = time.perf_counter()
    history = trainer.fit(model, generator)
    seconds = time.perf_counter() - start
    final_loss = history["loss"][-1] if history.get("loss") else float("nan")

    resolved_device = trainer.resolve_device(model)
    eval_batch = move_batch(generator.sample(2048), resolved_device)

    r2_sep_value: float | None = None
    dci_dis: float | None = None
    dci_comp: float | None = None
    dci_info: float | None = None
    shd_value: float | None = None
    wshd_value: float | None = None

    if cfg.kind == "sde":
        with torch.no_grad():
            outputs = model.forward(eval_batch)
            extras = outputs.extras or {}
            w2_value = float(extras.get("w2", torch.tensor(float("nan"))))
            assert outputs.x_recon is not None
            recon_mse = float(((outputs.x_recon - eval_batch.x) ** 2).mean())
        mcc_value = float("nan")
        r2_value = float("nan")
        finite = finite and torch.isfinite(torch.tensor(w2_value)).item()
        logger.info(
            "RESULT %s | w2=%.4f one_step_mse=%.4f loss=%.4f (%.1fs)",
            cfg.name,
            w2_value,
            recon_mse,
            final_loss,
            seconds,
        )
    else:
        task = BenchmarkTask(
            name=cfg.name,
            model=model,
            data=eval_batch,
            metrics=[
                "mcc",
                "r2_diag",
                "r2_sep",
                "dci_disentanglement",
                "dci_completeness",
                "dci_informativeness",
            ],
            causal_level="L1_association",
        )
        result = CrlBenchmarkRunner().run(task)
        scores = {m.name: m.value for m in result.metrics}
        with torch.no_grad():
            z_pred = model.encode(eval_batch.x, eval_batch.context)
            x_recon = model.decode(z_pred)
        recon_mse = float(((x_recon - eval_batch.x) ** 2).mean())
        finite = finite and all(
            torch.isfinite(torch.tensor(v)).item() for v in scores.values()
        )
        mcc_value = scores.get("mcc", float("nan"))
        r2_value = scores.get("r2_diag", float("nan"))
        r2_sep_value = scores.get("r2_sep")
        dci_dis = scores.get("dci_disentanglement")
        dci_comp = scores.get("dci_completeness")
        dci_info = scores.get("dci_informativeness")

        adj_true = eval_batch.adjacency
        if adj_true is not None:
            max_lag = int(getattr(generator, "max_lag", 1))
            native = model.get_latent_adjacency()
            if native is not None and tuple(native.shape) == tuple(adj_true.shape):
                adj_est = native
            else:
                adj_est = estimate_latent_adjacency(z_pred, max_lag=max_lag)
            shd_value = SHDMetric()(adj_est, adj_true).value
            wshd_value = WSHDMetric()(adj_est, adj_true).value

        if mcc_value < 0.2:
            logger.warning(
                "CASE %s MCC=%.3f is near chance; check configuration/training.",
                cfg.name,
                mcc_value,
            )
        graph_msg = (
            ""
            if shd_value is None
            else f" shd={shd_value:.1f} wshd={wshd_value:.2f}"
        )
        logger.info(
            "RESULT %s | mcc=%.3f r2_diag=%.3f r2_sep=%.3f dci=(%.3f/%.3f/%.3f)%s "
            "recon_mse=%.4f loss=%.4f (%.1fs)",
            cfg.name,
            mcc_value,
            r2_value,
            float("nan") if r2_sep_value is None else r2_sep_value,
            float("nan") if dci_dis is None else dci_dis,
            float("nan") if dci_comp is None else dci_comp,
            float("nan") if dci_info is None else dci_info,
            graph_msg,
            recon_mse,
            final_loss,
            seconds,
        )

    return CaseResult(
        case=cfg.name,
        method=type(model).__name__,
        epochs=cfg.epochs,
        n_parameters=n_params,
        final_loss=final_loss,
        recon_mse=recon_mse,
        mcc=mcc_value,
        r2_matched=r2_value,
        r2_sep=r2_sep_value,
        dci_disentanglement=dci_dis,
        dci_completeness=dci_comp,
        dci_informativeness=dci_info,
        shd=shd_value,
        wshd=wshd_value,
        w2=w2_value if cfg.kind == "sde" else None,
        z_shape_ok=z_shape_ok,
        roundtrip_ok=roundtrip_ok,
        finite=finite,
        seconds=seconds,
        device=str(resolved_device),
        note=cfg.note,
    )


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, stream_handler]


def save_results(results: list[CaseResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    (out_dir / "crl_benchmarks.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (out_dir / "crl_benchmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0].keys()))
        writer.writeheader()
        writer.writerows(payload)
    logger.info("Saved results to %s", out_dir)


def print_table(results: list[CaseResult]) -> None:
    header = (
        f"{'case':<22}{'method':<10}{'params':>9}{'loss':>9}"
        f"{'recon':>9}{'mcc':>7}{'r2d':>7}{'r2s':>7}{'dciD':>7}{'dciC':>7}{'dciI':>7}"
        f"{'shd':>6}{'wshd':>7}{'w2':>8}{'sec':>7}{'device':>7}"
    )
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        w2 = f"{r.w2:>8.3f}" if r.w2 is not None else f"{'—':>8}"

        def _fmt(value: float | None) -> str:
            return f"{'—':>7}" if value is None else f"{value:>7.3f}"

        def _gfmt(value: float | None, width: int) -> str:
            return f"{'—':>{width}}" if value is None else f"{value:>{width}.1f}"

        logger.info(
            f"{r.case:<22}{r.method:<10}{r.n_parameters:>9}{r.final_loss:>9.3f}"
            f"{r.recon_mse:>9.4f}{r.mcc:>7.3f}{r.r2_matched:>7.3f}{_fmt(r.r2_sep)}"
            f"{_fmt(r.dci_disentanglement)}{_fmt(r.dci_completeness)}"
            f"{_fmt(r.dci_informativeness)}{_gfmt(r.shd, 6)}{_gfmt(r.wshd, 7)}"
            f"{w2}{r.seconds:>7.1f}{r.device:>7}"
        )
    logger.info("-" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simple CRL benchmarks.")
    parser.add_argument("--quick", action="store_true", help="fewer epochs (smoke run)")
    parser.add_argument("--only", nargs="*", default=None, help="subset of case names")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--device",
        default="auto",
        help="device for training: 'auto' (default), 'cpu', 'cuda', 'cuda:0', ...",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size (larger improves GPU utilisation)",
    )
    args = parser.parse_args()

    configure_logging(args.out_dir / "crl_benchmarks.log")
    logger.info("Starting CRL benchmarks (quick=%s, batch=%d)", args.quick, args.batch_size)

    cases = build_cases(args.quick)
    if args.only:
        cases = [c for c in cases if c.name in set(args.only)]
        if not cases:
            logger.error("No cases matched --only %s", args.only)
            return 2
    results = [run_case(cfg, device=args.device, batch_size=args.batch_size) for cfg in cases]

    print_table(results)
    save_results(results, args.out_dir)

    all_ok = all(r.z_shape_ok and r.roundtrip_ok and r.finite for r in results)
    logger.info("Sanity checks passed for all cases: %s", all_ok)
    if not all_ok:
        logger.error("Some cases failed input/output sanity checks.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
