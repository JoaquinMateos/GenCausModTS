"""Optuna hyperparameter search for the shared CRL methods.

The final benchmark protocol is fixed separately; this script only chooses
method hyperparameters on a held-out seed of a representative NP/VAR process.
It writes a JSON mapping usable with
``scripts/run_crl_shared.py --hpo-config``.

Example::

    uv run python scripts/tune_crl_methods.py \
        --methods iVAE LEAP TDRL NCTRL MOSAIC CaDRe \
        --trials 8 --epochs 250 --device cuda --out configs/crl_hpo.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from gcmts.evaluation.report import save_rows

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_crl_shared as runner  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("gcmts.tune_crl")


def _search_space(method: str, trial: optuna.Trial) -> dict[str, Any]:
    if method == "iVAE":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 100, 128]),
            "n_layers": trial.suggest_int("n_layers", 2, 4),
            "activation": trial.suggest_categorical("activation", ["xtanh", "lrelu"]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "decoder_var": trial.suggest_categorical("decoder_var", [0.01, 0.1, 1.0]),
            "x_scale": trial.suggest_categorical("x_scale", [1.0, 4.0]),
        }
    if method == "LEAP":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "beta": trial.suggest_float("beta", 5e-4, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5e-2, log=True),
            "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        }
    if method == "TDRL":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [4, 8, 16]),
            "beta": trial.suggest_float("beta", 5e-4, 5e-2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
            "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        }
    if method == "TDRL":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [2, 4, 8]),
            "beta": trial.suggest_float("beta", 5e-4, 2e-2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        }
    if method == "NCTRL":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "beta": trial.suggest_float("beta", 5e-4, 5e-2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
            "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
            "hmm_lr": trial.suggest_float("hmm_lr", 3e-4, 3e-3, log=True),
            "hmm_mode": trial.suggest_categorical("hmm_mode", ["mle_scaled:H", "mle_scaled:Ht"]),
        }
    if method == "MOSAIC":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [4, 8, 16]),
            "hidden_per_z": trial.suggest_categorical("hidden_per_z", [4, 8, 16]),
            "beta": trial.suggest_float("beta", 5e-4, 5e-2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
            "freeze_epochs": trial.suggest_int("freeze_epochs", 1, 8),
        }
    if method == "CaDRe":
        return {
            "kl_z_weight": trial.suggest_float("kl_z_weight", 1e-3, 1e-1, log=True),
            "kl_s_weight": trial.suggest_float("kl_s_weight", 1e-3, 5e-2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-3, log=True),
            "beta": trial.suggest_float("beta", 1e-5, 1e-3, log=True),
            "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        }
    raise ValueError(f"unknown method {method!r}")


def _objective(
    method: str,
    suite_names: list[str],
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        params = _search_space(method, trial)
        runner.configure_hpo(None)
        runner.HPO_CONFIG[method] = params
        scores: list[float] = []
        try:
            for suite_name in suite_names:
                methods = runner.build_methods(runner.SUITES[suite_name], epochs_scale=1.0)
                spec = next((m for m in methods if m[0] == method), None)
                if spec is None:
                    raise ValueError(f"{method} is not in the temporal shared block")
                _, factory, _ = spec
                with tempfile.TemporaryDirectory() as tmp:
                    row = runner._run(
                        runner.SUITES[suite_name],
                        method,
                        factory,
                        epochs,
                        seed,
                        device,
                        batch_size,
                        Path(tmp),
                        save_curves=False,
                        scheduler="adaptive",
                        early_stopping_patience=min(100, max(20, epochs // 4)),
                        restore_best_weights=True,
                        kl_warmup_epochs=min(50, max(1, epochs // 5)),
                    )
                score = row.mcc
                if not isinstance(score, (int, float)) or score != score:
                    logger.warning("%s trial %d produced a non-finite score", method, trial.number)
                    return float("-inf")
                if row.shd is not None and row.shd == row.shd:
                    score -= 0.01 * float(row.shd)
                scores.append(float(score))
        except Exception as exc:  # noqa: BLE001 - unstable trial is a failed trial
            logger.warning("%s trial %d failed: %s", method, trial.number, exc)
            return float("-inf")
        mean_score = float(np.mean(scores))
        logger.info(
            "%s trial %d mean_score=%.4f suites=%s params=%s",
            method, trial.number, mean_score, suite_names, params,
        )
        return mean_score

    return objective


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["iVAE", "LEAP", "TDRL", "NCTRL", "MOSAIC", "CaDRe"])
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["npvar", "causal3d", "cartpole"],
        choices=sorted(runner.SUITES),
        help="representative suites averaged in the HPO objective",
    )
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("configs/crl_hpo.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload: dict[str, Any] = {
        "suites": args.suites,
        "seed": args.seed,
        "epochs": args.epochs,
        "methods": {},
    }
    for method in args.methods:
        study = optuna.create_study(direction="maximize", study_name=f"crl-{method}")
        study.optimize(
            _objective(method, args.suites, args.seed, args.epochs, args.batch_size, args.device),
            n_trials=args.trials,
        )
        payload["methods"][method] = dict(study.best_params)
        payload["methods"][method]["hpo_score"] = float(study.best_value)
        logger.info("BEST %s %.4f %s", method, study.best_value, study.best_params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
