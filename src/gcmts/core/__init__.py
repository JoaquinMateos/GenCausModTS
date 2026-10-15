"""Core TSCM utilities, neural backbones, graph ops, solvers, and trainer."""

from gcmts.core import backbones, graph_ops, solvers
from gcmts.core.base import BaseModel, BaseTrainer
from gcmts.core.ema import EMA
from gcmts.core.history import plot_training_curves, save_history, save_training_report
from gcmts.core.profiling import PhaseProfiler
from gcmts.core.trainer import TRAIN_BATCH_KEYS, AdaptiveScheduler, SimpleTrainer, move_batch
from gcmts.core.tscm import LTSCM, TSCM
from gcmts.core.utils import (
    hungarian_match,
    lagged_adjacency_to_dag,
    make_mlp,
    mcc,
)

__all__ = [
    "BaseModel",
    "BaseTrainer",
    "SimpleTrainer",
    "AdaptiveScheduler",
    "EMA",
    "PhaseProfiler",
    "TRAIN_BATCH_KEYS",
    "move_batch",
    "TSCM",
    "LTSCM",
    "backbones",
    "graph_ops",
    "solvers",
    "save_history",
    "plot_training_curves",
    "save_training_report",
    "hungarian_match",
    "lagged_adjacency_to_dag",
    "make_mlp",
    "mcc",
]
