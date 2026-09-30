"""Data generators and loaders."""

from gcmts.data.base import BaseDataGenerator
from gcmts.data.real import RealTimeSeries, RealTSGenerator
from gcmts.data.synthetic import (
    CartPoleGenerator,
    Causal3DIdentGenerator,
    InterventionalTemporalGenerator,
    LinearSDEGenerator,
    LTSCMGenerator,
    NCTRLGenerator,
    NonlinearICAGenerator,
    SlowFeatureGenerator,
    TDRLGenerator,
    TemporalNonlinearGenerator,
    VARGenerator,
)
from gcmts.data.synthetic_effect import HarmonicOscillatorGenerator
from gcmts.data.synthetic_sdd import EvolvingDomainGenerator, MultiDomainGenerator

__all__ = [
    "BaseDataGenerator",
    "RealTimeSeries",
    "RealTSGenerator",
    "CartPoleGenerator",
    "Causal3DIdentGenerator",
    "EvolvingDomainGenerator",
    "HarmonicOscillatorGenerator",
    "InterventionalTemporalGenerator",
    "LinearSDEGenerator",
    "LTSCMGenerator",
    "MultiDomainGenerator",
    "NCTRLGenerator",
    "NonlinearICAGenerator",
    "SlowFeatureGenerator",
    "TDRLGenerator",
    "TemporalNonlinearGenerator",
    "VARGenerator",
]
