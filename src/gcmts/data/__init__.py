"""Data generators and loaders."""

from gcmts.data.base import BaseDataGenerator
from gcmts.data.causal3dident import Causal3DIdentImageGenerator
from gcmts.data.real import RealTimeSeries, RealTSGenerator
from gcmts.data.semi_synthetic import ACICLoader, IHDPGenerator, IHDPLoader, TumourGrowthSimulator
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
from gcmts.data.voronoi import VoronoiGenerator

__all__ = [
    "ACICLoader",
    "BaseDataGenerator",
    "Causal3DIdentImageGenerator",
    "IHDPLoader",
    "IHDPGenerator",
    "RealTimeSeries",
    "RealTSGenerator",
    "TumourGrowthSimulator",
    "VoronoiGenerator",
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
