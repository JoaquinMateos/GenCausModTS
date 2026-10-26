"""Data generators and loaders."""

from gcmts.data.base import BaseDataGenerator
from gcmts.data.binary_scm import BinarySCMGenerator
from gcmts.data.canonical_static import CPSLoader, NewsLoader, TwinsLoader, load_jobs
from gcmts.data.causal3dident import Causal3DIdentImageGenerator
from gcmts.data.concept_ts import ConceptTimeSeriesGenerator
from gcmts.data.control import MountainCarGenerator, PendulumGenerator
from gcmts.data.hierarchical import HierarchicalTemporalGenerator
from gcmts.data.linear_processes import OUProcessGenerator, VARMAProcessGenerator
from gcmts.data.lorenz import LorenzGenerator
from gcmts.data.multi_objective import MultiObjectiveMDPGenerator
from gcmts.data.physics import PhysicsSCMGenerator
from gcmts.data.pool import PooledGenerator
from gcmts.data.propagation import PropagationGenerator
from gcmts.data.real import RealTimeSeries, RealTSGenerator
from gcmts.data.real_formats import (
    PerturbationScreen,
    load_clinical_csv,
    load_mocap,
    load_nifti,
    load_single_cell,
    netcdf_to_csv,
)
from gcmts.data.real_panels import (
    EffectCohort,
    build_covid_panel,
    load_lalonde,
    load_nhefs,
    load_real_dataset,
)
from gcmts.data.real_registry import REAL_DATASETS, RealDataset, real_by_role, real_label
from gcmts.data.sem_effect import TimeVaryingSEMGenerator
from gcmts.data.semi_synthetic import ACICLoader, IHDPGenerator, IHDPLoader, TumourGrowthSimulator
from gcmts.data.static_effect import STATIC_CATE_BENCHMARKS, StaticCateLoader
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
from gcmts.data.synthetic_store import (
    DatasetRecord,
    dataset_exists,
    dataset_path,
    load_dataset,
    read_manifest,
    save_dataset,
)
from gcmts.data.voronoi import VoronoiGenerator

__all__ = [
    "ACICLoader",
    "BaseDataGenerator",
    "BinarySCMGenerator",
    "Causal3DIdentImageGenerator",
    "CPSLoader",
    "NewsLoader",
    "TwinsLoader",
    "load_jobs",
    "ConceptTimeSeriesGenerator",
    "HierarchicalTemporalGenerator",
    "MultiObjectiveMDPGenerator",
    "PropagationGenerator",
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
    "TimeVaryingSEMGenerator",
    "LorenzGenerator",
    "PhysicsSCMGenerator",
    "PooledGenerator",
    "PendulumGenerator",
    "DatasetRecord",
    "save_dataset",
    "load_dataset",
    "dataset_path",
    "dataset_exists",
    "read_manifest",
    "MountainCarGenerator",
    "VARMAProcessGenerator",
    "OUProcessGenerator",
    "StaticCateLoader",
    "STATIC_CATE_BENCHMARKS",
    "REAL_DATASETS",
    "RealDataset",
    "real_by_role",
    "real_label",
    "EffectCohort",
    "build_covid_panel",
    "load_lalonde",
    "load_nhefs",
    "load_oregon",
    "load_real_dataset",
    "PerturbationScreen",
    "netcdf_to_csv",
    "load_single_cell",
    "load_mocap",
    "load_nifti",
    "load_clinical_csv",
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
