"""Concrete Causal Representation Learning methods.

Each method implements the identifiability result of its source paper and
subclasses :class:`~gcmts.causal_representation_learning.base.BaseCausalRepresentationLearner`.
"""

from gcmts.causal_representation_learning.methods.cadre import CaDRe
from gcmts.causal_representation_learning.methods.cegen import CEGEN
from gcmts.causal_representation_learning.methods.child import CHiLD
from gcmts.causal_representation_learning.methods.citris import CITRIS
from gcmts.causal_representation_learning.methods.icitris import iCITRIS
from gcmts.causal_representation_learning.methods.ivae import IVAE
from gcmts.causal_representation_learning.methods.leap import LEAP
from gcmts.causal_representation_learning.methods.mosaic import MOSAIC
from gcmts.causal_representation_learning.methods.nctrl import NCTRL
from gcmts.causal_representation_learning.methods.slow_flows import SlowFlows
from gcmts.causal_representation_learning.methods.tdrl import TDRL
from gcmts.causal_representation_learning.methods.tdrl_official import TDRLOfficial
from gcmts.causal_representation_learning.methods.ivae_official import IVAEOfficial
from gcmts.causal_representation_learning.methods.nctrl_official import NCTRLOfficial
from gcmts.causal_representation_learning.methods.mosaic_official import MOSAICOfficial
from gcmts.causal_representation_learning.methods.leap_official import LEAPOfficial

__all__ = [
    "CaDRe",
    "CEGEN",
    "CHiLD",
    "CITRIS",
    "iCITRIS",
    "IVAE",
    "LEAP",
    "MOSAIC",
    "NCTRL",
    "SlowFlows",
    "TDRL",
    "TDRLOfficial",
    "IVAEOfficial",
    "NCTRLOfficial",
    "MOSAICOfficial",
    "LEAPOfficial",
]
