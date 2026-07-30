from functools import partial

from praxis.dense.arc import ArcGLU
from praxis.dense.base import BaseDense
from praxis.dense.eml import EMLTree
from praxis.dense.glu import GatedLinearMLP
from praxis.dense.kan import KolmogorovArnoldNetwork
from praxis.dense.mlp import MultiLayerPerceptron
from praxis.dense.peer import ParameterEfficientExpertRetrieval
from praxis.dense.poly import PolynomialExpansionMLP
from praxis.dense.scatter import ScatterMLP
from praxis.dense.spline import SplineNetwork

DENSE_REGISTRY = dict(
    mlp=MultiLayerPerceptron,
    glu=GatedLinearMLP,
    arc=ArcGLU,
    poly=PolynomialExpansionMLP,
    scatter=ScatterMLP,
    kan=KolmogorovArnoldNetwork,
    peer=ParameterEfficientExpertRetrieval,
    # Gated experts: each retrieved expert becomes a GLU
    # (``up_e * (act(x . gate_e) * (x . down_e))``) instead of a rank-1
    # projection. The third bank row per expert is paid for out of the expert
    # COUNT, not the parameter budget, so this trades bank breadth for
    # per-expert expressiveness at a matched size.
    peer_glu=partial(ParameterEfficientExpertRetrieval, glu=True),
    eml_tree=EMLTree,
    spline=SplineNetwork,
)
