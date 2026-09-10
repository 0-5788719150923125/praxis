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

# NO ACTIVATION PROFILES HERE. `dual_act`, `peer_dual`, `peer_split` and
# `peer_mix` used to live in this registry, each one a feedforward bound to a
# fixed activation choice. That put the choice somewhere no card could see it:
# the Arguments tab serializes the launch namespace, so a run whose experts
# gated through a servant/swish split still reported `activation: servant` and
# the only way to find out otherwise was to read this file.
#
# Those arms are now declared where they are visible, as slots on the activation
# argument (praxis/activations/__init__.py):
#
#   dual_act    ffn_type: glu       + activation: {gate: ..., value: gelu}
#   peer_dual   ffn_type: peer_glu  + activation: {gate: ..., value: gelu}
#   peer_split  ffn_type: peer_glu  + activation: {gate: ..., expert: {type:
#                                     mix_split, values: [servant, swish]}}
#   peer_mix    ffn_type: peer_glu  + activation: {gate: ..., expert: {type:
#                                     mix_gated, values: [serpent, swish, linear]}}
#
# What this registry holds now is feedforward STRUCTURE, which is the one thing
# a name here can carry that a config value cannot.
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
