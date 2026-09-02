from functools import partial

from praxis.dense.arc import ArcGLU
from praxis.dense.base import BaseDense
from praxis.dense.eml import EMLTree
from praxis.dense.dual_act import DualActivationMLP
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
    dual_act=DualActivationMLP,
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
    # peer_glu with BOTH halves activated instead of one. The plain GLU expert
    # multiplies an activated gate by a LINEAR value branch, so half the
    # channels never meet a nonlinearity; this puts a non-periodic one there
    # (gelu) against the periodic gate `config.activation` supplies. Two
    # multiplied function classes rather than one steering a linear half - the
    # PEER-preserving twin of `dual_act`, so an ablation against `peer_glu` is
    # one variable and does not also swap the expert-retrieval FFN out.
    # Parameter-identical: gelu carries no parameters.
    peer_dual=partial(ParameterEfficientExpertRetrieval, glu=True, act_value="gelu"),
    # peer_glu with HETEROGENEOUS gates: half the retrieval heads activate with
    # `config.activation` (periodic, here Servant) and half with swish. Not a
    # capacity change and not an extra nonlinearity - the GLU's linear value
    # branch survives untouched, so this is one variable away from `peer_glu`
    # and TWO away from `peer_dual`, which fills that linear slot instead.
    #
    # The hypothesis it tests is coverage, not depth: `peer_dual` asked whether
    # a second function class helps when stacked on top of the first, and the
    # answer came back confounded and expensive. This asks the cheaper question
    # of whether the model wants BOTH classes available side by side, spending
    # half its periodic budget to get a non-periodic one. Parameter-identical to
    # `peer_glu` - swish carries none.
    peer_split=partial(ParameterEfficientExpertRetrieval, glu=True, act_alt="swish"),
    eml_tree=EMLTree,
    spline=SplineNetwork,
)
