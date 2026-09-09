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

# The two activation banks these profiles run. Written out here, next to the
# profiles that use them, because the whole point of the `{type, values}` spec
# is that a reader can see what a name means without opening another file.
#
# The line's own periodic activation against a non-periodic one, nothing else -
# the bank PEER's expert-index split was built to test.
SPLIT_BANK = ["servant", "swish"]
# Periodic, non-periodic, pass-through. The pass-through matters: it lets the
# model decline both function classes for a feature rather than being forced to
# pick one.
HARMONIC_BANK = ["serpent", "swish", "linear"]

DENSE_REGISTRY = dict(
    mlp=MultiLayerPerceptron,
    glu=GatedLinearMLP,
    # A GLU with BOTH halves activated instead of one. The plain GLU multiplies
    # an activated gate by a LINEAR value branch, so half the channels never
    # meet a nonlinearity; this puts a non-periodic one there (gelu) against the
    # periodic gate `config.activation` supplies. Two multiplied function
    # classes rather than one steering a linear half. Parameter-identical to
    # `glu` - gelu carries none.
    dual_act=partial(GatedLinearMLP, activation_value="gelu"),
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
    # `dual_act`'s change applied to the PEER expert instead of the dense FFN,
    # so an ablation against `peer_glu` is one variable and does not also swap
    # the expert-retrieval feedforward out.
    peer_dual=partial(
        ParameterEfficientExpertRetrieval, glu=True, activation_value="gelu"
    ),
    # peer_glu with HETEROGENEOUS gates: half the expert bank activates with
    # Servant (periodic) and half with swish. Not a capacity change and not an
    # extra nonlinearity - the GLU's linear value branch survives untouched, so
    # this is one variable away from `peer_glu` and TWO away from `peer_dual`,
    # which fills that linear slot instead.
    #
    # The hypothesis it tests is coverage, not depth: `peer_dual` asked whether
    # a second function class helps when stacked on top of the first, and the
    # answer came back confounded and expensive. This asks the cheaper question
    # of whether the model wants BOTH classes available side by side, spending
    # half its periodic budget to get a non-periodic one. Parameter-identical to
    # `peer_glu` - swish carries none.
    #
    # The split itself is not PEER's business: `mix_split` is an
    # `ActivationMixture` keyed on an external index, and all PEER does is hand
    # it each element's position in the bank. The bank names `servant` outright
    # rather than inheriting `config.activation`, so the profile means the same
    # thing under any `--activation` (it matches abstractinator-a, which is what
    # this line runs).
    peer_split=partial(
        ParameterEfficientExpertRetrieval,
        glu=True,
        activation={"type": "mix_split", "values": SPLIT_BANK},
    ),
    # The CONTINUOUS arm against `peer_split`'s discrete one. Both ask whether
    # the model wants more than one function class available; `peer_split`
    # answers by freezing a class onto each half of the bank at init, this one
    # by letting every element blend between them and re-decide per token. Same
    # module, same bank size, different coefficient source - so the pair is a
    # one-variable ablation rather than two experiments.
    #
    # NOT parameter-identical to `peer_glu`: the gate carries 2N scalars per
    # mixture (6 here) and Serpent's per-feature spectrum is materialized inside
    # the bank rather than at the top level. Both are rounding error against the
    # bank, but they are not zero.
    peer_mix=partial(
        ParameterEfficientExpertRetrieval,
        glu=True,
        activation={"type": "mix_gated", "values": HARMONIC_BANK},
    ),
    eml_tree=EMLTree,
    spline=SplineNetwork,
)
