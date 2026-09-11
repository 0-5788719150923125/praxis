"""One --precision flag, three knobs kept in agreement.

The failure this pins down is drift: the model built in one dtype, Lightning
stepping in another, and the matmul policy set from a third place. Every
consumer resolves the same profile from the same registry, so what is asserted
here is that the flag reaches all three - model dtype, Lightning precision
string, float32 matmul policy - and that a level the hardware cannot honor is
downgraded to something that runs rather than exploding mid-step.
"""

from praxis import registry


def test_only_float64_forbids_tf32():
    """ "medium" is defined as permitting a bf16 internal datatype, so no
    profile uses it: the fp32-carrying levels take TF32 via "high", and the
    level whose entire premise is arithmetic width takes none of it."""
    for name, profile in registry.namespace("precision").items():
        assert profile.matmul == ("highest" if name == "float64" else "high")


def test_every_profile_is_internally_coherent():
    """A profile that names a param dtype must run the trainer in the matching
    precision - a bf16 model stepped by a 32-true trainer is the exact drift
    this registry exists to prevent."""
    expected = {"float64": "64-true", "bfloat16": "bf16-true", "float16": "16-true"}
    for name, profile in registry.namespace("precision").items():
        assert profile.name == name
        if profile.param_dtype is not None:
            assert profile.lightning == expected[profile.param_dtype]
        else:
            # No cast means fp32 master weights: either plain fp32 or a mixed
            # plugin that keeps them.
            assert profile.lightning in ("32-true", "16-mixed")
