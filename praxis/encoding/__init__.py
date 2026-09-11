from praxis import registry
from praxis.encoding.alibi import ALiBi
from praxis.encoding.archope import ArcHoPE
from praxis.encoding.hope import HoPE
from praxis.encoding.nope import NoPE
from praxis.encoding.rope import RoPE

registry.declare(
    "encoding",
    title="Positional encoding",
    doc=(
        (
            "RoPE, ALiBi, NoPE and friends - the rotational / additive position priors "
            "injected into attention."
        )
    ),
    entries={
        "nope": NoPE,
        "alibi": ALiBi,
        "rope": RoPE,
        "hope": HoPE,
        "arc": ArcHoPE,
    },
)
