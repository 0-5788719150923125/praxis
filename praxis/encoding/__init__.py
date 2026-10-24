from praxis import registry
from praxis.encoding.alibi import ALiBi
from praxis.encoding.archope import ArcHoPE
from praxis.encoding.hope import HoPE
from praxis.encoding.learned_rates import LearnedRateRoPE
from praxis.encoding.nope import NoPE
from praxis.encoding.rope import RoPE
from praxis.registry import Entry

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
        "rope_learned_rates": Entry(
            LearnedRateRoPE,
            (
                "RoPE with a learned rate for every rotary pair of every KV head, "
                "initialized on the geometric schedule, plus a per-depth shift. "
                "Kuramoto Attention's per-oscillator natural frequencies "
                "(arXiv:2606.11585). Logs the mean rate per layer."
            ),
        ),
        "hope": HoPE,
        "arc": ArcHoPE,
    },
)
