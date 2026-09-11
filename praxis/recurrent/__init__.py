from praxis import registry
from praxis.recurrent.gru import GRU
from praxis.recurrent.min_gru import MinGRU

registry.declare(
    "recurrent",
    title="Recurrent cells",
    doc=(
        "Minimal gated recurrent cells (GRU, MinGRU). Used by the recurrent block "
        "types and as a sequence mixer inside the byte-latent encoder."
    ),
    entries={
        "min_gru": MinGRU,
        "gru": GRU,
    },
)
