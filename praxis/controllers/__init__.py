from functools import partial

from praxis import registry
from praxis.controllers.attention import AttentionChanneler
from praxis.controllers.base import BaseController
from praxis.controllers.graph import GraphRouter
from praxis.controllers.layer_shuffle import LayerShuffle
from praxis.controllers.pathfinder import Pathfinder
from praxis.registry import Entry

registry.declare(
    "controllers",
    title="Layer-routing controllers",
    doc=(
        (
            "Decide which expert / block a token visits at each depth. Enables "
            "out-of-order layers and graph-style routing."
        )
    ),
    entries={
        "base": BaseController,
        "layer_shuffle": LayerShuffle,
        "graph": GraphRouter,
        "pathfinder": Pathfinder,
        "shortcutter": Entry(
            partial(Pathfinder, allow_early_exits=True),
            (
                "Pathfinder whose gates carry one extra choice past the last layer: "
                "when the batch votes for it, the pass exits early."
            ),
        ),
        "attention": AttentionChanneler,
        "counter_attention": Entry(
            partial(AttentionChanneler, max_tokens=5, initial_queries=3),
            (
                "The attention controller reading the last 5 tokens with 3 learned "
                "initial queries, rather than one token and one query."
            ),
        ),
    },
)
