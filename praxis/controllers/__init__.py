from functools import partial

from praxis.controllers.attention import AttentionChanneler
from praxis.controllers.base import BaseController
from praxis.controllers.graph import GraphRouter
from praxis.controllers.layer_shuffle import LayerShuffle
from praxis.controllers.neural import NeuralController
from praxis.controllers.pathfinder import Pathfinder

CONTROLLER_REGISTRY = dict(
    base=BaseController,
    layer_shuffle=LayerShuffle,
    graph=GraphRouter,
    pathfinder=Pathfinder,
    shortcutter=partial(Pathfinder, allow_early_exits=True),
    neural=NeuralController,
    attention=AttentionChanneler,
    counter_attention=partial(AttentionChanneler, max_tokens=5, initial_queries=3),
)
