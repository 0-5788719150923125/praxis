from functools import partial

from praxis.controllers.base import BaseController

# from praxis.controllers.autopilot import Autopilot
from praxis.controllers.graph import GraphRouter
from praxis.controllers.layer_shuffle import LayerShuffle
from praxis.controllers.pathfinder import Pathfinder

CONTROLLER_REGISTRY = dict(
    base=BaseController,
    layer_shuffle=LayerShuffle,
    graph=GraphRouter,
    # autopilot=Autopilot,
    pathfinder=Pathfinder,
    shortcutter=partial(Pathfinder, allow_early_exits=True),
)
