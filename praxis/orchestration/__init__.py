from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.dense import DENSE_REGISTRY
from praxis.orchestration.experts import LocalExpert, RemoteExpert
from praxis.recurrent import RECURRENT_REGISTRY
from praxis.routers import ROUTER_REGISTRY

EXPERT_REGISTRY: Dict[str, Type[nn.Module]] = {**DENSE_REGISTRY}

# ORCHESTRATION_REGISTRY is kept for backward compatibility
# Actual orchestration management (like Hivemind) is now handled via integrations
ORCHESTRATION_REGISTRY: Dict[str, Any] = {}
