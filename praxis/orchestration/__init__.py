from typing import Any, Dict, Type

from torch import nn

from praxis.dense import DENSE_REGISTRY
from praxis.layers import LocalLayer, RemoteLayer

EXPERT_REGISTRY: Dict[str, Type[nn.Module]] = {**DENSE_REGISTRY}

# ORCHESTRATION_REGISTRY is kept for backward compatibility
# Actual orchestration management (like Hivemind) is now handled via integrations
ORCHESTRATION_REGISTRY: Dict[str, Any] = {}
