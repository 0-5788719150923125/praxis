from typing import Any, Dict, List, Optional, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.blocks import BLOCK_REGISTRY
from praxis.compression import COMPRESSION_REGISTRY
from praxis.controllers import CONTROLLER_REGISTRY
from praxis.experimental.evolution import GenomicBottleneck
from praxis.orchestration import (
    EXPERT_REGISTRY,
    LocalExpert,
    RemoteExpert,
)
from praxis.sorting import SORTING_REGISTRY

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseDecoder(nn.Module):
    """
    A module that wraps decoder operations.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.checkpoint_every = config.checkpoint_every
        self.num_experts = config.num_experts
        self.controller = CONTROLLER_REGISTRY.get(config.controller_type)(config)
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.compressor = COMPRESSION_REGISTRY.get(config.compression_type)(config)
        self.manager = False
        self.order = SORTING_REGISTRY.get(config.sorting_type)(config)
        self.locals = nn.ModuleList()
        self.remotes: List[nn.Module] = []
        
        # Call integration hooks for decoder initialization
        # This allows integrations like Hivemind to inject their management systems
        self._call_integration_hooks(config)
        if "scatter" in config.meta or config.expert in ["scatter"]:
            block = BLOCK_REGISTRY[config.block_type](config)
            expert = LocalExpert(config, block=block)
            for i in range(self.num_experts):
                self.locals.append(expert)
        elif config.router_type == "smear" and hasattr(config, "num_smear") and config.num_smear > 1:
            # For SMEAR with num_smear > 1, create a single SMEAR router managing multiple experts
            from praxis.routers import ROUTER_REGISTRY
            
            # Create the expert blocks
            expert_blocks = []
            for i in range(config.num_smear):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert_blocks.append(block)
            
            # Create a single SMEAR router with all blocks
            smear_router = ROUTER_REGISTRY["smear"](config, experts=expert_blocks)
            
            # Create a single LocalExpert with the SMEAR router
            # The SMEAR router will handle merging and routing internally
            expert = LocalExpert(config, block=expert_blocks[0], router=smear_router)
            
            # Add it multiple times to match num_experts for compatibility
            for i in range(self.num_experts):
                self.locals.append(expert)
        elif config.router_type == "smear":
            # Original SMEAR implementation for backward compatibility
            # For SMEAR, we need to create all blocks first, then pass them to each expert
            expert_blocks = []
            for i in range(self.num_experts):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert_blocks.append(block)

            # Now create LocalExperts with access to all blocks
            for i in range(self.num_experts):
                expert = LocalExpert(
                    config, block=expert_blocks[i], expert_blocks=expert_blocks
                )
                self.locals.append(expert)
        else:
            for i in range(self.num_experts):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert = LocalExpert(config, block=block)
                self.locals.append(expert)
        self.norm = (
            nn.LayerNorm(config.hidden_size, bias=True)
            if config.block_type == "mru"
            else False
        )
    
    def _call_integration_hooks(self, config: ConfigType) -> None:
        """Call integration hooks for decoder initialization.
        
        This allows integrations to modify the decoder during initialization.
        For example, the Hivemind integration uses this to inject its management system.
        
        Args:
            config: Model configuration
        """
        try:
            # Try to get the integration loader if available
            # We import it this way to avoid circular imports and other issues
            import sys
            if 'cli' in sys.modules and hasattr(sys.modules['cli'], 'integration_loader'):
                integration_loader = sys.modules['cli'].integration_loader
                
                # Call on_decoder_init for all active integrations
                for integration in integration_loader.loaded_integrations.values():
                    if hasattr(integration, 'on_decoder_init'):
                        integration.on_decoder_init(self, config)
        except (ImportError, AttributeError, RuntimeError):
            # Integration loader not available or other issues, skip hooks
            pass
        
        # If manager was injected by integration, start serving experts
        if self.manager and hasattr(self.manager, 'serve_experts'):
            self.manager.serve_experts()

    def post_layer(self, states: Tensor, current_depth: int) -> Tensor:
        """
        Process states after going through a layer.

        Args:
            states: Hidden states from the layer
            current_depth: Current layer depth

        Returns:
            Processed states
        """
        processed_states = states
        if self.genome and current_depth == 4:
            processed_states = self.genome(processed_states)
        return processed_states

    def post_decoding(self, states: Tensor) -> Tensor:
        """
        Process states after going through all layers.

        Args:
            states: Final hidden states

        Returns:
            Normalized or unchanged states
        """
        if self.norm:
            return self.norm(states)
        else:
            return states

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current prediction accuracies and other metrics.

        Returns:
            Dictionary of metrics
        """
        extras: Dict[str, Any] = {}
        if self.genome:
            extras = {**extras, **self.genome.get_metrics()}
        return {
            "experts": dict(
                local=len(self.locals),
                remote=len(self.remotes),
            ),
            **extras,
        }
