import copy
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
from praxis.halting import HALTING_REGISTRY
from praxis.layers import LocalLayer, RemoteLayer
from praxis.orchestration import EXPERT_REGISTRY
from praxis.sorting import SORTING_REGISTRY
from praxis.width import WIDTH_REGISTRY

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseDecoder(nn.Module):
    """
    A module that wraps decoder operations.
    """

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.checkpoint_every = config.checkpoint_every
        self.num_experts = config.num_experts
        # Use num_layers for the actual number of layer components in the model
        self.num_layers = getattr(config, "num_layers", config.num_experts)
        # Mixture-of-widths: the policy that deflates each step's inner rank.
        # Registered first so it sits atop the decoder on the blueprint, where the
        # loop reaches for it (see SequentialDecoder).
        self.width = WIDTH_REGISTRY[getattr(config, "width_type", None) or "none"]()
        self._width_realized = None  # mean active width used in the last forward
        self.controller = CONTROLLER_REGISTRY.get(config.controller_type)(config)
        self.genome = GenomicBottleneck(config) if config.evolve else False
        self.compressor = COMPRESSION_REGISTRY.get(config.compression_type)(config)
        self.manager = False
        self.order = SORTING_REGISTRY.get(config.sorting_type)(config)
        halting_type = getattr(config, "halting_type", None) or "none"
        self.halting = HALTING_REGISTRY[halting_type](config)
        # Mono-forward graph cutting (praxis/decoders/mono.py): a no-op unless
        # config.mono_type names a cut schedule. Sequential-only; build_mono
        # raises on other decoder types rather than silently never cutting.
        from praxis.decoders.mono import build_mono

        self.mono = build_mono(config)
        self.locals = nn.ModuleList()
        self.remotes: List[nn.Module] = []

        # Remote-expert pool (orchestration). A registered submodule so the swarm
        # has a visible home in the model blueprint, alongside ``locals``. Passive
        # observer for now - identity in forward, NOT added to the routing experts
        # (see ExpertPoolLayer). Only present when --orchestration-type is set.
        from praxis.orchestration import get_orchestration_profile

        _orch = get_orchestration_profile(getattr(config, "orchestration_type", "none"))
        if _orch:
            from praxis.orchestration.layer import ExpertPoolLayer

            self.swarm = ExpertPoolLayer(
                profile_name=config.orchestration_type,
                mixing=_orch.get("mixing", "vote"),
                sidecar=_orch.get("sidecar", True),
                init_experts=int(_orch.get("init_experts", 4)),
            )

        # Call integration hooks for decoder initialization
        # This allows integrations like Hivemind to inject their management systems
        self._call_integration_hooks(config)
        if "scatter" in config.meta or config.expert in ["scatter"]:
            block = BLOCK_REGISTRY[config.block_type](config)
            expert = LocalLayer(config, block=block)
            for i in range(self.num_layers):
                self.locals.append(expert)
        elif config.router_type in ("smear", "vear", "distance"):
            # For SMEAR/VEAR and Distance routers with multiple experts, create a single LocalLayer
            # that manages all experts and reuse it across all positions
            expert_blocks = []
            for expert_idx in range(self.num_experts):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert_blocks.append(block)

            # Create a single LocalLayer with all expert blocks
            expert = LocalLayer(
                config, block=expert_blocks[0], expert_blocks=expert_blocks
            )
            # Reuse the same expert for all layer positions
            for i in range(self.num_layers):
                self.locals.append(expert)
        elif config.router_type == "prismatic":
            # For Prismatic with architectural diversity, create experts with different encoding
            # Philosophy: Test "Blind Watchmaker" hypothesis - architectural diversity
            # reveals patterns single approaches cannot discover
            expert_blocks = []
            encodings = ["alibi", "rope"]  # Expert 0: ALiBi, Expert 1: RoPE

            original_encoding = getattr(config, "encoding", None)

            for expert_idx in range(self.num_experts):
                # Set positional encoding for this expert
                config.encoding = encodings[expert_idx % len(encodings)]

                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert_blocks.append(block)

                print(
                    f"[PRISMATIC v7.0] Created expert {expert_idx} with encoding={config.encoding}"
                )

            print(f"[PRISMATIC v7.0] Architectural diversity: ALiBi vs RoPE")
            print(f"  Expert 0: ALiBi (linear distance bias)")
            print(f"  Expert 1: RoPE (rotational encoding)")

            # Restore original encoding
            if original_encoding is not None:
                config.encoding = original_encoding
            elif hasattr(config, "encoding"):
                delattr(config, "encoding")

            # Create a single LocalLayer with all expert blocks
            expert = LocalLayer(
                config, block=expert_blocks[0], expert_blocks=expert_blocks
            )
            # Reuse the same expert for all layer positions
            for i in range(self.num_layers):
                self.locals.append(expert)
        else:
            for i in range(self.num_layers):
                if self.manager:
                    block = self.manager.register_expert(config)
                else:
                    block = BLOCK_REGISTRY[config.block_type](config)
                expert = LocalLayer(config, block=block)
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

            if "cli" in sys.modules and hasattr(
                sys.modules["cli"], "integration_loader"
            ):
                integration_loader = sys.modules["cli"].integration_loader

                # Call on_decoder_init for all active integrations
                for integration in integration_loader.loaded_integrations.values():
                    if hasattr(integration, "on_decoder_init"):
                        integration.on_decoder_init(self, config)
        except (ImportError, AttributeError, RuntimeError):
            # Integration loader not available or other issues, skip hooks
            pass

        # If manager was injected by integration, start serving experts
        if self.manager and hasattr(self.manager, "serve_experts"):
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

    def router_aux_losses(self) -> Dict[str, Any]:
        """Named auxiliary losses from unique routers, collected once per step
        OUTSIDE the (gradient-checkpointed, recurrent) forward - so a router's
        parameter-only loss (e.g. VEAR's repulsion) is added exactly once and
        doesn't escape a checkpointed region. Dedups by router id like
        ``get_metrics`` (the smear/vear branch shares one router across all
        positions)."""
        out: Dict[str, Any] = {}
        seen: set = set()
        for local_layer in self.locals:
            router = getattr(local_layer, "router", None)
            # router_aux_loss (not aux_loss) is VEAR-specific; using a unique name
            # avoids colliding with MixtureOfDepths.aux_loss(x, y) on arc routers.
            fn = getattr(router, "router_aux_loss", None)
            if router is None or not callable(fn) or id(router) in seen:
                continue
            seen.add(id(router))
            for name, value in fn().items():
                out[name] = out[name] + value if name in out else value
        return out

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current prediction accuracies and other metrics.

        Collects metrics from routers (SMEAR, Prismatic) for convergence tracking.

        Returns:
            Dictionary of metrics
        """
        extras: Dict[str, Any] = {}
        if self.genome:
            extras = {**extras, **self.genome.get_metrics()}

        halting = getattr(self, "halting", None)
        if halting is not None and hasattr(halting, "get_metrics"):
            extras.update(halting.get_metrics())

        # Mixture-of-widths: the per-depth active-width arch (inflate early,
        # decay through the tail). One key per depth so the dashboard can plot
        # the profile and watch it move once the schedule is learned.
        profile = self.width.profile(self.depth)
        if profile is not None:
            for d, frac in enumerate(profile):
                extras[f"width/active_d{d}"] = frac
            if self._width_realized is not None:
                extras["width/realized_mean"] = self._width_realized

        # Collect metrics from routers (expert convergence tracking)
        # Routers accumulate per-layer metrics internally using current_depth
        # We only need to collect from unique router instances to avoid duplicates
        if self.locals and len(self.locals) > 0:
            seen_routers: set = set()
            for local_layer in self.locals:
                if hasattr(local_layer, "router") and hasattr(
                    local_layer.router, "get_metrics"
                ):
                    router_id = id(local_layer.router)
                    if router_id in seen_routers:
                        continue
                    seen_routers.add(router_id)

                    router_metrics = local_layer.router.get_metrics()
                    if router_metrics:
                        # Metrics already have layer prefixes from router
                        extras.update(router_metrics)

        # Depth-trajectory (spectral-attractor) metrics, computed in the
        # SequentialDecoder forward; merged here so they flow with the rest.
        extras.update(getattr(self, "_depth_metrics", {}))

        # Mono-forward goodness scores (per cut + mean), same transport.
        extras.update(self.mono.metrics())

        # The remote count includes the live expert pool (orchestration), so the
        # dashboards' remote_layers reflects the swarm growing as peers join.
        remote_count = len(self.remotes)
        if getattr(self, "swarm", None) is not None:
            from praxis.orchestration import status as _pool_status

            remote_count += int(_pool_status.snapshot().get("experts_alive", 0))

        return {
            "layers": dict(
                local=len(self.locals),
                remote=remote_count,
            ),
            **extras,
        }
