"""
Prismatic Attention: Architectural Diversity Through Parameter Merging

Receives pre-created experts with different architectures (RoPE vs ALiBi).
Soft-merges learnable parameters, hard-gates architectural operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Prismatic(nn.Module):
    """
    Prismatic router with architectural diversity.

    Forward pass:
    1. Compute routing probabilities
    2. Soft-merge all learnable parameters across experts
    3. Hard-gate architecture choice based on routing
    4. Execute forward pass with merged params + selected architecture
    """

    __version__ = "3.0.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize Prismatic with architecturally diverse experts.

        Args:
            config: Configuration with num_experts, hidden_size, architectures
            layout: Unused, kept for compatibility
            **kwargs: Either 'expert_class' or 'experts'
        """
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

        # Get experts - they should be pre-created with different architectures
        experts = kwargs.get("experts")
        if experts is None:
            raise ValueError(
                "Prismatic requires 'experts' parameter with pre-created expert blocks. "
                "Experts should be created in base.py with different pos_types via config."
            )

        if len(experts) != self.num_experts:
            raise ValueError(
                f"Number of provided experts ({len(experts)}) doesn't match "
                f"config.num_experts ({self.num_experts})"
            )

        self.experts = nn.ModuleList(experts)

        # Debug: Print received expert architectures
        print(f"[PRISMATIC ROUTER] Received {len(self.experts)} experts:")
        for i, expert in enumerate(self.experts):
            # Try to get pos_type from expert's attention layer
            pos_type = self._get_expert_pos_type(expert)
            print(f"  Expert {i}: pos_type={pos_type}")

        # Router: learns which architecture for which input
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        # Metrics
        self._metrics: Dict[str, float] = {}

        # Architecture selection counters
        self._arch_selection_counts = [0 for _ in range(self.num_experts)]
        self._total_selections = 0

    def forward(
        self, *args, **kwargs
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], float],
        Tuple[
            torch.Tensor,
            Optional[Union[torch.Tensor, List, Dict]],
            Optional[torch.Tensor],
            float,
        ],
    ]:
        """Forward pass with architecture gating."""
        if self._is_router_mode(args, kwargs):
            return self._router_forward(*self._parse_router_args(args, kwargs))
        else:
            return self._direct_forward(*self._parse_direct_args(args, kwargs))

    def _router_forward(
        self,
        layer: nn.Module,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Union[torch.Tensor, List, Dict]],
        current_state: Optional[torch.Tensor],
        current_depth: int,
        block_ids: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[Union[torch.Tensor, List, Dict]],
        Optional[torch.Tensor],
        float,
    ]:
        """Router mode forward pass."""
        # Compute routing probabilities
        routing_probs = self._compute_routing(inputs)

        # Hard-gate architecture: pick based on routing
        arch_idx = routing_probs.mean(0).argmax().item()
        selected_expert = self.experts[arch_idx]

        # Track architecture selection BEFORE merging (so metrics reflect current state)
        self._arch_selection_counts[arch_idx] += 1
        self._total_selections += 1

        # Soft-merge parameters (this logs metrics with updated counts)
        merged_params = self._soft_merge_parameters(routing_probs)

        # Forward with merged params
        result = torch.func.functional_call(
            selected_expert,
            merged_params,
            (
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            ),
            {},
        )

        # Normalize return format
        if isinstance(result, tuple) and len(result) == 4:
            return result
        elif isinstance(result, tuple) and len(result) == 3:
            return result[0], result[1], result[2], 0.0
        else:
            return result, past_key_values, current_state, 0.0

    def _direct_forward(
        self,
        inputs: torch.Tensor,
        current_state: Optional[torch.Tensor],
        current_depth: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Direct mode forward pass."""
        routing_probs = self._compute_routing(inputs)

        # Pick architecture
        arch_idx = routing_probs.mean(0).argmax().item()
        selected_expert = self.experts[arch_idx]

        # Track architecture selection BEFORE merging
        self._arch_selection_counts[arch_idx] += 1
        self._total_selections += 1

        # Soft-merge parameters (this logs metrics with updated counts)
        merged_params = self._soft_merge_parameters(routing_probs)

        result = torch.func.functional_call(
            selected_expert, merged_params, (inputs, current_state), {}
        )

        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1], 0.0
        else:
            return result, current_state, 0.0

    def _compute_routing(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities."""
        router_input = inputs.mean(dim=1)
        router_input = self.router_norm(router_input)

        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)
        routing_probs = F.softmax(logits, dim=-1)

        return routing_probs

    def _soft_merge_parameters(
        self, routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Soft-merge parameters weighted by routing probabilities."""
        expert_weights = routing_probs.mean(dim=0)
        self._log_routing_metrics(expert_weights, routing_probs)

        merged_params = {}
        param_names = [name for name, _ in self.experts[0].named_parameters()]

        for param_name in param_names:
            merged_param = None

            for expert_idx in range(self.num_experts):
                param = dict(self.experts[expert_idx].named_parameters())[param_name]
                weighted = param * expert_weights[expert_idx].to(param.device)
                merged_param = (
                    weighted if merged_param is None else merged_param + weighted
                )

            merged_params[param_name] = merged_param

        return merged_params

    def _log_routing_metrics(
        self, expert_weights: torch.Tensor, routing_probs: torch.Tensor
    ):
        """Log routing metrics including architecture selection counts."""
        try:
            # Routing probabilities
            for i in range(self.num_experts):
                self._metrics[f"routing/expert_{i}_weight"] = expert_weights[i].item()

            probs = expert_weights + 1e-10
            entropy = -(probs * probs.log()).sum()
            self._metrics["routing/entropy"] = entropy.item()
            self._metrics["routing/concentration"] = expert_weights.max().item()
            self._metrics["routing/variance"] = expert_weights.var().item()

            ideal_weight = 1.0 / self.num_experts
            balance = 1.0 - ((expert_weights - ideal_weight).abs().sum().item() / 2.0)
            self._metrics["routing/balance"] = balance

            # Architecture selection counts and percentages
            for i in range(self.num_experts):
                self._metrics[f"arch/expert_{i}_count"] = float(self._arch_selection_counts[i])

                # Percentage of total selections
                if self._total_selections > 0:
                    percentage = (self._arch_selection_counts[i] / self._total_selections) * 100.0
                    self._metrics[f"arch/expert_{i}_pct"] = percentage

            self._metrics["arch/total_selections"] = float(self._total_selections)

        except Exception:
            pass

    def get_metrics(self) -> Dict[str, float]:
        """Return routing metrics."""
        return self._metrics.copy()

    def log_gradient_dynamics(self) -> Optional[Dict[str, float]]:
        """Log gradient statistics for all experts.

        Returns flat dict with per-expert gradient norms and variance.
        Compatible with any router type, any number of experts.

        Returns:
            {
                "expert_0_grad_norm": 0.12,
                "expert_0_grad_var": 0.003,
                "expert_1_grad_norm": 0.08,
                "expert_1_grad_var": 0.002,
                ...
            }
        """
        if not hasattr(self, "experts") or len(self.experts) == 0:
            return None

        metrics = {}

        for expert_idx, expert in enumerate(self.experts):
            grad_norms = []
            grad_vars = []

            for param in expert.parameters():
                if param.grad is None:
                    continue

                # Compute gradient norm for this parameter
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                # Compute gradient variance for this parameter
                grad_var = param.grad.var().item()
                grad_vars.append(grad_var)

            # Aggregate across all parameters for this expert
            if grad_norms:
                # L2 norm across all parameter gradient norms
                metrics[f"expert_{expert_idx}_grad_norm"] = sum(
                    g**2 for g in grad_norms
                ) ** 0.5
            if grad_vars:
                # Mean variance
                metrics[f"expert_{expert_idx}_grad_var"] = sum(grad_vars) / len(
                    grad_vars
                )

        return metrics if metrics else None

    def _get_expert_pos_type(self, expert: nn.Module) -> str:
        """Extract pos_type from expert block (looks inside TransformerBlock for attention)."""
        # If expert is HexAttention directly
        if hasattr(expert, "pos_type"):
            return expert.pos_type

        # If expert is TransformerBlock, look inside for attention
        if hasattr(expert, "attn") and hasattr(expert.attn, "pos_type"):
            return expert.attn.pos_type

        # Recursively search for pos_type in any submodule
        for module in expert.modules():
            if hasattr(module, "pos_type"):
                return module.pos_type

        return "unknown"

    def _is_router_mode(self, args: tuple, kwargs: dict) -> bool:
        return len(args) == 7 or "layer" in kwargs

    def _parse_router_args(self, args: tuple, kwargs: dict) -> tuple:
        if len(args) == 7:
            return args
        return (
            kwargs["layer"],
            kwargs["inputs"],
            kwargs.get("attention_mask"),
            kwargs.get("past_key_values"),
            kwargs.get("current_state"),
            kwargs.get("current_depth", 0),
            kwargs.get("block_ids"),
        )

    def _parse_direct_args(
        self, args: tuple, kwargs: dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        if len(args) >= 3:
            return args[0], args[1], args[2]
        elif len(args) == 2:
            return args[0], args[1], 0
        elif len(args) == 1:
            return args[0], None, 0
        else:
            inputs = kwargs.get("inputs")
            if inputs is None:
                raise ValueError("No inputs provided")
            return inputs, kwargs.get("current_state"), kwargs.get("current_depth", 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_experts={self.num_experts}, "
            f"version={self.__version__})"
        )
