"""
Prismatic Attention: Architectural Diversity Through Gradient-Space Constraints

This module implements prismatic attention by maintaining multiple independent experts
that explore different optimization trajectories through gradient-space modifications.

Theoretical Foundation (Quantum No-Cloning Constraint):
-------------------------------------------------------
The quantum no-cloning theorem states you cannot create an identical copy of an
arbitrary quantum state. Applied to neural networks: you cannot clone a model and
expect the copies to meaningfully diverge through static perturbations alone.

Previous approaches tried to create diversity by perturbing weights at runtime:
    Expert_i = BaseExpert + Perturbation_i

This fails because:
1. The router learns "clean weights perform best" → collapse to Expert 0
2. Perturbations exist outside gradient graph → no learning signal
3. Violates the spirit of no-cloning: trying to force diversity from identical copies

New Approach - Gradient-Space Experts:
--------------------------------------
Instead of perturbing weights, we perturb GRADIENTS. Each expert is an independent
model that explores a different optimization trajectory:

    Forward:  W_merged = Σ routing[i] × W_i (soft-merge for expressivity)
    Backward: Expert 0 → ∇L (pure gradient descent)
              Expert 1 → ∇L + suppress_top(∇L) (favor weak connections)
              Expert 2 → ∇L + amplify_bottom(∇L) (aggressive awakening)
              Expert N → ∇L + strategy_N(∇L) (diverse trajectories)

Each expert maintains its own parameters and trains along a different gradient
trajectory. The router learns which optimization strategies to combine, not which
noise levels to avoid.

Connection to "The Blind Watchmaker" Paper:
-------------------------------------------
The paper's computational substrate hypothesis: "Different architectural constraints
force different gradient trajectories through floating-point approximation space."

Gradient modifications ARE architectural constraints. They force each expert to
traverse the loss landscape differently, discovering patterns in different regions
of the approximation space without falling into the no-cloning trap.

Usage:
------
    # In training loop:
    optimizer.zero_grad()
    output = model(batch)
    loss.backward()

    # Modify gradients for experts 1+ (maintains diversity)
    for module in model.modules():
        if isinstance(module, Prismatic):
            module.modify_expert_gradients()

    optimizer.step()
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrismaticConfig:
    """Configuration for Prismatic attention.

    Attributes:
        hidden_size: Hidden dimension size
        num_experts: Number of independent experts (minimum 2)
        gradient_scale: Scale factor for gradient modifications (default: 0.3)
        sparsity: Fraction of weights to apply gradient modifications to (default: 0.1)
        dropout: Dropout probability for expert dropout during training (default: 0.0)
    """

    hidden_size: int
    num_experts: int
    gradient_scale: float = 0.3
    sparsity: float = 0.1
    dropout: float = 0.0


class Prismatic(nn.Module):
    """
    Prismatic Attention: Multiple independent experts with gradient-space constraints.

    Creates N independent experts that train along different optimization trajectories.
    Expert 0 remains pure (standard gradient descent). Experts 1+ have gradient
    modifications applied after backward pass, forcing diverse learning dynamics.

    Forward pass: Soft-merge expert parameters weighted by routing probabilities
    Backward pass: Apply expert-specific gradient modifications (except Expert 0)
    """

    __version__ = "2.0.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """Initialize Prismatic with N independent experts.

        Args:
            config: Configuration with num_experts, hidden_size, etc.
            layout: Layout type (unused, kept for interface compatibility)
            **kwargs: Must contain 'base_expert' or 'experts' parameter
        """
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.gradient_scale = getattr(config, "gradient_scale", 0.3)
        self.sparsity = getattr(config, "sparsity", 0.1)
        self.dropout_rate = getattr(config, "dropout", 0.0)

        if self.num_experts < 2:
            raise ValueError(
                "Prismatic requires at least 2 experts (1 pure + 1+ modified)"
            )

        # Get base expert for cloning
        base_expert = kwargs.get("base_expert") or (
            kwargs.get("experts", [None])[0] if kwargs.get("experts") else None
        )
        if base_expert is None:
            raise ValueError("Prismatic requires 'base_expert' or 'experts' parameter")

        # Create N independent experts (deep copies, not references)
        self.experts = nn.ModuleList(
            [copy.deepcopy(base_expert) for _ in range(self.num_experts)]
        )

        # Router: learns which optimization strategies to combine
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        # Metrics storage
        self._metrics: Dict[str, float] = {}

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], float],
        Tuple[
            torch.Tensor,
            Optional[Union[torch.Tensor, List, Dict]],
            Optional[torch.Tensor],
            float,
        ],
    ]:
        """Forward pass with soft-merged expert parameters.

        Supports two modes:
        1. Direct: (inputs, state, depth) → (output, state, aux_loss)
        2. Router: (layer, inputs, mask, kv, state, depth, ids) → (output, kv, state, aux_loss)
        """
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
        """Router mode: process inputs through soft-merged experts."""
        routing_probs = self._compute_routing(inputs)
        merged_params = self._soft_merge_experts(routing_probs)

        # Use first expert as structure, apply merged parameters
        result = torch.func.functional_call(
            self.experts[0],
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
        """Direct mode: simpler interface for RecurrentBlock usage."""
        routing_probs = self._compute_routing(inputs)
        merged_params = self._soft_merge_experts(routing_probs)

        result = torch.func.functional_call(
            self.experts[0], merged_params, (inputs, current_state), {}
        )

        # Normalize return format
        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1], 0.0
        else:
            return result, current_state, 0.0

    def _compute_routing(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities with dropout and normalization."""
        router_input = inputs.mean(dim=1)  # [batch, hidden]
        router_input = self.router_norm(router_input)

        # Normalized weights for stability
        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)
        routing_probs = F.softmax(logits, dim=-1)  # [batch, num_experts]

        # Expert dropout during training
        if self.training and self.dropout_rate > 0:
            mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * mask
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        return routing_probs

    def _soft_merge_experts(
        self, routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Soft-merge expert parameters weighted by routing probabilities.

        Returns merged parameters: W_merged = Σ routing[i] × W_i
        This provides ensemble expressivity from multiple independently-trained experts.
        """
        expert_weights = routing_probs.mean(dim=0)  # [num_experts]
        self._log_routing_metrics(expert_weights, routing_probs)

        merged_params = {}

        # Get all parameter names from first expert
        param_names = [name for name, _ in self.experts[0].named_parameters()]

        for param_name in param_names:
            merged_param = None

            for expert_idx in range(self.num_experts):
                # Get parameter from this expert
                param = dict(self.experts[expert_idx].named_parameters())[param_name]

                # Weight by routing probability
                weighted = param * expert_weights[expert_idx].to(param.device)

                merged_param = (
                    weighted if merged_param is None else merged_param + weighted
                )

            merged_params[param_name] = merged_param

        return merged_params

    def modify_expert_gradients(self):
        """
        Apply gradient-space constraints to experts 1+.

        IMPORTANT: Call this AFTER loss.backward() but BEFORE optimizer.step()

        Expert 0: Pure gradient descent (no modification)
        Expert 1: Suppress gradients to top-magnitude weights
        Expert 2: Amplify gradients to bottom-magnitude weights
        Expert 3+: Helical-modulated exploration

        This is where architectural diversity emerges - not from weight perturbations,
        but from different optimization trajectories through the loss landscape.
        """
        for expert_idx in range(1, self.num_experts):
            expert = self.experts[expert_idx]

            for param_name, param in expert.named_parameters():
                if param.grad is None:
                    continue

                # Apply expert-specific gradient strategy
                param.grad = self._apply_gradient_strategy(
                    param.grad, param, expert_idx, param_name
                )

    def _apply_gradient_strategy(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        expert_idx: int,
        param_name: str,
    ) -> torch.Tensor:
        """Apply gradient modification using continuous phase modulation.

        Each expert receives a phase angle θ = 2π × (expert_idx / num_experts).
        The phase determines how gradients are modified across weight tiers:

        - cos(θ) > 0: Suppress top weights, amplify bottom weights
        - cos(θ) < 0: Amplify top weights, suppress bottom weights
        - cos(θ) ≈ 0: Balanced modification

        This ensures continuous distribution of gradient strategies:
        - 2 experts: complementary phases (π apart)
        - 3 experts: evenly distributed (2π/3 apart)
        - N experts: evenly distributed around unit circle

        All experts explore different optimization trajectories without hard-coded branches.
        """
        # Create sparse mask for top/bottom weights
        mask = self._create_sparse_mask(param)  # +1=top, -1=bottom, 0=middle

        # Phase angle for this expert [0, 2π)
        theta = 2 * math.pi * expert_idx / self.num_experts

        # Split mask into top/bottom components
        top_mask = (mask > 0).float()
        bottom_mask = (mask < 0).float()

        # Apply phase-modulated modifications
        # Top weights: modified by cos(θ)
        # Bottom weights: modified by -cos(θ) (opposite phase)
        modification = (
            self.gradient_scale
            * grad
            * (
                math.cos(theta) * top_mask  # Suppress when cos>0, amplify when cos<0
                + -math.cos(theta) * bottom_mask  # Opposite for bottom weights
            )
        )

        return grad + modification

    def _create_sparse_mask(self, param: torch.Tensor) -> torch.Tensor:
        """Create tiered mask for gradient modifications.

        Returns:
            +1.0 for top sparsity/2 weights (highest magnitude)
            -1.0 for bottom sparsity/2 weights (lowest magnitude, non-zero)
             0.0 for middle tier (unchanged)
        """
        num_params = param.numel()
        num_per_side = max(1, int(num_params * self.sparsity / 2))

        flat_param = param.flatten().abs()

        # Top tier: highest magnitude weights
        top_threshold = torch.topk(flat_param, num_per_side).values[-1]
        top_mask = (flat_param >= top_threshold).float()

        # Bottom tier: lowest magnitude non-zero weights
        non_zero = flat_param[flat_param > 0]
        if len(non_zero) >= num_per_side:
            bottom_threshold = torch.topk(non_zero, num_per_side, largest=False).values[
                -1
            ]
            bottom_mask = ((flat_param <= bottom_threshold) & (flat_param > 0)).float()
        else:
            bottom_mask = (flat_param > 0).float()

        # Signed mask: +1 top, -1 bottom, 0 middle
        mask = (top_mask - bottom_mask).reshape(param.shape)
        return mask

    def _log_routing_metrics(
        self, expert_weights: torch.Tensor, routing_probs: torch.Tensor
    ):
        """Store routing metrics with stable flat schema."""
        try:
            # Per-expert routing weights
            for i in range(self.num_experts):
                self._metrics[f"routing/expert_{i}_weight"] = expert_weights[i].item()

            # Entropy: measures routing balance
            probs = expert_weights + 1e-10
            entropy = -(probs * probs.log()).sum()
            self._metrics["routing/entropy"] = entropy.item()

            # Concentration: max routing weight
            self._metrics["routing/concentration"] = expert_weights.max().item()

            # Variance: routing stability across experts
            self._metrics["routing/variance"] = expert_weights.var().item()

            # Balance: How evenly distributed (1.0 = perfectly balanced)
            ideal_weight = 1.0 / self.num_experts
            balance = 1.0 - ((expert_weights - ideal_weight).abs().sum().item() / 2.0)
            self._metrics["routing/balance"] = balance

        except Exception:
            pass  # Don't break training on metric failures

    def get_metrics(self) -> Dict[str, float]:
        """Return metrics with stable flat schema.

        All keys follow pattern: category/metric_name
        All values are Python floats (not tensors)

        This ensures API compatibility and chart stability.
        """
        return self._metrics.copy()

    def log_gradient_dynamics(self) -> Optional[Dict[str, Any]]:
        """Log gradient statistics for all experts.

        Call after backward() but before optimizer.step().
        Returns flat dict compatible with Dynamics tab charts.
        """
        if not hasattr(self, "experts") or len(self.experts) == 0:
            return None

        metrics = {}

        # Expert 0: Log tier-specific gradients (top/bottom/middle)
        expert_0_tier_stats = self._compute_expert_tier_gradients(0)
        if expert_0_tier_stats:
            metrics.update(expert_0_tier_stats)

        # Experts 1+: Log router learning gradients
        for expert_idx in range(1, self.num_experts):
            router_stats = self._compute_router_gradient_for_expert(expert_idx)
            if router_stats:
                metrics.update(router_stats)

        if not metrics:
            return None

        return metrics

    def _compute_expert_tier_gradients(
        self, expert_idx: int
    ) -> Optional[Dict[str, float]]:
        """Compute tier-specific gradient statistics for Expert 0.

        Returns gradients for top/bottom/middle weight tiers.
        """
        expert = self.experts[expert_idx]
        tier_grads = {"top": [], "bottom": [], "middle": []}

        for param in expert.parameters():
            if param.grad is None:
                continue

            # Compute tier masks based on weight magnitudes
            num_params = param.numel()
            num_per_side = max(1, int(num_params * self.sparsity / 2))
            flat_param = param.flatten().abs()
            flat_grad = param.grad.flatten().abs()

            # Top tier: highest magnitude weights
            top_threshold = torch.topk(flat_param, num_per_side).values[-1]
            is_top = flat_param >= top_threshold

            # Bottom tier: lowest magnitude non-zero weights
            non_zero_mask = flat_param > 0
            if non_zero_mask.sum() >= num_per_side:
                non_zero_param = flat_param[non_zero_mask]
                bottom_threshold = torch.topk(
                    non_zero_param, num_per_side, largest=False
                ).values[-1]
                is_bottom = (flat_param <= bottom_threshold) & non_zero_mask
            else:
                is_bottom = non_zero_mask

            # Middle tier: everything else
            is_middle = ~(is_top | is_bottom)

            # Collect gradient norms for each tier
            if is_top.sum() > 0:
                tier_grads["top"].append(flat_grad[is_top].norm().item())
            if is_bottom.sum() > 0:
                tier_grads["bottom"].append(flat_grad[is_bottom].norm().item())
            if is_middle.sum() > 0:
                tier_grads["middle"].append(flat_grad[is_middle].norm().item())

        # Compute L2 norm across all gradients in each tier
        result = {}
        for tier in ["top", "bottom", "middle"]:
            if tier_grads[tier]:
                # L2 norm: sqrt(sum of squares)
                tier_norm = sum(g**2 for g in tier_grads[tier]) ** 0.5
                result[f"expert_{expert_idx}_{tier}_norm"] = tier_norm

        return result if result else None

    def _compute_router_gradient_for_expert(
        self, expert_idx: int
    ) -> Optional[Dict[str, float]]:
        """Compute router gradient for a specific expert view.

        Shows how the router learns to select this expert.
        """
        if not hasattr(self.router, "weight") or self.router.weight.grad is None:
            return None

        # Get gradient norm for this expert's routing weight
        expert_router_grad = self.router.weight.grad[expert_idx]
        norm = expert_router_grad.norm().item()

        return {f"expert_{expert_idx}_router_weight_expert_{expert_idx}_norm": norm}

    def _is_router_mode(self, args: tuple, kwargs: dict) -> bool:
        """Check if we're in router mode based on arguments."""
        return len(args) == 7 or "layer" in kwargs

    def _parse_router_args(self, args: tuple, kwargs: dict) -> tuple:
        """Parse arguments for router mode."""
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
        """Parse arguments for direct mode."""
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
            f"gradient_scale={self.gradient_scale}, "
            f"sparsity={self.sparsity})"
        )
