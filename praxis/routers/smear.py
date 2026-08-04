import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


@dataclass
class SMEARConfig:
    """Configuration class for SMEAR module"""

    hidden_size: int
    num_experts: int
    expert_dropout: float = 0.1


class SMEAR(nn.Module):
    """
    This module implements Soft-Merging of Experts with Adaptive Routing (SMEAR):
    https://arxiv.org/abs/2306.03745

    SMEAR dynamically merges expert parameters based on routing probabilities,
    rather than routing inputs to multiple experts. This enables more efficient
    parameter sharing and adaptation to input patterns.
    """

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize the SMEAR module.

        Args:
            config: Configuration object
            layout: Layout type (not used by SMEAR, kept for interface compatibility)
            *args: Additional arguments
            **kwargs: Additional keyword arguments including 'experts' list
        """
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.dropout_rate = getattr(
            config, "expert_dropout", 0.1
        )  # Probability of dropping entire experts (default 0.1 per SMEAR paper)

        # Get experts from kwargs - required for SMEAR
        self.experts = kwargs.get("experts", None)
        if self.experts is None:
            raise ValueError(
                "SMEAR router requires 'experts' to be provided during initialization"
            )
        self.experts = nn.ModuleList(self.experts)

        self.parameter_names: List[str] = []

        # Router network with layer normalization as per paper
        # Use actual number of experts, not config.num_experts
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, len(self.experts))

        # Metrics storage for convergence tracking
        self._metrics = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_experts={len(self.experts)})"

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], float],  # Direct mode
        Tuple[
            torch.Tensor,
            Optional[Union[torch.Tensor, List, Dict]],
            Optional[torch.Tensor],
            float,
        ],  # Router mode
    ]:
        """
        Forward pass with SMEAR routing.

        Supports two modes:
        1. Direct mode: Used by RecurrentBlock with (inputs, current_state)
        2. Router mode: Used as a router in LocalLayer with full signature

        Returns:
            Appropriate tuple based on the mode
        """
        # Determine which mode we're in based on arguments
        if self._is_router_mode(args, kwargs):
            # Extract router mode arguments
            router_args = self._parse_router_args(args, kwargs)
            return self._router_forward(*router_args)
        else:
            # Extract direct mode arguments
            inputs, current_state = self._parse_direct_args(args, kwargs)
            return self._direct_forward(inputs, current_state)

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
        # Get routing probabilities with proper normalization
        router_input = inputs.mean(dim=1)  # Average across sequence length
        router_input = self.router_norm(router_input)  # Layer norm on input

        # Get logits with normalized router weights
        # Apply weight normalization without modifying in-place
        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)

        routing_probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # The router's OWN opinion, captured before expert dropout and before any
        # subclass transform. This is what the routing diagnostics are computed
        # on; see _log_routing_metrics for why that distinction matters.
        router_probs = routing_probs

        # Apply expert dropout during training (drop entire experts)
        if self.training and self.dropout_rate > 0:
            # Create dropout mask for experts
            expert_mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * expert_mask
            # Renormalize to ensure probabilities sum to 1
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(
            routing_probs, current_depth, router_probs=router_probs
        )

        # Use the first expert as the base module structure
        base_module = self.experts[0]

        # Create args tuple for functional_call
        forward_args = (
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        )

        # Apply the merged parameters using functional_call. tie_weights=False is
        # REQUIRED. This is functional_call's OWN arg (not model weight-tying - the
        # model need not tie anything): with the default (True), its param-aliasing
        # machinery corrupts the merged-param graph when the SAME expert module is
        # reparametrized repeatedly across recurrent depths, so the next step's
        # backward re-enters a freed graph ("backward through the graph a second
        # time"). Verified on a model with zero tied weights. Only bites under the
        # smear/vear shared-expert recurrent reuse.
        result = torch.func.functional_call(
            base_module, merged_state_dict, forward_args, {}, tie_weights=False
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 4:
            return result
        elif isinstance(result, tuple) and len(result) == 3:
            # Add zero aux loss if not provided
            return result[0], result[1], result[2], 0.0
        else:
            # Fallback for unexpected return format
            return result, past_key_values, current_state, 0.0

    def _collect_parameter_names(
        self, module: nn.Module, prefix: str = ""
    ) -> List[str]:
        """
        Recursively collect all parameter names from a module.

        Args:
            module: Module to collect parameter names from
            prefix: Prefix for nested parameter names

        Returns:
            List of parameter names with full path
        """
        parameter_names = []
        for name, submodule in module.named_children():
            parameter_names.extend(
                self._collect_parameter_names(submodule, prefix + name + ".")
            )
        for name, _ in module.named_parameters(recurse=False):
            parameter_names.append(prefix + name)
        return parameter_names

    def _merge_expert_parameters(
        self,
        routing_probs: torch.Tensor,
        current_depth: int = 0,
        router_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Merge expert parameters based on routing probabilities.

        Args:
            routing_probs: Probabilities the merge actually uses, shape
                [batch_size, num_experts]. Subclasses may transform these
                (VEAR sharpens by ``p**4``), so they are NOT the router's output.
            current_depth: Current layer depth for per-layer metric tracking
            router_probs: The router's own pre-transform, pre-dropout output.
                Defaults to ``routing_probs`` (correct for plain SMEAR, where the
                merge uses exactly what the router emitted). Subclasses that
                transform the probabilities MUST forward the original here or the
                diagnostics describe the transform instead of the router.

        Returns:
            Dictionary of merged parameters

        Raises:
            ValueError: If a parameter is not found in an expert
        """
        # Initialize a dictionary to hold the merged parameters
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # Compute the mean routing probability across the batch for each expert
        expert_weights = routing_probs.mean(dim=0)  # [num_experts]

        # Log expert convergence metrics for visualization (with layer prefix)
        self._log_routing_metrics(
            expert_weights,
            routing_probs if router_probs is None else router_probs,
            current_depth,
            merge_weights=expert_weights,
        )

        # Iterate over all parameter names
        self.parameter_names = self._collect_parameter_names(self.experts[0])
        for param_name in self.parameter_names:
            # Initialize the merged parameter with zeros
            merged_param: Optional[torch.Tensor] = None

            for expert_idx, expert in enumerate(self.experts):
                # Get the parameter from the expert
                param = self._get_module_parameter(expert, param_name)

                if param is None:
                    raise ValueError(
                        f"Parameter '{param_name}' not found in expert {expert_idx}."
                    )

                # Ensure param is on the same device as expert_weights before multiplication
                if param.device != expert_weights.device:
                    param = param.to(expert_weights.device)

                # Weight the parameter by the expert's routing probability
                weighted_param = param * expert_weights[expert_idx]

                if merged_param is None:
                    merged_param = weighted_param
                else:
                    merged_param = merged_param + weighted_param

            assert merged_param is not None, "Merged parameter should not be None"
            merged_state_dict[param_name] = merged_param

        return merged_state_dict

    def _get_module_parameter(
        self, module: nn.Module, param_name: str
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a parameter from a module using a fully qualified parameter name.

        Args:
            module: Module to get parameter from
            param_name: Fully qualified parameter name (e.g., "layer1.weight")

        Returns:
            Parameter tensor if found, None otherwise
        """
        parts = param_name.split(".")
        submodule = module
        for part in parts[:-1]:
            if hasattr(submodule, part):
                submodule = getattr(submodule, part)
            else:
                return None
        return getattr(submodule, parts[-1], None)

    @staticmethod
    def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Shannon entropy in nats, numerically safe.

        ``clamp_min``, NOT ``+ eps``. Adding an epsilon pushes a weight of exactly
        1.0 above 1, which makes ``log`` positive and the resulting "entropy"
        slightly NEGATIVE. That artifact is how the old routing_entropy reported
        ``-0.00000`` under VEAR, and a negative entropy is also the tell that the
        distribution had saturated to one-hot in float.
        """
        p = probs.clamp_min(1e-12)
        # clamp_min(0) on the result too: a single-expert router gives
        # -(1.0 * log(1.0)) = -0.0, and a negative zero is noise on a log axis.
        return (-(p * p.log()).sum(dim=dim)).clamp_min(0.0)

    def _log_routing_metrics(
        self,
        expert_weights: torch.Tensor,
        routing_probs: torch.Tensor,
        current_depth: int = 0,
        merge_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store routing metrics for expert convergence tracking.

        WHAT IS MEASURED ON WHAT. Every diagnostic here describes the ROUTER, so
        it is computed on ``routing_probs`` = the router's own pre-transform,
        pre-dropout output. The single exception is
        ``expert_{i}_routing_weight`` / ``routing_merge_entropy``, which describe
        the MERGE and therefore use the post-transform weights.

        That split exists because VEAR sharpens by ``p**4`` before merging
        (praxis/routers/vear.py). Measuring the router's diagnostics on the
        sharpened probabilities made every one of them report VEAR's own
        exponent rather than anything the router learned: entropy saturated to
        float-exact one-hot and became insensitive to the weights entirely -
        bit-identical across models whose losses differed by 5%. A metric that
        cannot vary with the thing it claims to measure is worse than absent,
        because it reads as a finding.

        Metrics are stored with layer prefixes to support per-layer visualization
        when the same router is called at multiple layer positions.

        Metrics flow: SMEAR router → Decoder.get_metrics() → Model.get_metrics() →
                     BackpropagationTrainer.log_dict() → MetricsLoggerCallback →
                     SQLite → API → Web dashboard

        Args:
            expert_weights: Mean routing probability per expert [num_experts]
            routing_probs: The ROUTER's probabilities [batch_size, num_experts],
                before any subclass transform and before expert dropout
            current_depth: Current layer depth for per-layer metric tracking
            merge_weights: Batch-mean weights the merge actually used. Differs
                from the router's own mean whenever a subclass transforms them.
        """
        try:
            layer_prefix = f"layer_{current_depth}_"

            # Per-expert weights the MERGE used - individual expert convergence
            # trajectories, i.e. which experts are actually being applied.
            for i, weight in enumerate(expert_weights):
                self._metrics[f"{layer_prefix}expert_{i}_routing_weight"] = (
                    weight.item()
                )

            n = routing_probs.size(-1)
            router_mean = routing_probs.mean(dim=0)  # [num_experts]

            # Entropy of the router's batch-mean: aggregate expert LOAD BALANCE.
            # High = the batch spreads across experts; low = the whole batch is
            # landing on one expert.
            h_mean = self._entropy(router_mean)
            self._metrics[f"{layer_prefix}routing_entropy"] = h_mean.item()

            # Concentration: max batch-mean weight (1/N balanced .. 1.0 hogging).
            self._metrics[f"{layer_prefix}routing_concentration"] = (
                router_mean.max().item()
            )

            # Variance of the batch-mean, NORMALIZED to [0, 1]. Raw variance maxes
            # at (N-1)/N^2 (~0.19 for N=4), so the raw number reads misleadingly
            # small; here 0 = perfectly balanced load, 1 = collapsed onto one expert.
            max_var = (n - 1) / (n * n) if n > 1 else 1.0
            self._metrics[f"{layer_prefix}routing_variance"] = (
                router_mean.var(unbiased=False) / max_var
            ).item()

            # --- Specialization, computed PER SEQUENCE (before the batch-mean) ---
            # This is what VEAR's sharpening/repulsion actually move: how committed
            # each sequence's routing is. The batch-mean metrics above CANNOT show
            # it - different sequences picking different experts average back to
            # uniform, so routing_variance/entropy plateau low even at perfect
            # per-sequence specialization.
            # routing_peak: mean per-sequence top weight (1/N uniform .. 1.0 committed).
            peak = routing_probs.max(dim=-1).values.mean()
            self._metrics[f"{layer_prefix}routing_peak"] = peak.item()
            # Mean per-sequence entropy: how undecided a TYPICAL routing decision
            # is, as opposed to how balanced the aggregate load is.
            h_seq = self._entropy(routing_probs, dim=-1).mean()
            self._metrics[f"{layer_prefix}routing_entropy_seq"] = h_seq.item()

            if n > 1:
                # routing_specialization: peak rescaled to [0, 1] - 0 = uniform
                # routing, 1 = every sequence commits to a single expert.
                self._metrics[f"{layer_prefix}routing_specialization"] = (
                    (n * peak - 1.0) / (n - 1)
                ).item()

                # INPUT DEPENDENCE: I(input; expert) = H(mean p) - mean H(p),
                # normalized by log(N) to [0, 1]. This is the number that answers
                # "is the router reading its input at all". Zero means every
                # sequence in the batch got the same routing distribution, so the
                # router is a constant; one means each sequence commits to a
                # different expert. Neither load balance nor specialization can
                # distinguish those cases on its own - a router that sends the
                # WHOLE batch to one expert scores maximum specialization.
                mi = (h_mean - h_seq) / math.log(n)
                self._metrics[f"{layer_prefix}routing_input_dependence"] = (
                    mi.clamp(0.0, 1.0).item()
                )

            # Entropy of the weights the MERGE used. For plain SMEAR this equals
            # routing_entropy; under VEAR the gap between them IS the sharpening,
            # so reading the two together shows how much p**4 is doing.
            if merge_weights is not None:
                self._metrics[f"{layer_prefix}routing_merge_entropy"] = (
                    self._entropy(merge_weights).item()
                )
        except Exception:
            # Silently fail if metric computation fails - don't break training
            pass

    def get_metrics(self) -> dict:
        """Return collected metrics for logging."""
        return self._metrics.copy()

    def log_gradient_dynamics(self) -> Optional[Dict[str, float]]:
        """Log gradient statistics for all experts.

        Returns flat dict with per-expert gradient norms and variance.

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

                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                grad_var = param.grad.var().item()
                grad_vars.append(grad_var)

            # Aggregate across all parameters
            if grad_norms:
                metrics[f"expert_{expert_idx}_grad_norm"] = (
                    sum(g**2 for g in grad_norms) ** 0.5
                )
            if grad_vars:
                metrics[f"expert_{expert_idx}_grad_var"] = sum(grad_vars) / len(
                    grad_vars
                )

        return metrics if metrics else None

    def _is_router_mode(self, args: tuple, kwargs: dict) -> bool:
        """Check if we're in router mode based on arguments."""
        # Router mode if we have 7 positional args or 'layer' in kwargs
        return len(args) == 7 or "layer" in kwargs

    def _parse_router_args(self, args: tuple, kwargs: dict) -> tuple:
        """Parse arguments for router mode."""
        if len(args) == 7:
            # Positional arguments
            return args
        else:
            # Keyword arguments
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parse arguments for direct mode."""
        if len(args) >= 2:
            # Positional args: (inputs, current_state)
            return args[0], args[1]
        elif len(args) == 1:
            # Only inputs provided
            return args[0], None
        else:
            # Try kwargs
            inputs = kwargs.get("inputs")
            if inputs is None:
                raise ValueError(f"No inputs provided. Args: {args}, Kwargs: {kwargs}")
            return inputs, kwargs.get("current_state")

    def _direct_forward(
        self,
        inputs: torch.Tensor,
        current_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Direct mode forward pass for RecurrentBlock usage."""
        # Get routing probabilities with proper normalization
        router_input = inputs.mean(dim=1)  # Average across sequence length
        router_input = self.router_norm(router_input)  # Layer norm on input

        # Get logits with normalized router weights
        # Apply weight normalization without modifying in-place
        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)

        routing_probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # Router's own opinion, before dropout and before any subclass transform.
        router_probs = routing_probs

        # Apply expert dropout during training (drop entire experts)
        if self.training and self.dropout_rate > 0:
            # Create dropout mask for experts
            expert_mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * expert_mask
            # Renormalize to ensure probabilities sum to 1
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(
            routing_probs, router_probs=router_probs
        )

        # Use the first expert as the base module structure
        base_module = self.experts[0]

        # Apply the merged parameters using functional_call (tie_weights=False:
        # see _router_forward - avoids the shared-expert recurrent double-backward).
        result = torch.func.functional_call(
            base_module,
            merged_state_dict,
            (inputs, current_state),
            {},
            tie_weights=False,
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            # Add zero aux loss if not provided
            return result[0], result[1], 0.0
        else:
            # Fallback
            return result, current_state, 0.0
