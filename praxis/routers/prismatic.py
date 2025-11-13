"""
Prismatic Attention: Architectural Diversity via Sparse Routing

v7.0 - Positional Encoding Diversity:
Routes sequences to experts with different positional encoding strategies:
- Expert 0: ALiBi (Attention with Linear Biases)
- Expert 1: RoPE (Rotary Position Embedding)

Clean, simple test of the core hypothesis from "The Blind Watchmaker":
Different architectural constraints force different gradient trajectories through
the computational substrate, revealing patterns single approaches cannot discover.

Philosophy:
"Architectural Diversity" - Same input, same masking, different positional encodings.
Let the model learn which architectural constraint suits which pattern.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Prismatic(nn.Module):
    """
    Prismatic router with architectural diversity (ALiBi vs RoPE).

    Routes entire sequences to one of two experts with different positional encodings:
    - Expert 0: ALiBi (linear distance bias)
    - Expert 1: RoPE (rotational encoding)

    The router:
    1. Computes routing probabilities per sequence
    2. Selects ONE expert per sequence (sparse, top-1)
    3. Executes selected expert (both use standard causal masking)
    4. Applies load balancing loss to encourage balanced usage

    Key design: Clean architectural diversity. Different pos_type per expert,
    everything else identical. Tests core "Blind Watchmaker" hypothesis.
    """

    __version__ = "7.0.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize Prismatic with architectural diversity.

        Args:
            config: Configuration with num_experts, hidden_size
            layout: Unused, kept for compatibility
            **kwargs: Must include 'experts' parameter
        """
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

        # Get experts - should have different pos_type via modulus cycling
        experts = kwargs.get("experts")
        if experts is None:
            raise ValueError(
                "Prismatic requires 'experts' parameter with pre-created expert blocks."
            )

        if len(experts) != self.num_experts:
            raise ValueError(
                f"Number of provided experts ({len(experts)}) doesn't match "
                f"config.num_experts ({self.num_experts})"
            )

        self.experts = nn.ModuleList(experts)

        # Architecture list for modulus cycling
        self.architectures = ["alibi", "rope"]  # Cycles: alibi, rope, alibi, rope, ...

        print(f"[PRISMATIC v7.0] Architectural diversity with {len(self.experts)} experts")
        print(f"  Architectures: {', '.join(self.architectures)} (cycling via modulus)")
        for i in range(len(self.experts)):
            arch = self.architectures[i % len(self.architectures)]
            print(f"  Expert {i}: {arch.upper()}")
        print(f"  Routing: Sequence-level, top-1 sparse")
        print(f"  Masking: Standard causal (no temporal tricks)")

        # Router: learns to select ALiBi vs RoPE per sequence
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        # Load balancing
        self.balance_loss_coef = getattr(config, "router_balance_loss_coef", 0.01)

        # Metrics
        self._metrics: Dict[str, float] = {}

        # Cumulative expert selection counters (for actual k=1 usage tracking)
        self.register_buffer("expert_selection_counts", torch.zeros(self.num_experts, dtype=torch.long))

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
        """Forward pass with sparse routing."""
        if self._is_router_mode(args, kwargs):
            return self._router_forward(*args, **kwargs)
        else:
            return self._expert_mode_forward(*args, **kwargs)

    def _is_router_mode(self, args: tuple, kwargs: dict) -> bool:
        """Check if we're in router mode (multiple args) vs expert mode (single arg)."""
        if len(args) >= 6:
            return True
        return "current_depth" in kwargs

    def _expert_mode_forward(
        self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Expert mode: simple pass-through to first expert."""
        output, cache, aux_loss = self.experts[0](inputs, attention_mask)
        return output, cache, aux_loss

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
        """
        Router mode: select expert per sequence based on architectural suitability.

        Args:
            layer: The LocalLayer wrapper (unused in sparse routing)
            inputs: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask
            past_key_values: KV cache (if using)
            current_state: Current hidden state
            current_depth: Current layer depth
            block_ids: Block IDs for attention

        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_values: Updated cache
            current_state: Updated state (None for now)
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, _ = inputs.shape

        # Compute routing probabilities per sequence
        # Use mean pooling to get sequence-level representation
        seq_repr = inputs.mean(dim=1)  # [batch, hidden_size]
        seq_repr = self.router_norm(seq_repr)
        logits = self.router(seq_repr)  # [batch, 2]
        probs = F.softmax(logits, dim=-1)  # [batch, 2]

        # Top-1 expert selection (sparse)
        expert_indices = torch.argmax(probs, dim=-1)  # [batch]

        # Execute experts in parallel batches (efficient sparse MoE)
        # Group sequences by expert assignment and execute each expert once
        output = torch.zeros_like(inputs)
        total_aux_loss = 0.0

        # Execute each expert on its assigned sequences
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)  # [batch]

            if expert_mask.any():
                expert_inputs = inputs[expert_mask]  # [n_i, seq_len, hidden]
                expert_output, _, _, expert_aux = self.experts[expert_idx](
                    expert_inputs,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    current_state=None,
                    current_depth=current_depth,
                    block_ids=block_ids,
                )
                output[expert_mask] = expert_output
                # Weight aux loss by proportion of batch
                total_aux_loss += expert_aux * expert_mask.sum().float() / batch_size

        # Compute load balancing loss
        balance_loss = self._compute_balance_loss(probs)
        total_aux_loss = total_aux_loss + self.balance_loss_coef * balance_loss

        # Update metrics
        self._update_metrics(expert_indices, probs, balance_loss)

        return output, past_key_values, current_state, total_aux_loss

    def _compute_balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage 50/50 expert usage.

        Args:
            probs: Routing probabilities [batch, 2]

        Returns:
            Scalar loss - MSE from uniform distribution
        """
        # Average probability per expert across batch
        avg_probs = probs.mean(dim=0)  # [2]

        # Target: uniform distribution (0.5, 0.5)
        target = torch.ones_like(avg_probs) / self.num_experts

        # L2 loss
        balance_loss = F.mse_loss(avg_probs, target)

        return balance_loss

    def _update_metrics(
        self,
        expert_indices: torch.Tensor,
        probs: torch.Tensor,
        balance_loss: torch.Tensor,
    ) -> None:
        """Update routing metrics for logging.

        Metrics are compatible with the web app's Research tab charts:
        - routing/expert_*_weight: Individual expert routing weights (probabilities)
        - expert_selection/expert_*_count: Actual selection counts (k=1 sparse usage)
        - routing/entropy: Routing balance (high = balanced, low = collapsed)
        - routing/concentration: Max weight (measures collapse)
        - routing/variance: Routing stability across experts
        - routing/balance: How close to uniform distribution (1.0 = perfect)
        - architecture/alibi_usage: % sequences using ALiBi
        - architecture/rope_usage: % sequences using RoPE
        """
        # Update cumulative selection counts (actual k=1 expert usage)
        for expert_idx in range(self.num_experts):
            count = (expert_indices == expert_idx).sum().item()
            self.expert_selection_counts[expert_idx] += count

        # Calculate mean routing weights per expert across batch
        expert_weights = probs.mean(dim=0)  # [num_experts]

        # Per-expert routing weights (for "Expert Routing Weights" chart)
        # These show routing probabilities (soft weights)
        for i in range(self.num_experts):
            self._metrics[f"routing/expert_{i}_weight"] = expert_weights[i].item()

        # Per-expert actual selection counts (for "Expert Selection" chart)
        # These show actual k=1 sparse usage (hard counts)
        for i in range(self.num_experts):
            self._metrics[f"expert_selection/expert_{i}_count"] = self.expert_selection_counts[i].item()

        # Architecture-specific metrics (aggregate by architecture type)
        arch_usage = {arch: 0.0 for arch in self.architectures}
        for i in range(self.num_experts):
            arch = self.architectures[i % len(self.architectures)]
            arch_usage[arch] += expert_weights[i].item()

        # Log aggregated architecture usage
        for arch, usage in arch_usage.items():
            self._metrics[f"architecture/{arch}_usage"] = usage * 100.0

        # Entropy: H = -Σ(p_i * log(p_i))
        probs_safe = expert_weights + 1e-10
        entropy = -(probs_safe * probs_safe.log()).sum()
        self._metrics["routing/entropy"] = entropy.item()

        # Concentration: max routing weight
        concentration = expert_weights.max()
        self._metrics["routing/concentration"] = concentration.item()

        # Variance: measures routing stability
        variance = expert_weights.var()
        self._metrics["routing/variance"] = variance.item()

        # Balance: distance from uniform distribution
        uniform_weight = 1.0 / self.num_experts
        max_deviation = (expert_weights - uniform_weight).abs().max()
        balance = 1.0 - max_deviation.item()
        self._metrics["routing/balance"] = balance

        # Debug metrics
        self._metrics["routing/balance_loss"] = balance_loss.item()
        self._metrics["routing/avg_confidence"] = probs.max(dim=-1)[0].mean().item()

    def get_metrics(self) -> Dict[str, float]:
        """Get current routing metrics."""
        return self._metrics

    def log_gradient_dynamics(self) -> Optional[Dict[str, float]]:
        """Log gradient statistics for all experts.

        Returns flat dict with per-expert gradient norms and variance for web app.

        Returns:
            {
                "expert_0_grad_norm": 0.12,  # ALiBi
                "expert_0_grad_var": 0.003,
                "expert_1_grad_norm": 0.08,   # RoPE
                "expert_1_grad_var": 0.002,
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
                metrics[f"expert_{expert_idx}_grad_norm"] = sum(
                    g**2 for g in grad_norms
                ) ** 0.5
            if grad_vars:
                metrics[f"expert_{expert_idx}_grad_var"] = sum(grad_vars) / len(
                    grad_vars
                )

        return metrics if metrics else None

    def __repr__(self) -> str:
        return (
            f"Prismatic(num_experts={self.num_experts}, "
            f"version={self.__version__})"
        )
