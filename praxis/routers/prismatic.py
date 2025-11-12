"""
Prismatic Attention: Sparse Bidirectional Temporal Routing

v6.0 - Sparse Mixture-of-Eyes:
Routes sequences to one of two "eyes" with different temporal perspectives:
- Expert 0 (Forward Eye): Standard causal masking - sees past, infers future
- Expert 1 (Backward Eye): Inverted causal masking - sees future, infers past

Unlike previous SMEAR approach (soft parameter merging), this uses sparse routing:
each sequence is processed by ONE expert with the appropriate temporal mask.

Philosophy:
"Temporal Perspective Selection" - The model learns which temporal perspective
is most useful for each sequence, then commits to that perspective for processing.
Clean separation: router creates masks and routes, experts just process with given mask.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Prismatic(nn.Module):
    """
    Prismatic router with sparse bidirectional temporal masking.

    Routes entire sequences to one of two experts:
    - Expert 0 (Forward Eye): Sees past, infers future
    - Expert 1 (Backward Eye): Sees future, infers past

    The router:
    1. Computes routing probabilities per sequence
    2. Selects ONE expert per sequence (sparse, top-1)
    3. Creates appropriate mask (forward or backward)
    4. Executes selected expert with that mask
    5. Applies load balancing loss to encourage 50/50 usage

    Key design: Router handles all masking logic. Experts are identical architecture,
    differentiated only by which mask the router passes to them.
    """

    __version__ = "6.0.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize Prismatic with sparse bidirectional routing.

        Args:
            config: Configuration with num_experts, hidden_size
            layout: Unused, kept for compatibility
            **kwargs: Must include 'experts' parameter
        """
        super().__init__()

        self.num_experts = config.num_experts
        if self.num_experts != 2:
            raise ValueError(
                f"Prismatic requires exactly 2 experts (forward/backward), got {self.num_experts}"
            )

        self.hidden_size = config.hidden_size

        # Get experts - should be identical architecture
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

        print(f"[PRISMATIC v6.0] Sparse routing with {len(self.experts)} experts")
        print(f"  Expert 0: Forward eye (sees past)")
        print(f"  Expert 1: Backward eye (sees future)")
        print(f"  Routing: Sequence-level, top-1 sparse")

        # Router: learns to select forward vs backward perspective per sequence
        # Uses sequence representation (mean pooling) to decide
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        # Load balancing
        self.balance_loss_coef = getattr(config, "router_balance_loss_coef", 0.01)

        # Metrics
        self._metrics: Dict[str, float] = {}

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
        Router mode: select expert per sequence and execute with appropriate mask.

        Args:
            layer: The LocalLayer wrapper (unused in sparse routing)
            inputs: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional padding mask (combined with causal mask)
            past_key_values: KV cache (if using)
            current_state: Current hidden state (for routing)
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

        # Create forward and backward masks
        # These will be passed to experts via attention_mask parameter
        forward_mask = self._create_causal_mask(
            seq_len, direction="forward", device=inputs.device
        )
        backward_mask = self._create_causal_mask(
            seq_len, direction="backward", device=inputs.device
        )

        # Pad masks for ghost token (used by HexAttention)
        # Forward: ghost at START (standard ghostmax - escape from past)
        # Backward: ghost at END (inverted ghostmax - escape from future)
        kv_len = seq_len + 1

        # Forward mask: ghost at position 0 (start) - standard ghostmax
        # Prepend zero column - ghost always accessible
        forward_mask = torch.cat([torch.zeros(seq_len, 1, device=inputs.device), forward_mask], dim=1)

        # Backward mask: ghost at position seq_len (end) - inverted ghostmax
        # Append zero column - ghost always accessible
        backward_mask = torch.cat([backward_mask, torch.zeros(seq_len, 1, device=inputs.device)], dim=1)

        # Combine with provided attention_mask if present (for padding)
        if attention_mask is not None:
            # attention_mask from padding is typically [batch, seq_len] or [batch, 1, 1, seq_len]
            # We need to broadcast it to [seq_len, seq_len] format and add
            # For now, assume attention_mask is handled separately by the expert
            # TODO: Properly combine padding mask with causal mask
            pass

        # Execute experts in parallel batches (efficient sparse MoE)
        # Group sequences by expert assignment and execute each expert once

        output = torch.zeros_like(inputs)
        total_aux_loss = 0.0

        # Create masks for which sequences go to which expert
        expert_0_mask = (expert_indices == 0)  # [batch]
        expert_1_mask = (expert_indices == 1)  # [batch]

        # Execute expert 0 on all sequences assigned to it (forward eye)
        # Ghost at START for forward perspective (standard ghostmax - escape from past)
        if expert_0_mask.any():
            expert_0_inputs = inputs[expert_0_mask]  # [n0, seq_len, hidden]
            expert_0_output, _, _, expert_0_aux = self.experts[0](
                expert_0_inputs,
                attention_mask=forward_mask,
                past_key_values=None,
                current_state=None,
                current_depth=current_depth,
                block_ids=block_ids,
                ghost_position="start",  # Standard ghostmax
            )
            output[expert_0_mask] = expert_0_output
            # Weight aux loss by proportion of batch
            total_aux_loss += expert_0_aux * expert_0_mask.sum().float() / batch_size

        # Execute expert 1 on all sequences assigned to it (backward eye)
        # Ghost at END for backward perspective (inverted ghostmax - escape from future)
        if expert_1_mask.any():
            expert_1_inputs = inputs[expert_1_mask]  # [n1, seq_len, hidden]
            expert_1_output, _, _, expert_1_aux = self.experts[1](
                expert_1_inputs,
                attention_mask=backward_mask,
                past_key_values=None,
                current_state=None,
                current_depth=current_depth,
                block_ids=block_ids,
                ghost_position="end",  # Inverted ghostmax
            )
            output[expert_1_mask] = expert_1_output
            # Weight aux loss by proportion of batch
            total_aux_loss += expert_1_aux * expert_1_mask.sum().float() / batch_size

        # Compute load balancing loss and add to expert aux losses
        balance_loss = self._compute_balance_loss(probs)
        total_aux_loss = total_aux_loss + self.balance_loss_coef * balance_loss

        # Update metrics
        self._update_metrics(expert_indices, probs, balance_loss)

        return output, past_key_values, current_state, total_aux_loss

    def _create_causal_mask(
        self, seq_len: int, direction: str, device: torch.device
    ) -> torch.Tensor:
        """
        Create causal attention mask.

        Args:
            seq_len: Sequence length
            direction: "forward" or "backward"
            device: Device to create mask on

        Returns:
            Additive mask [seq_len, seq_len] with 0 for allowed, -inf for masked
        """
        if direction == "forward":
            # Standard causal: position i can see j where j <= i
            # Lower triangular matrix (including diagonal)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
        elif direction == "backward":
            # Inverted causal: position i can see j where j >= i
            # Upper triangular matrix (including diagonal)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=-1
            )
        else:
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction}")

        return mask

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
        - routing/expert_*_weight: Individual expert routing weights
        - routing/entropy: Routing balance (high = balanced, low = collapsed)
        - routing/concentration: Max weight (measures collapse)
        - routing/variance: Routing stability across experts
        - routing/balance: How close to uniform distribution (1.0 = perfect)
        """
        # Calculate mean routing weights per expert across batch
        expert_weights = probs.mean(dim=0)  # [num_experts]

        # Per-expert routing weights (for "Expert Routing Weights" chart)
        # Web app expects: routing/expert_0_weight, routing/expert_1_weight
        for i in range(self.num_experts):
            self._metrics[f"routing/expert_{i}_weight"] = expert_weights[i].item()

        # Entropy: H = -Σ(p_i * log(p_i))
        # Measures routing balance: high = balanced, low = collapsed
        probs_safe = expert_weights + 1e-10  # Avoid log(0)
        entropy = -(probs_safe * probs_safe.log()).sum()
        self._metrics["routing/entropy"] = entropy.item()

        # Concentration: max routing weight
        # Measures expert collapse: 1.0 = fully collapsed, 1/N = uniform
        concentration = expert_weights.max()
        self._metrics["routing/concentration"] = concentration.item()

        # Variance: measures routing stability across experts
        # High variance = specialized experts, low variance = uniform
        variance = expert_weights.var()
        self._metrics["routing/variance"] = variance.item()

        # Balance: distance from uniform distribution (1.0 = perfect balance)
        # Computed as 1 - max_deviation_from_uniform
        uniform_weight = 1.0 / self.num_experts
        max_deviation = (expert_weights - uniform_weight).abs().max()
        balance = 1.0 - max_deviation.item()
        self._metrics["routing/balance"] = balance

        # Additional debug metrics
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
                "expert_0_grad_norm": 0.12,
                "expert_0_grad_var": 0.003,
                "expert_1_grad_norm": 0.08,
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
