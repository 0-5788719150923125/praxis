"""
Distance Router: SMEAR with Parameter Distance Loss

Extends SMEAR with a diversity loss that encourages experts to maintain
different parameters from the base (expert 0). This addresses the challenge
that merged experts in SMEAR can collapse toward similar parameter values,
limiting gradient manipulation options.

The distance loss computes L2 distance between each expert's parameters
and the base expert's parameters, encouraging divergence.
"""

from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

from praxis.routers.smear import SMEAR


class Distance(SMEAR):
    """
    Distance Router: SMEAR with Parameter Diversity Loss.

    Extends SMEAR's soft parameter merging with an auxiliary loss that
    encourages experts to maintain distinct parameters from the base expert.
    This prevents expert collapse and enables better gradient manipulation.

    The diversity loss is:
        L_div = -Σ ||θ_i - θ_0||_2 for i in [1, num_experts)

    Where θ_i are the parameters of expert i and θ_0 are base expert parameters.
    The negative sign encourages distance (higher loss = more similar).
    """

    __version__ = "1.0.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize Distance router with parameter diversity loss.

        Args:
            config: Configuration object with diversity_loss_coef attribute
            layout: Layout type (not used, kept for compatibility)
            *args: Additional arguments
            **kwargs: Additional keyword arguments including 'experts' list
        """
        super().__init__(config, layout, *args, **kwargs)

        # Diversity loss coefficient - controls strength of parameter distance loss
        self.diversity_loss_coef = getattr(config, "diversity_loss_coef", 0.01)

        print(f"[DISTANCE v{self.__version__}] Parameter diversity loss enabled")
        print(f"  Num experts: {len(self.experts)}")
        print(f"  Diversity coefficient: {self.diversity_loss_coef}")
        print(f"  Base expert: 0, comparing against experts 1-{len(self.experts)-1}")

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
        """Router mode forward pass with diversity loss."""
        # Call parent's router forward
        output, pkv, state, aux_loss = super()._router_forward(
            layer, inputs, attention_mask, past_key_values, current_state, current_depth, block_ids
        )

        # Compute and add diversity loss
        diversity_loss = self._compute_diversity_loss()
        total_aux_loss = aux_loss + self.diversity_loss_coef * diversity_loss

        # Store diversity loss in metrics
        self._metrics["routing/diversity_loss"] = diversity_loss.item()

        return output, pkv, state, total_aux_loss

    def _direct_forward(
        self,
        inputs: torch.Tensor,
        current_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Direct mode forward pass with diversity loss."""
        # Call parent's direct forward
        output, state, aux_loss = super()._direct_forward(inputs, current_state)

        # Compute and add diversity loss
        diversity_loss = self._compute_diversity_loss()
        total_aux_loss = aux_loss + self.diversity_loss_coef * diversity_loss

        # Store diversity loss in metrics
        self._metrics["routing/diversity_loss"] = diversity_loss.item()

        return output, state, total_aux_loss

    def _compute_diversity_loss(self) -> torch.Tensor:
        """
        Compute parameter distance loss encouraging expert diversity.

        Measures L2 distance between each expert's parameters and the base
        expert (expert 0). Returns negative sum to encourage maximizing distance.

        Returns:
            Scalar diversity loss (negative distance sum)
        """
        if len(self.experts) <= 1:
            # No diversity loss if only one expert
            return torch.tensor(0.0, device=next(self.experts[0].parameters()).device)

        base_params = list(self.experts[0].parameters())
        diversity_loss = torch.tensor(
            0.0, device=base_params[0].device, dtype=base_params[0].dtype
        )

        # Compare each expert to the base expert
        for expert_idx in range(1, len(self.experts)):
            expert_params = list(self.experts[expert_idx].parameters())

            for base_p, expert_p in zip(base_params, expert_params):
                # Compute L2 distance between parameters
                distance = torch.norm(expert_p - base_p)
                # Negative to encourage distance (minimize negative = maximize distance)
                diversity_loss -= distance

        # Normalize by number of comparisons
        num_comparisons = len(self.experts) - 1
        diversity_loss = diversity_loss / num_comparisons

        return diversity_loss

    def __repr__(self) -> str:
        return (
            f"Distance(num_experts={len(self.experts)}, "
            f"diversity_coef={self.diversity_loss_coef}, "
            f"version={self.__version__})"
        )
