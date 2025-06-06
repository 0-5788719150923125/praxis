from typing import Any, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.containers.loss import LossContainer

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class Taxus(nn.Module):
    """
    Taxus: A depth-buying router that allows models to dynamically exit at optimal layers.

    Named after the Taxus (yew) tree, known for its layered growth rings and
    economic value - fitting for a router that "buys" computational layers.

    Each decoder layer has an associated cost, and the model learns to balance
    computational expense against task requirements, enabling early exits when
    confident and deeper processing when needed.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: ConfigType,
        target_depth_ratio: float = 0.5,
        min_exit_layer: int = 1,
        temperature: float = 1.0,
        entropy_weight: float = 0.01,
        usage_weight: float = 0.1,
        budget_weight: float = 1.0,  # Increased from 0.1 for stronger budget pressure
        computational_budget: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.debug = False
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.target_depth_ratio = target_depth_ratio
        self.min_exit_layer = min_exit_layer
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.usage_weight = usage_weight
        self.budget_weight = budget_weight
        self.computational_budget = computational_budget or (0.7 * self.depth)

        # Exit decision gates for each layer
        self.exit_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size // 4, 2),  # [continue, exit]
                )
                for _ in range(self.depth)
            ]
        )

        # Initialize exit gates with strong bias towards target depth
        target_layer = int(self.target_depth_ratio * self.depth)
        for i, gate in enumerate(self.exit_gates):
            if i >= self.min_exit_layer:
                with torch.no_grad():
                    # Strong bias to exit near target layer
                    distance_from_target = abs(i - target_layer) / self.depth
                    # Stronger bias: range from -2 to +2
                    exit_bias = 2.0 * (1.0 - 2.0 * distance_from_target)
                    gate[-1].bias[0] = -exit_bias  # Continue logit
                    gate[-1].bias[1] = exit_bias  # Exit logit

        # Layer costs - learnable parameters that increase with depth
        # Use linear growth to avoid exponential instability
        self.layer_costs = nn.Parameter(torch.linspace(0.1, 0.5, self.depth))

        # Exit confidence predictor (helps with auxiliary loss)
        self.confidence_predictor = nn.Linear(self.hidden_size, 1)

        # Statistics tracking for debugging
        if self.debug:
            self.register_buffer("exit_counts", torch.zeros(self.depth))
            self.register_buffer("total_samples", torch.tensor(0))

    def forward(
        self,
        layer: nn.Module,
        inputs: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Any],
        current_state: Optional[Tensor],
        current_depth: int,
        block_ids: Optional[Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Any], Optional[Tensor], LossContainer]:
        """
        Route through layer with early exit capability.

        Args:
            layer: The decoder layer to potentially execute
            inputs: Input hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            past_key_values: Optional cached key/values
            current_state: Optional current state
            current_depth: Current layer index (0-based)
            block_ids: Optional block identifiers

        Returns:
            Tuple of (outputs, past_key_values, current_state, losses)
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Debug logging - test if we're hitting this method at all
        if self.debug and not self.training:
            print(
                f"DEBUG: Taxus forward called - layer {current_depth}, batch_size={batch_size}, training={self.training}"
            )

        # Initialize loss container
        losses = LossContainer()

        # Cannot exit before minimum layer
        if current_depth < self.min_exit_layer:
            # Process normally without exit consideration
            outputs, new_kv, new_state, layer_loss = layer(
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
                *args,
                **kwargs,
            )
            if isinstance(layer_loss, (int, float, Tensor)):
                losses.add_loss("layer", layer_loss)
            elif isinstance(layer_loss, LossContainer):
                losses.add_loss_container(layer_loss)
            return outputs, new_kv, new_state, losses

        # Compute exit decision
        # Pool over sequence dimension for decision making
        pooled_hidden = inputs.mean(dim=1)  # [batch_size, hidden_dim]

        # Get exit logits and probabilities
        exit_logits = self.exit_gates[current_depth](pooled_hidden)
        exit_probs = F.softmax(exit_logits / self.temperature, dim=-1)

        # Compute confidence for this exit point
        confidence = torch.sigmoid(self.confidence_predictor(pooled_hidden))

        # Sample exit decision (differentiable during training)
        if self.training:
            # Gumbel-softmax for differentiable discrete sampling
            exit_decision = F.gumbel_softmax(
                exit_logits, tau=self.temperature, hard=True
            )
            should_exit = exit_decision[:, 1]  # [batch_size]
        else:
            # Deterministic decision during inference
            should_exit = (exit_probs[:, 1] > 0.5).float()

        # Debug logging for exit decisions
        if self.debug and not self.training:
            avg_exit_prob = exit_probs[:, 1].mean().item()
            print(
                f"DEBUG: Taxus layer {current_depth} exit decision: avg_exit_prob={avg_exit_prob:.3f}, should_exit_any={should_exit.max().item():.1f}"
            )

        # Track statistics in debug mode
        if hasattr(self, "exit_counts") and not self.training:
            exit_rate = should_exit.mean().item()
            self.exit_counts[current_depth] += exit_rate * batch_size
            self.total_samples += batch_size

        # Compute layer cost
        layer_cost = self.layer_costs[current_depth]

        # Auxiliary losses (all positive, bounded)
        # 1. Entropy regularization (encourage decisive exits)
        entropy = -(exit_probs * torch.log(exit_probs + 1e-8)).sum(dim=-1).mean()
        # Penalize low entropy (indecisive exits) - keep positive
        losses.add_loss(
            "taxus_entropy", self.entropy_weight * torch.clamp(-entropy, 0, 10)
        )

        # 2. Usage regularization (penalize deviation from target depth)
        current_depth_ratio = (current_depth + 1) / self.depth

        # Asymmetric loss: penalize continuing past target more than exiting before
        if current_depth_ratio > self.target_depth_ratio:
            # Past target - strong penalty for continuing
            usage_deviation = 2.0 * (current_depth_ratio - self.target_depth_ratio) ** 2
        else:
            # Before target - mild penalty for early exit
            usage_deviation = 0.5 * (current_depth_ratio - self.target_depth_ratio) ** 2

        # Scale by whether we're exiting or continuing
        usage_loss = usage_deviation * (1 - should_exit).mean()
        losses.add_loss("taxus_usage", self.usage_weight * usage_loss)

        # 3. Confidence-based loss (bounded)
        confidence_loss = confidence.mean() * current_depth_ratio
        losses.add_loss("taxus_confidence", 0.1 * torch.clamp(confidence_loss, 0, 2))

        # 4. Computational cost tracking (bounded)
        batch_cost = layer_cost * (1 - should_exit).mean()
        losses.add_loss("taxus_cost", torch.clamp(batch_cost, 0, 1))

        # 5. Early exit incentive (reformulated as positive loss)
        # Penalize continuing at later depths instead of rewarding early exits
        continue_penalty = (1 - should_exit).mean() * current_depth_ratio
        losses.add_loss("taxus_continue_penalty", 0.5 * continue_penalty)

        # 5b. Target depth encouragement
        # Reward exits near the target depth
        target_layer = int(self.target_depth_ratio * self.depth)
        distance_from_target = abs(current_depth - target_layer) / self.depth
        target_bonus = should_exit.mean() * (1 - distance_from_target)
        # Convert reward to penalty (lower is better)
        losses.add_loss("taxus_target_bonus", 0.5 * (1 - target_bonus))

        # 6. Budget constraint loss - bidirectional pressure
        if self.training and self.computational_budget is not None:
            # Expected cost if we exit at this layer
            # We pay for all layers up to current, weighted by exit probability
            expected_cost = (
                sum(self.layer_costs[: current_depth + 1]) * should_exit.mean()
            )

            # Expected remaining cost if we continue
            if current_depth < self.depth - 1:
                continue_prob = (1 - should_exit).mean()
                # Assume we'll use remaining budget efficiently
                remaining_layers = self.depth - current_depth - 1
                expected_remaining = continue_prob * sum(
                    self.layer_costs[current_depth + 1 :]
                )
                total_expected_cost = expected_cost + expected_remaining
            else:
                total_expected_cost = sum(self.layer_costs)  # Last layer, must use all

            # Bidirectional loss: penalize both over and under-utilization
            budget_diff = total_expected_cost - self.computational_budget
            budget_loss = torch.abs(budget_diff) + 0.1 * torch.square(budget_diff)
            losses.add_loss(
                "taxus_budget", self.budget_weight * torch.clamp(budget_loss, 0, 2)
            )

        # Add exit signal to loss container BEFORE any early returns
        avg_exit_prob = exit_probs[:, 1].mean()

        if self.training:
            # During training, pass the soft probability to allow gradient flow
            losses.add_loss("taxus_exit_prob", avg_exit_prob)
            exit_decision_prob = (
                exit_decision[:, 1].mean()
                if "exit_decision" in locals()
                else avg_exit_prob
            )
            losses.add_loss("taxus_should_exit", exit_decision_prob)
        else:
            # During inference, use hard threshold
            should_exit_decoder = avg_exit_prob.item() > 0.5
            losses.add_loss("taxus_should_exit", float(should_exit_decoder))

            # Debug logging for decoder exit signal
            if self.debug:
                print(
                    f"DEBUG: Taxus layer {current_depth}: avg_exit_prob={avg_exit_prob.item():.3f}, should_exit_decoder={should_exit_decoder}"
                )

        # Check if we should exit (for any samples in the batch)
        exit_mask = should_exit.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]

        if not self.training and should_exit.all():
            # All samples want to exit - skip layer processing
            return inputs, past_key_values, current_state, losses

        # Process through layer
        outputs, new_kv, new_state, layer_loss = layer(
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
            *args,
            **kwargs,
        )

        # Add layer losses
        if isinstance(layer_loss, (int, float, Tensor)):
            losses.add_loss("layer", layer_loss)
        elif isinstance(layer_loss, LossContainer):
            losses.add_loss_container(layer_loss)

        # During training, blend outputs based on exit decision
        # This allows gradients to flow through both paths
        if self.training:
            outputs = exit_mask * inputs + (1 - exit_mask) * outputs

        return outputs, new_kv, new_state, losses

    def get_exit_statistics(self) -> dict:
        """Get debugging statistics about exit patterns."""
        if not hasattr(self, "exit_counts"):
            return {}

        total = self.total_samples.item()
        if total == 0:
            return {}

        exit_rates = (self.exit_counts / total).cpu().numpy()
        avg_depth = sum(i * rate for i, rate in enumerate(exit_rates))

        return {
            "exit_rates_by_layer": exit_rates.tolist(),
            "average_exit_depth": avg_depth,
            "average_depth_ratio": avg_depth / self.depth,
            "total_samples": total,
        }

    def reset_statistics(self) -> None:
        """Reset exit statistics."""
        if hasattr(self, "exit_counts"):
            self.exit_counts.zero_()
            self.total_samples.zero_()
