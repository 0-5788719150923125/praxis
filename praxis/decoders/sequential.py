import math
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn

from praxis.containers import LossContainer
from praxis.decoders.base import BaseDecoder
from praxis.decoders.checkpoint import create_forward, should_checkpoint

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class SequentialDecoder(BaseDecoder):
    """The default decoder. Walks layers one at a time, with the controller
    picking the next expert at each step and the halting strategy deciding
    when to stop. Supports multi-cycle "reasoning" passes through the expert
    pool when ``meta`` contains ``use_reason``.
    """

    def __init__(self, config: ConfigType) -> None:
        # Heuristically determine reasoning steps based on depth and num_experts
        if "use_reason" in config.meta:
            # Calculate how many "cycles" through the expert pool we make
            # Each complete cycle is considered a reasoning step
            self.steps = max(1, math.ceil(config.depth / config.num_experts))
        else:
            self.steps = 1
        super().__init__(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[List[Any], Dict[str, Any]]] = None,
        current_state: Optional[List[Any]] = None,
        block_ids: Optional[Tensor] = None,
        losses: LossContainer = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[
        Tensor, Optional[Union[List[Any], Dict[str, Any]]], Optional[List[Any]], Tensor
    ]:
        """
        Forward pass through the sequential decoder.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/values for faster inference
            current_state: Optional current layer states
            block_ids: Optional block identification tensor
            losses: A storage class for auxiliary losses
            labels: Optional labels tensor for supervised learning

        Returns:
            Tuple containing:
                - Output hidden states
                - Updated past key values
                - Updated layer states
                - Auxiliary loss container
        """

        _, seq_len, _ = hidden_states.shape

        effective_depth = self.halting.get_depth()
        self.halting.seed(hidden_states)

        current_route: List[int] = []
        realized_widths: List[float] = []  # active width fraction per executed step
        depth_prints: List[Tensor] = []  # per-depth hidden-state fingerprint
        # Entry fingerprint: the trajectory starts at the decoder input, so even a
        # single executed depth yields one transition (entry -> first cluster).
        if hidden_states.dim() == 3:
            depth_prints.append(hidden_states.detach().float().mean(dim=(0, 1)))

        controller_state = None
        sequential_experts: List[nn.Module] = list(self.locals) + list(self.remotes)
        ordered_experts: List[nn.Module] = self.controller.sort_experts(
            sequential_experts.copy()
        )

        for total_calls, (reason_step, current_depth) in enumerate(
            product(range(self.steps), range(effective_depth))
        ):
            # Enforce depth budget - exit if we've made enough expert calls
            if total_calls >= effective_depth:
                break

            hidden_states, controller_state, controller_loss, next_expert_idx = (
                self.controller.get_next_expert(
                    hidden_states,
                    controller_state,
                    sequential_experts,
                    ordered_experts,
                    current_route,
                    current_depth,
                )
            )

            losses.add_loss_container(controller_loss)

            if next_expert_idx is None:
                break

            expert = ordered_experts[next_expert_idx]

            # Handle current_state access with modulo for multiple reasoning steps
            state_idx = (
                next_expert_idx % len(ordered_experts)
                if current_state is not None
                else None
            )
            layer_state = (
                current_state[state_idx] if current_state is not None else None
            )
            realized_widths.append(self.width.fraction(current_depth, self.depth))
            with self.width.scope([expert], current_depth, max_depth=self.depth):
                (
                    hidden_states,
                    past_key_values,
                    layer_state,
                    decoder_loss,
                    exit_signal,
                ) = create_forward(
                    expert,
                    self.controller,
                    self.manager,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    layer_state,
                    current_depth,
                    block_ids,
                    should_checkpoint(
                        self.training, current_depth, self.checkpoint_every
                    ),
                )

            # Update route immediately after expert execution
            current_route = self.controller.update_route(
                hidden_states, current_route, current_depth, next_expert_idx
            )

            # Per-depth representation fingerprint: mean over batch+seq -> [D],
            # shape-robust across compression. The trajectory of these over depth
            # is the spectral-attractor probe (next/harmonic_memory_velocity.md):
            # does the iteration settle to a fixed point, and in discrete hops or
            # a smooth drift? Detached - diagnostic only.
            if hidden_states.dim() == 3:
                depth_prints.append(hidden_states.detach().float().mean(dim=(0, 1)))

            # Handle expert decoder loss (can be scalar/tensor or LossContainer)
            if isinstance(decoder_loss, LossContainer):
                losses.add_loss_container(decoder_loss)
            else:
                losses.add_loss("decoder", decoder_loss)

            # Check halting strategy at loop boundaries
            if self.halting.check(hidden_states, current_depth):
                break

            # Check for Taxus early exit signal (passed directly, not through LossContainer)
            if exit_signal is not None:
                # Debug: Show exit signal during inference and training
                if hasattr(self, "config") and getattr(self.config, "debug", False):
                    mode = "training" if self.training else "inference"
                    print(
                        f"DEBUG: Decoder got exit_signal at depth {current_depth} ({mode}): {exit_signal}"
                    )

                if exit_signal:
                    # Early exit signaled - stop decoding
                    if hasattr(self, "config") and getattr(self.config, "debug", False):
                        mode = "training" if self.training else "inference"
                        print(f"DEBUG: Early exit at depth {current_depth} ({mode})!")
                    break
            hidden_states = self.compressor.reduce_sequence(hidden_states)
            block_ids = self.compressor.reduce_block_ids(block_ids)
            hidden_states = self.post_layer(hidden_states, current_depth)
            if current_state is not None:
                # Use the same state index for updating
                current_state[state_idx] = layer_state

        # Mean active width actually used this forward. Varies over training even
        # under the fixed schedule, because halting samples how many depths run.
        if realized_widths:
            self._width_realized = sum(realized_widths) / len(realized_widths)

        # Depth-trajectory (spectral-attractor) metrics from the fingerprints:
        # relative step size per depth transition, plus a convergence ratio
        # (<1 = settling toward a fixed point, ~1 = no convergence) and a jump
        # concentration (high = one big hop then settle = discrete; ~1 = smooth).
        self._depth_metrics = {}
        if len(depth_prints) >= 2:
            prints = torch.stack(depth_prints)  # [n, D]
            steps = (prints[1:] - prints[:-1]).norm(dim=-1) / (
                prints[:-1].norm(dim=-1) + 1e-8
            )
            s = steps.tolist()
            for i, v in enumerate(s):
                self._depth_metrics[f"depth/step_d{i}"] = v
            if len(s) >= 2:  # a ratio / concentration needs 2+ transitions
                self._depth_metrics["depth/convergence_ratio"] = s[-1] / (s[0] + 1e-8)
                self._depth_metrics["depth/jump_concentration"] = max(s) / (
                    sum(s) / len(s) + 1e-8
                )

        hidden_states = self.compressor.expand_sequence(hidden_states, seq_len)

        hidden_states = self.post_decoding(hidden_states)

        self.controller.post_forward(hidden_states, current_route)

        # Apply feature sorting
        hidden_states = self.order(hidden_states)

        return hidden_states, past_key_values, current_state, losses
