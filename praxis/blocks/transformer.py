from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.attention import ATTENTION_REGISTRY
from praxis.normalization import NORMALIZATION_REGISTRY
from praxis.orchestration import EXPERT_REGISTRY
from praxis.residuals import RESIDUAL_REGISTRY
from praxis.utils import norm_scaling

# Conditional hivemind import - only used if hivemind integration is loaded
try:
    from hivemind.moe.server.layers.custom_experts import register_expert_class

    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False

    # Create a no-op decorator when hivemind is not available
    def register_expert_class(name, shape_fn):
        def decorator(cls):
            return cls

        return decorator


ConfigType = TypeVar("ConfigType", bound="AutoConfig")
input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class TransformerBlock(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".
    """

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.attn_res = RESIDUAL_REGISTRY.get(config.residual_type)(config.hidden_size)
        self.attn_norm = NORMALIZATION_REGISTRY[config.norm_type](
            config.hidden_size, eps=config.epsilon
        )
        self.attn = ATTENTION_REGISTRY[config.attention_type](config)

        self.ffn_res = RESIDUAL_REGISTRY.get(config.residual_type)(config.hidden_size)
        self.ffn_norm = NORMALIZATION_REGISTRY[config.norm_type](
            config.hidden_size, eps=config.epsilon
        )
        self.ffn = EXPERT_REGISTRY[config.expert](config)
        self.use_scaler = config.scaled

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Union[List[Any], Dict[str, Any]]] = None,
        current_state: Optional[Any] = None,
        current_depth: int = 0,
        block_ids: Optional[Tensor] = None,
        router_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[
        Tensor, Optional[Union[List[Any], Dict[str, Any]]], Optional[Any], Tensor
    ]:
        """
        Forward pass through the transformer block.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/values for faster inference
            current_state: Optional current layer state
            current_depth: Current depth in the network
            block_ids: Optional block identification tensor
            router_weights: Optional weights from router for expert gating

        Returns:
            Tuple containing:
                - Output tensor
                - Updated past key values
                - Updated layer state (None in this implementation)
                - Auxiliary loss
        """

        aux_loss = 0

        # =========== Attention Block =============
        residual, beta = self.attn_res.connect_width(inputs)

        # Apply pre-normalization (if configured)
        attn_input = self.attn_norm(self.attn_res.format_state(residual), mode="pre")
        if self.use_scaler:
            attn_input = norm_scaling(attn_input, current_depth)
        attn_output, past_key_values, aux_loss = self.attn(
            attn_input, attention_mask, past_key_values, block_ids, current_depth
        )
        # Apply post-normalization (if configured)
        attn_output = self.attn_norm(attn_output, mode="post")

        attn_merged = self.attn_res.connect_depth(residual, attn_output, beta)

        # =========== FeedForward Block ===========
        residual, beta_ffn = self.ffn_res.connect_width(
            self.ffn_res.format_state(attn_merged)
        )

        # Apply pre-normalization (if configured)
        ffn_input = self.ffn_norm(self.ffn_res.format_state(residual), mode="pre")
        if self.use_scaler:
            ffn_input = norm_scaling(ffn_input, current_depth)
        ffn_output = self.ffn(ffn_input, current_depth)
        # Apply post-normalization (if configured)
        ffn_output = self.ffn_norm(ffn_output, mode="post")

        if torch.is_tensor(router_weights):
            # this is a super hack because hivemind
            if not self._is_zero_tensor(router_weights):
                ffn_output = ffn_output * router_weights

        # Merge expansions
        final_output = self.ffn_res.connect_depth(residual, ffn_output, beta_ffn)
        return self.ffn_res.format_state(final_output), past_key_values, None, aux_loss

    def _is_zero_tensor(self, tensor: Tensor, tolerance: float = 1e-10) -> bool:
        """
        Check if a tensor contains all zeros (or values close to zero).

        Args:
            tensor: The tensor to check
            tolerance: Threshold for considering values as zero

        Returns:
            True if the tensor contains all zeros or near-zeros, False otherwise
        """
        if tensor.dtype == torch.int64:
            return bool(torch.all(tensor == 0))
        return bool(torch.abs(tensor).max().item() < tolerance)
