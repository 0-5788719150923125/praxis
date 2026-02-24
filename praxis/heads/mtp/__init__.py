"""Multi-Token Prediction (MTP) module.

Implements the sequential MTP design from DeepSeek-V3, where each depth
builds on the previous hidden states and incorporates ground-truth position
embeddings. During training, provides an auxiliary loss for denser
supervision. During inference, drafts speculative tokens for verification
by the main model (~1.8x throughput via speculative decoding).

Two execution paths are handled internally:
  Standard (token-level): embeds from nn.Embedding, CE loss vs token IDs
  Encoder (patch-level): patch embeds projected to embed_size, MSE loss
    vs target patch representations — owns the projection and head.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.containers import LossContainer
from praxis.heads.mtp.conv import ConvMTPModule
from praxis.heads.mtp.transformer import TransformerMTPModule

MTP_REGISTRY = {
    "transformer": TransformerMTPModule,
    "conv": ConvMTPModule,
}


@dataclass
class MTPInputs:
    """Bundled inputs for an MTP forward pass."""

    hidden_states: torch.Tensor
    embeds: torch.Tensor
    targets: torch.Tensor
    head: nn.Module
    loss_fn: Callable
    attention_mask: Optional[torch.Tensor] = None


class MultiTokenPrediction(nn.Module):
    """Manages all MTP depths and computes the auxiliary loss.

    Each depth k predicts targets at offset k+2 from the input. Returns
    a LossContainer with a tagged "mtp" loss so the strategy system can
    handle weighting dynamically alongside other losses.
    """

    def __init__(self, config):
        super().__init__()
        self.num_depths = config.mtp_depth
        self.mtp_type = config.mtp_type
        self.encoder_path = config.encoder_type is not None
        module_cls = MTP_REGISTRY[config.mtp_type]
        self.depths = nn.ModuleList([module_cls(config) for _ in range(self.num_depths)])

        # Encoder path owns its own projection and head for patch-level MTP
        if self.encoder_path:
            self.embed_proj = nn.Linear(
                config.hidden_size, config.embed_size, bias=False
            )
            self.patch_head = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type='{self.mtp_type}', "
            f"depths={self.num_depths})"
        )

    def prepare_inputs(self, hidden_states, input_ids, attention_mask, embed_fn, head, patch_embeds=None):
        """Build path-appropriate MTPInputs for the current execution path.


        Args:
            hidden_states: Decoder output [batch, seq, hidden_size]
            input_ids: Original input token IDs [batch, seq]
            attention_mask: Attention mask or None
            embed_fn: Input embedding function (nn.Embedding)
            head: LM head module (token path) — ignored on encoder path
            patch_embeds: Patch embeddings from encoder (encoder path only)
        """
        if self.encoder_path:
            return MTPInputs(
                hidden_states=hidden_states,
                embeds=self.embed_proj(patch_embeds),
                targets=patch_embeds,
                head=self.patch_head,
                loss_fn=lambda p, t: F.mse_loss(p, t),
                attention_mask=attention_mask,
            )
        else:
            return MTPInputs(
                hidden_states=hidden_states,
                embeds=embed_fn(input_ids),
                targets=input_ids,
                head=head,
                loss_fn=lambda p, t: F.cross_entropy(
                    p.reshape(-1, p.size(-1)), t.reshape(-1)
                ),
                attention_mask=attention_mask,
            )

    def forward(self, inputs: MTPInputs):
        total_loss = 0.0
        h_prev = inputs.hidden_states
        depths_computed = 0

        for k, module in enumerate(self.depths):
            offset = k + 1

            # Guard: need enough positions for this depth
            if inputs.embeds.size(1) < offset + 2:
                break

            # Trim hidden states by 1 position from the end
            h_trimmed = h_prev[:, :-1, :]

            # Ground-truth embeddings at the shifted offset
            position_embeds = inputs.embeds[:, offset:]

            # Align lengths
            min_len = min(h_trimmed.size(1), position_embeds.size(1))
            h_trimmed = h_trimmed[:, :min_len, :]
            position_embeds = position_embeds[:, :min_len, :]

            # Trim attention mask if provided
            mask = (
                inputs.attention_mask[:, :min_len]
                if inputs.attention_mask is not None
                else None
            )

            # Run through MTP module
            h_k = module(h_trimmed, position_embeds, mask)

            # Predictions via head, trimmed for alignment
            preds = inputs.head(h_k)[:, :-1]

            # Targets at offset k+2
            targets = inputs.targets[:, offset + 1 :]

            # Align predictions and targets
            min_out = min(preds.size(1), targets.size(1))
            preds = preds[:, :min_out].contiguous()
            targets = targets[:, :min_out].contiguous()

            total_loss = total_loss + inputs.loss_fn(preds, targets)

            # Chain: this depth's output becomes input for next depth
            h_prev = h_k
            depths_computed += 1

        losses = LossContainer()
        if depths_computed > 0:
            losses.add_loss("mtp", total_loss / depths_computed)
        return losses

    @torch.no_grad()
    def draft_next_tokens(self, hidden_state, first_token_id, embed_fn, head_fn):
        """Draft N additional tokens greedily using MTP modules.

        Used at inference for speculative decoding on the standard
        (token-level) path only. Each MTP depth takes the previous depth's
        hidden state and the last predicted token's embedding to produce a
        draft for the next position.

        Args:
            hidden_state: Hidden state at last position [batch, 1, hidden_size]
            first_token_id: Token predicted by main model [batch, 1]
            embed_fn: nn.Embedding for token embeddings
            head_fn: Head module for computing logits

        Returns:
            Tensor of drafted token IDs [batch, num_depths]
        """
        drafted = []
        h_prev = hidden_state
        prev_token = first_token_id

        for module in self.depths:
            token_embeds = embed_fn(prev_token)
            h_k = module(h_prev, token_embeds, attention_mask=None)
            logits = head_fn(h_k)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            drafted.append(next_token)
            h_prev = h_k
            prev_token = next_token

        if drafted:
            return torch.cat(drafted, dim=1)
        return torch.empty(
            hidden_state.size(0), 0, dtype=torch.long, device=hidden_state.device
        )
