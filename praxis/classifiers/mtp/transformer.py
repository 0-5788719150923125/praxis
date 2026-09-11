"""Transformer-based MTP module.

Uses a full transformer block per depth — same as the original MTP implementation.
"""

import torch
import torch.nn as nn

from praxis import registry


class TransformerMTPModule(nn.Module):
    """Single MTP depth module using a transformer block.

    Takes hidden states from the previous depth and ground-truth position
    embeddings, normalizes both, concatenates, projects back to hidden_size,
    and runs through a transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.norm_hidden = registry.lookup("normalization", config.norm_type)(
            config.hidden_size, eps=config.epsilon
        )
        self.norm_embed = registry.lookup("normalization", config.norm_type)(
            config.embed_size, eps=config.epsilon
        )
        self.projection = nn.Linear(
            config.hidden_size + config.embed_size, config.hidden_size, bias=False
        )
        self.block = registry.lookup("blocks", config.block_type)(config)

    def forward(self, hidden_states, token_embeds, attention_mask):
        h = self.norm_hidden(hidden_states, mode="direct")
        e = self.norm_embed(token_embeds, mode="direct")
        combined = torch.cat([h, e], dim=-1)
        projected = self.projection(combined)
        output, _, _, _ = self.block(projected, attention_mask)
        return output
