"""Convolutional MTP module.

Lightweight alternative to the transformer-based MTP module. Replaces the full
transformer block with stacked dilated causal convolutions, mirroring the
ConvBlock pattern from praxis/encoders/byte_latent/encoder.py.
"""

import torch
import torch.nn as nn

from praxis.normalization import NORMALIZATION_REGISTRY


class CausalConvLayer(nn.Module):
    """Single causal conv layer: RMSNorm -> Conv1d (causal, dilated) -> SiLU -> residual."""

    def __init__(self, dim, kernel_size=3, dilation=1, eps=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, dilation=dilation, padding=self.padding
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # x: [batch, seq_len, dim]
        residual = x
        h = self.norm(x)
        h = h.transpose(1, 2)  # [batch, dim, seq_len]
        h = self.conv(h)
        h = h[..., : -self.padding] if self.padding > 0 else h
        h = h.transpose(1, 2)  # [batch, seq_len, dim]
        return self.activation(h) + residual


class ConvMTPModule(nn.Module):
    """Single MTP depth module using causal convolutions.

    Same front-end as TransformerMTPModule (norm + concat + project), but
    replaces the transformer block with 2 stacked dilated causal conv layers
    (dilation 1 and 2, kernel_size=3) for a ~7-position receptive field.
    """

    def __init__(self, config):
        super().__init__()
        self.norm_hidden = NORMALIZATION_REGISTRY[config.norm_type](
            config.hidden_size, eps=config.epsilon
        )
        self.norm_embed = NORMALIZATION_REGISTRY[config.norm_type](
            config.embed_size, eps=config.epsilon
        )
        self.projection = nn.Linear(
            config.hidden_size + config.embed_size, config.hidden_size, bias=False
        )
        self.convs = nn.Sequential(
            CausalConvLayer(config.hidden_size, kernel_size=3, dilation=1, eps=config.epsilon),
            CausalConvLayer(config.hidden_size, kernel_size=3, dilation=2, eps=config.epsilon),
        )

    def forward(self, hidden_states, token_embeds, attention_mask):
        h = self.norm_hidden(hidden_states, mode="direct")
        e = self.norm_embed(token_embeds, mode="direct")
        combined = torch.cat([h, e], dim=-1)
        projected = self.projection(combined)
        return self.convs(projected)
