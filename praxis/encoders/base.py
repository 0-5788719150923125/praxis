"""Base class for Praxis encoders.

Encoders sit in the model's input slot: ``encode`` turns token ids into the
patch/latent sequence the global transformer consumes (plus an auxiliary
loss), and ``decode`` turns the transformer's hidden states back into
features (or logits). The optional hooks below let an encoder opt into
behaviors the model checks for - owning its loss, aligning its outputs,
naming an input-embedding profile, or driving its own generation loop -
without the model needing to know the concrete type.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Shared contract for encoders plugged into the model's input slot."""

    # Input-embedding profile key (resolved against EMBEDDING_REGISTRY by the
    # model). None means the encoder owns its embeddings (e.g. CALM).
    embedding_profile: Optional[str] = None

    @abstractmethod
    def encode(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.LongTensor] = None
    ):
        """Return (patch_embeds, h_encoder, patch_lengths, block_ids,
        encoder_loss, local_decoder_tokens)."""

    @abstractmethod
    def decode(
        self,
        h: torch.Tensor,
        h_encoder: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        local_decoder_tokens: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Return (logits_or_None, embeds) from global hidden states."""

    # ------------------------------------------------------------------
    # Optional hooks; defaults suit a standard CE encoder (e.g. byte-latent).
    # ------------------------------------------------------------------

    @property
    def handles_loss(self) -> bool:
        """If True, the encoder registers its own losses; the model skips CE."""
        return False

    @property
    def outputs_are_aligned(self) -> bool:
        """If True, decode logits are already aligned (no label shift)."""
        return False

    @property
    def classifier(self) -> Optional[nn.Module]:
        """Classifier used by cut-cross-entropy paths, if the encoder owns one."""
        return None

    @property
    def sequence_length_multiplier(self) -> int:
        """Factor to scale the user-supplied sequence length by (8 for byte)."""
        return 1

    def consume_pending_losses(self) -> Dict[str, torch.Tensor]:
        """Pop side-channel losses registered during the last decode()."""
        return {}

    def custom_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        base_forward: Callable[[torch.Tensor], object],
        generation_config=None,
        **kwargs,
    ):
        """Encoder-owned generation loop. Return None to defer to the standard
        HF generate path. ``base_forward(input_ids)`` runs the global
        transformer and returns an output exposing ``last_hidden_state``.
        """
        return None
