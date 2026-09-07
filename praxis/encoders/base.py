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
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Shared contract for encoders plugged into the model's input slot."""

    # Input-embedding profile key (resolved against EMBEDDING_REGISTRY by the
    # model). None means the encoder owns its embeddings (e.g. CALM).
    embedding_profile: Optional[str] = None

    # True if the encoder builds its own input embeddings sized to the
    # tokenizer's true vocab (e.g. CALM's 264-wide lm_tok_emb), rather than an
    # external hash embedding whose width is the overloaded config.vocab_size.
    # Declarative (class-level) so config processing can read it without an
    # instance, to unify vocab_size onto the real representational width.
    owns_embeddings: bool = False

    # Decoding paths this encoder can DRIVE through custom_generate. Empty
    # means it drives none and generation falls through to the model's own
    # loop. An encoder with exactly one mode (CALM, which only ever decodes by
    # vote) declares it here and the user never has to name it in a config.
    generation_modes: Tuple[str, ...] = ()

    # Which of those to use when the run does not ask for one. None falls back
    # to the first entry.
    default_generation_mode: Optional[str] = None

    # Resolved once by the model at build time. "standard" (or None) means this
    # encoder is not driving generation.
    generation_mode: Optional[str] = None

    def resolve_generation_mode(self, requested: Optional[str] = None) -> Optional[str]:
        """Settle the run's decoding path against what this encoder offers.

        Inference-only, so it is deliberately NOT part of the model hash: both
        paths are trained by the same objectives, and making the choice
        architectural would force a separate training run just to compare
        decoders on one checkpoint.
        """
        modes = tuple(self.generation_modes)
        if not modes:
            if requested and requested != "standard":
                raise ValueError(
                    f"{type(self).__name__} cannot drive generation_mode="
                    f"{requested!r}; it has no custom generation path."
                )
            return "standard"
        if requested is None:
            return self.default_generation_mode or modes[0]
        if requested not in modes:
            raise ValueError(
                f"{type(self).__name__} supports generation_mode "
                f"{modes}, got {requested!r}."
            )
        return requested

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

    def info_overrides(self) -> Dict[str, Any]:
        """Amend the model-info panel that both dashboards (CLI + web) render.

        Returns a mapping from a ``model_info`` key to a replacement value, or
        to ``None`` to drop that field entirely. This lets an encoder correct
        or hide a stat that doesn't describe it - e.g. CALM packs K token
        embeddings into one ``hidden_size`` patch vector, so a lone
        ``embed_size`` is misleading and CALM removes it. Applied once at the
        shared source (``build_model_info``), so both surfaces stay in sync.
        Default: no changes."""
        return {}

    # ------------------------------------------------------------------
    # Optional self-supervised pretraining phase (e.g. CALM's autoencoder
    # warmup). Defaults make this a no-op, so standard encoders are
    # unaffected. While in_pretraining() is True the model locks everything
    # except pretraining_parameters() and trains only pretraining_loss().
    # ------------------------------------------------------------------

    def in_pretraining(self) -> bool:
        """True while the encoder still needs its isolated pretraining phase.
        The rest of the model stays locked until this returns False."""
        return False

    def training_stage(self) -> Optional[str]:
        """Semantic label for the current training stage, surfaced to the
        dashboard (e.g. "preflight" while the encoder pretrains, "pretrain"
        once standard LM training is underway). None lets the caller fall back
        to its default label. Generic on purpose: an encoder with more discrete
        phases can name each one here."""
        return None

    def pretraining_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Self-supervised objective for the pretraining phase. Only called
        while ``in_pretraining()`` is True; the global transformer is skipped."""
        raise NotImplementedError

    def pretraining_parameters(self):
        """Parameters that remain trainable during the pretraining phase.
        Everything else in the model is locked. Defaults to none."""
        return ()

    def freeze_after_pretraining(self) -> None:
        """Fired once when the pretraining phase ends (e.g. freeze the codec).
        The rest of the model is unlocked immediately afterward."""
        return None

    def custom_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        base_forward: Callable[[torch.Tensor], object],
        generation_config=None,
        **kwargs,
    ):
        """Encoder-owned generation loop. Return None to defer to the standard
        HF generate path.

        ``base_forward(input_ids)`` runs the global transformer from tokens and
        returns an output exposing ``last_hidden_state`` (plus ``patch_embeds``,
        ``h_encoder``, ``patch_lengths`` and ``local_decoder_tokens`` for
        encoders that patch).

        ``latent_forward(patch_embeds, positions=None)`` arrives in ``kwargs``
        and runs the trunk directly on a latent sequence, returning its hidden
        states. An encoder that autoregresses over PATCHES needs it: the patch
        it just predicted has no bytes behind it, so ``base_forward`` cannot
        reach it.
        """
        return None
