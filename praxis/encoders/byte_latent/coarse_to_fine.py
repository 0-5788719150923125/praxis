"""Coarse-to-fine generation over byte-latent patches.

Next-Concept Modeling (arXiv:2610.14162) on the byte-latent stack, trained in one
loop. The local encoder's patch latents are cut into chunks of ``chunk`` patches,
and a multiscale residual quantizer describes each chunk coarse to fine: one code,
then two, four, ... one per patch. The trunk is the generator: it predicts every
level of every chunk in one forward, each level from the reconstruction of the
levels before it, with earlier chunks as clean context. The local decoder is the
tokenizer's decoder: it spells each patch's bytes from that patch's reconstructed
latent and the bytes of the same patch before them. No byte of an earlier patch
reaches it except through the codes, so when nested dropout cuts the ladder the
coarse levels are all it has (the paper's decoder reads codes alone).

The trunk reads detached latents and the codebooks learn by EMA, so the code loss
never reaches the tokenizer (the paper's frozen-tokenizer setting, in a single
training run). Its loss plus the decoder's is an upper bound on bits per byte,
because the codes are a fixed function of the bytes.

See next/next_concept_modeling.md.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.generation.utils import GenerateDecoderOnlyOutput

from praxis import registry
from praxis.attention.layout import BlockLayout
from praxis.attention.self_attention import SelfAttention
from praxis.encoders.quantization.multiscale import MultiScaleResidualVQ

from .encoder import ByteLatentEncoder
from .patcher import decoder_patch_ids_from_lengths, patch_ids_from_lengths

# Encodings that rotate by an explicit position (or ignore position). ALiBi biases
# by sequence-index distance and HoPE/ArcHoPE warp it, which a layout cannot steer.
POSITIONED_ENCODINGS = ("rope", "rope_learned_rates", "nope")


class CoarseToFineEncoder(ByteLatentEncoder):
    """Byte-latent encoder whose trunk generates multiscale patch codes."""

    # The decoder reads the latent of the patch its target byte sits in, so its
    # per-byte logits are reconstructions, not next-byte predictions.
    emits_next_token_likelihood = False

    def __init__(
        self,
        config,
        *,
        chunk: int = 32,
        codebook_size: int = 4096,
        nested_dropout: float = 0.5,
        corruption: float = 0.5,
        commitment: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        if self.nb_boe != 0:
            raise ValueError(
                "coarse-to-fine decoding aligns each byte with its own patch, which "
                "needs a patcher whose first patch is one byte (space or static)"
            )
        if self.byte_config.cross_attn_encoder:
            raise ValueError("coarse-to-fine patch latents come from pooling")

        dim = config.hidden_size
        self.dim = dim
        self.chunk = int(chunk)
        self.nested_dropout = float(nested_dropout)
        self.corruption = float(corruption)
        self.commitment = float(commitment)
        self.pad_token_id = getattr(config, "pad_token_id", None)

        self.ladder = MultiScaleResidualVQ(dim, self.chunk, codebook_size)
        levels = len(self.ladder.lengths)
        # The generator's own input and output layers. They live here because the
        # level blocks are this encoder's construction; only the code loss trains
        # them.
        self.level_inputs = nn.Linear(dim, dim)
        self.level_embeddings = nn.Parameter(torch.randn(levels, dim) * 0.02)
        self.code_classifiers = nn.ModuleList(
            nn.Linear(dim, dim) for _ in range(levels)
        )
        self.code_temperatures = nn.Parameter(torch.zeros(levels))
        # The decoder's own byte table: the shared embeddings hash n-grams across
        # patch boundaries. The extra row marks a patch's first byte.
        self.spell_start = self.input_vocab_size
        self.spelling = nn.Embedding(self.input_vocab_size + 1, self.input_dim)
        # A patch's slot in its chunk, so a coarse cut (one vector over many
        # patches) can still say different things about each.
        self.chunk_positions = nn.Parameter(torch.randn(self.chunk, dim) * 0.02)

        self.trunk_layout: Optional[BlockLayout] = None
        self._state: Dict[str, Any] = {}
        self._pending: Dict[str, Tensor] = {}
        self._level_stats: List[tuple] = []
        self._code_bits: Optional[Tensor] = None
        self._code_nats: Optional[Tensor] = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"architecture='{self.local_architecture}', "
            f"patching='{self.byte_config.patching_mode}', "
            f"chunk={self.chunk}, codebook_sizes={self.ladder.sizes})"
        )

    def check_trunk(self, config, trunk: nn.Module) -> None:
        """Refuse a trunk the published layout cannot steer. Called by the
        model once the trunk is built."""
        attention = registry.lookup("attention", config.attention_type)
        attention = getattr(attention, "func", attention)
        steerable = isinstance(attention, type) and issubclass(attention, SelfAttention)
        if not steerable or not any(
            isinstance(m, SelfAttention) for m in trunk.modules()
        ):
            raise ValueError(
                f"attention_type {config.attention_type!r} cannot read a published "
                "layout; coarse-to-fine needs a SelfAttention-family attention"
            )
        if config.encoding not in POSITIONED_ENCODINGS:
            raise ValueError(
                f"encoding {config.encoding!r} cannot take explicit positions; "
                f"use one of {POSITIONED_ENCODINGS}"
            )
        if getattr(config, "mtp_type", None) or getattr(config, "diffusion_type", None):
            raise ValueError("coarse-to-fine generation replaces next-token objectives")

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def _real_patches(self, input_ids: Tensor, patch_lengths: Tensor) -> Tensor:
        """[B, P] True where a patch holds at least one real input byte."""
        B, N = input_ids.shape
        ids = patch_ids_from_lengths(patch_lengths, N)
        real = torch.ones_like(input_ids, dtype=torch.long)
        if self.pad_token_id is not None:
            real = (input_ids != self.pad_token_id).long()
        count = torch.zeros(patch_lengths.shape, dtype=torch.long, device=ids.device)
        return count.scatter_add_(1, ids, real) > 0

    def _scored_bytes(self, input_ids: Tensor) -> Tensor:
        """Label positions holding a real byte (the trainer's bits-per-byte count)."""
        labels = input_ids[:, 1:]
        if self.pad_token_id is None:
            return torch.tensor(labels.numel(), device=labels.device)
        return (labels != self.pad_token_id).sum()

    def _corruption_rates(self) -> List[float]:
        """Replacement rate per conditioning level: linear, falling with depth,
        averaging ``corruption`` (the paper's "shifted linear decreasing")."""
        count = len(self.ladder.lengths) - 1
        if count <= 1:
            return [self.corruption] * count
        return [self.corruption * (1.5 - j / (count - 1)) for j in range(count)]

    def _corrupt(self, codes: List[Tensor], masks: List[Tensor], valid: Tensor):
        """Cumulative reconstructions from codes with some replaced at random."""
        noisy = []
        for k, rate in enumerate(self._corruption_rates()):
            level_codes = codes[k]
            swap = (
                torch.rand(level_codes.shape, device=level_codes.device) < rate
            ) & masks[k]
            random = torch.randint_like(level_codes, self.ladder.levels[k].K)
            noisy.append(torch.where(swap, random, level_codes))
        return self.ladder.reconstructions(noisy, valid)

    def _level_blocks(self, conditioning: List[Tensor], valid: Tensor) -> Tensor:
        """Generator inputs [B, n, S, D]: level k reads the reconstruction of the
        levels before it, resized to its own length."""
        B, n = valid.shape[:2]
        blocks = []
        for k, length in enumerate(self.ladder.lengths):
            if k == 0:
                prior = valid.new_zeros(
                    (B, n, 1, self.dim), dtype=self.level_embeddings.dtype
                )
            else:
                prior, _ = self.ladder.pool(conditioning[k - 1], valid, length)
            blocks.append(self.level_inputs(prior) + self.level_embeddings[k])
        return torch.cat(blocks, dim=2)

    def _block_pattern(self, device) -> tuple:
        """Per position of one chunk's level blocks: offset (span center, in
        patches) and level."""
        C = self.chunk
        offsets, levels = [], []
        for k, length in enumerate(self.ladder.lengths):
            span = C // length
            offsets.append(torch.arange(length, device=device) * span + (span - 1) / 2)
            levels.append(torch.full((length,), k, device=device))
        return torch.cat(offsets).float(), torch.cat(levels).long()

    def _layout(
        self, B: int, n: int, patch_docs: Optional[Tensor], device
    ) -> BlockLayout:
        """Context stream of ``n * chunk`` patches, then each chunk's level blocks."""
        C, K = self.chunk, len(self.ladder.lengths)
        P = n * C
        offsets, levels = self._block_pattern(device)
        S = offsets.numel()
        chunk_of = torch.arange(n, device=device).view(n, 1)
        positions = torch.cat(
            [
                torch.arange(P, device=device).float(),
                (chunk_of * C + offsets.view(1, S)).reshape(-1),
            ]
        )
        group = torch.cat(
            [
                torch.full((P,), -1, device=device, dtype=torch.long),
                (chunk_of * K + levels.view(1, S)).reshape(-1),
            ]
        )
        chunk = torch.cat(
            [
                torch.arange(P, device=device) // C,
                chunk_of.expand(n, S).reshape(-1),
            ]
        )
        context_docs = torch.zeros(B, P, dtype=torch.long, device=device)
        if patch_docs is not None:
            width = min(P, patch_docs.shape[1])
            context_docs[:, :width] = patch_docs[:, :width]
        # A block reads context from the document its chunk opens in.
        block_docs = context_docs[:, ::C].repeat_interleave(S, dim=1)
        doc = torch.cat([context_docs, block_docs], dim=1)
        return BlockLayout(positions=positions, group=group, chunk=chunk, doc=doc)

    def encode(self, input_ids: Tensor, block_ids: Optional[Tensor] = None):
        h, h_encoder, patch_lengths, patch_docs, aux_loss, decoder_tokens = (
            super().encode(input_ids, block_ids=block_ids)
        )
        B, P, D = h.shape
        C = self.chunk
        n = max(1, -(-P // C))
        valid = self._real_patches(input_ids, patch_lengths)
        z = F.rms_norm(h, (D,)) * valid.unsqueeze(-1)
        if n * C > P:
            z = F.pad(z, (0, 0, 0, n * C - P))
            valid = F.pad(valid, (0, n * C - P))
        chunks, chunk_valid = z.view(B, n, C, D), valid.view(B, n, C)
        # Codebooks learn only on real training steps. The lazy-init dummy
        # forward runs in train mode without grad on a row of identical bytes,
        # and seeding from it would start every code on the same vector.
        codes, masks, sums = self.ladder(
            chunks, chunk_valid, learn=torch.is_grad_enabled()
        )

        # The tokenizer: decode from the ladder, cut at a random level half the
        # time (nested dropout), with the gradient passed straight through to z.
        recon = sums[-1]
        if self.training and self.nested_dropout > 0 and len(sums) > 1:
            cut = torch.full((B,), len(sums) - 1, device=z.device)
            drop = torch.rand(B, device=z.device) < self.nested_dropout
            cut = torch.where(drop, torch.randint_like(cut, len(sums) - 1), cut)
            recon = torch.stack(sums)[cut, torch.arange(B, device=z.device)]
        recon = recon.reshape(B, n * C, D)
        weight = valid.unsqueeze(-1).to(z.dtype)
        commitment = ((z - recon) ** 2 * weight).sum() / (weight.sum() * D).clamp_min(
            1.0
        )
        latents = z + (recon - z).detach()

        # The generator: earlier chunks as clean context, then every level of
        # every chunk, each from (corrupted, while training) coarser levels.
        conditioning = sums
        if self.training and self.corruption > 0:
            conditioning = self._corrupt(codes, masks, chunk_valid)
        blocks = self._level_blocks(conditioning, chunk_valid)
        trunk_inputs = torch.cat([z.detach(), blocks.reshape(B, -1, D)], dim=1)
        self.trunk_layout = self._layout(B, n, patch_docs, z.device)

        self._state = dict(
            codes=codes,
            masks=masks,
            sums=sums,
            latents=latents,
            n=n,
            scored=self._scored_bytes(input_ids),
        )
        aux_loss = aux_loss + self.commitment * commitment
        # The trunk sequence carries its own mask, so no document ids go with it.
        return trunk_inputs, h_encoder, patch_lengths, None, aux_loss, decoder_tokens

    def code_logits(self, level: int, hidden: Tensor) -> Tensor:
        """Score every code of ``level`` by squared distance to a predicted point.

        Distances to the live codebook, so a code the EMA moves or a reset
        replaces is scored by its new vector rather than by a stale weight row.
        The point is predicted in units of the level's own code scale (residuals
        shrink with depth) and the score is divided by the mean code energy, so
        every level starts near uniform however unevenly its code norms spread.
        The head sharpens it by growing its point, the temperature by scaling.
        """
        codebook = self.ladder.levels[level].codebook.detach()
        energy = codebook.pow(2).sum(-1)
        mean_energy = energy.mean().clamp_min(1e-12)
        point = self.code_classifiers[level](hidden) * mean_energy.div(self.dim).sqrt()
        scale = self.code_temperatures[level].exp() / mean_energy
        return (2 * point @ codebook.t() - energy) * scale

    def decode(
        self,
        h: Tensor,
        h_encoder: Tensor,
        input_ids: Tensor,
        patch_lengths: Tensor,
        local_decoder_tokens: Tensor,
        block_ids: Optional[Tensor] = None,
    ):
        state = self._state
        B, n, C = h.shape[0], state["n"], self.chunk
        lengths = list(self.ladder.lengths)
        blocks = h[:, n * C :].reshape(B, n, sum(lengths), -1).split(lengths, dim=2)
        nats = h.new_zeros(())
        stats, level_nats = [], []
        for k, (out, codes, mask) in enumerate(
            zip(blocks, state["codes"], state["masks"])
        ):
            if not mask.any():
                stats.append(None)
                level_nats.append(h.new_zeros(()))
                continue
            logits = self.code_logits(k, out[mask]).float()
            target = codes[mask]
            ce = F.cross_entropy(logits, target, reduction="none")
            nats = nats + ce.sum()
            level_nats.append(ce.sum().detach())
            stats.append(
                (
                    ce.mean().detach(),
                    (logits.argmax(-1) == target).float().mean().detach(),
                )
            )
        scored = state["scored"].clamp_min(1)
        self._pending["code"] = nats / scored
        self._level_stats = stats
        self._code_nats = nats.detach()
        self._code_bits = (nats / scored).detach() / math.log(2.0)

        # Each byte position reads the latent of the patch its TARGET byte is in:
        # decoder_patch_ids index patch_lengths[:, 1:], so +1 is the target's patch.
        target_patch = (
            decoder_patch_ids_from_lengths(
                patch_lengths, self.nb_boe, local_decoder_tokens.shape[-1]
            )
            + 1
        )
        state.update(
            level_nats=level_nats,
            target_patch=target_patch,
            tokens=local_decoder_tokens,
        )
        return None, self._spell(local_decoder_tokens, target_patch, state["latents"])

    def _spell(
        self, tokens: Tensor, target_patch: Tensor, latents: Tensor, origin: int = 0
    ) -> Tensor:
        """Decoder features per position, read from its target's patch alone.

        Position ``t`` predicts byte ``t + 1`` and reads that byte's patch: its
        latent from ``latents``, its slot in its chunk, and the bytes of the
        patch before ``t + 1``. The conv layers stop at each patch's first
        position, whose own byte ends the previous patch and is read as the start
        marker instead. ``origin`` is the patch chunks are counted from.
        """
        B, N = tokens.shape
        index = torch.arange(N, device=tokens.device).expand(B, N)
        opens = F.pad(target_patch[:, 1:] != target_patch[:, :-1], (1, 0), value=True)
        positions = index - torch.cummax(index * opens, dim=1).values
        spelled = tokens.masked_fill(positions == 0, self.spell_start)
        slots = (target_patch - origin).remainder(self.chunk)
        aligned = latents.gather(
            1, target_patch.unsqueeze(-1).expand(-1, -1, latents.shape[-1])
        )
        return self.decoder(
            tokens=spelled,
            embeds=self.spelling(spelled),
            patch_embeds=aligned + self.chunk_positions[slots],
            cross_mask=None,
            mask=None,
            block_ids=None,
            segment_positions=positions,
        )

    @torch.no_grad()
    def ladder_cuts(self, classifier) -> List[tuple]:
        """Per cut ``k`` of the last forward's ladder: the decoder's logits from
        levels ``0..k`` alone, and the code nats of those levels. Logits plus
        nats bound bits per byte as the full ladder does. Read after an eval
        forward, where no level is cut or corrupted."""
        state = self._state
        if "level_nats" not in state:
            return []
        B, length, D = state["latents"].shape
        out, nats = [], 0.0
        for total, level in zip(state["sums"], state["level_nats"]):
            nats = nats + level
            features = self._spell(
                state["tokens"], state["target_patch"], total.reshape(B, length, D)
            )
            out.append((classifier(features), nats))
        return out

    def consume_pending_losses(self) -> Dict[str, Tensor]:
        out, self._pending = self._pending, {}
        return out

    def first_affected_position(self, input_ids: Tensor, position: int) -> int:
        """The earliest logit an edit to byte ``position`` of row 0 may move.

        A chunk's codes summarize all of its bytes, so every logit whose target
        is in the edited byte's chunk reads the edit; logits whose target is in
        an earlier chunk must not move. Patch boundaries up to ``position`` do
        not depend on that byte, so neither does its chunk's first byte.
        """
        lengths, _ = self.patcher.patch(input_ids[:1], include_next_token=True)
        starts = torch.cumsum(lengths[0], 0) - lengths[0]
        patch = int((starts <= position).sum()) - 1
        first_byte = int(starts[patch // self.chunk * self.chunk])
        return max(0, first_byte - 1)

    def code_nats(self) -> Optional[Tensor]:
        """Total code loss (nats) of the last forward, over every real code.

        Added to the decoder's reconstruction loss it bounds the batch's negative
        log-likelihood. Exact only in eval mode, where no code is corrupted and
        the ladder is never cut.
        """
        return self._code_nats

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------

    def training_metrics(self) -> dict:
        out = super().training_metrics()
        out.update(self.ladder.telemetry())
        for k, stat in enumerate(self._level_stats):
            if stat is None:
                continue
            loss, accuracy = float(stat[0]), float(stat[1])
            out[f"code_loss_l{k}"] = loss
            out[f"code_acc_l{k}"] = accuracy
            marginal = self.ladder.marginal_entropy(k)
            if marginal is not None:
                out[f"code_gain_l{k}"] = marginal - loss
        if self._code_bits is not None:
            out["code_bits_per_byte"] = float(self._code_bits)
        return out

    metric_descriptions = {
        **ByteLatentEncoder.metric_descriptions,
        "code_bits_per_byte": {
            "description": (
                "The generator's code loss in bits per byte: every code of every "
                "level, summed, over the bytes they describe. Plus the decoder's "
                "reconstruction bits it bounds bits per byte."
            ),
            "chart": {
                "title": "Code Bits per Byte",
                "y_label": "bits per byte",
                "y_scale": "linear",
                "group": "coarse_to_fine",
                "group_order": 73,
                "order": 0,
            },
        },
        **{
            f"code_loss_l{k}": {
                "description": (
                    f"Generator cross-entropy on level-{k} codes ({2**k} per chunk), "
                    "nats per code. Level 0 is one code per chunk."
                ),
                "chart": {
                    "title": "Code Loss by Level",
                    "y_label": "nats per code",
                    "y_scale": "linear",
                    "group": "coarse_to_fine",
                    "order": 1,
                    "series_group": "code_loss",
                    "series_label": f"level {k}",
                },
            }
            for k in range(8)
        },
        **{
            f"code_gain_l{k}": {
                "description": (
                    f"Level-{k} code loss below a predictor that knows only how often "
                    "each code is used (the entropy of the EMA usage). 0 = the trunk "
                    "has learned only the marginal; read this first."
                ),
                "chart": {
                    "title": "Code Gain over Marginal",
                    "y_label": "nats per code",
                    "y_scale": "linear",
                    "group": "coarse_to_fine",
                    "order": 2,
                    "series_group": "code_gain",
                    "series_label": f"level {k}",
                },
            }
            for k in range(8)
        },
        **{
            f"code_acc_l{k}": {
                "description": f"Share of level-{k} codes the generator names exactly.",
                "chart": {
                    "title": "Code Accuracy by Level",
                    "y_label": "accuracy",
                    "y_scale": "linear",
                    "group": "coarse_to_fine",
                    "order": 3,
                    "series_group": "code_acc",
                    "series_label": f"level {k}",
                },
            }
            for k in range(8)
        },
        **{
            f"vq_usage_entropy_l{k}": {
                "description": (
                    f"Level-{k} codebook utilization: entropy of the EMA code usage "
                    "over log(K). 1 = the whole bank in uniform use, 0 = collapsed."
                ),
                "chart": {
                    "title": "Codebook Utilization by Level",
                    "y_label": "entropy / log(K)",
                    "y_scale": "linear",
                    "group": "coarse_to_fine",
                    "order": 4,
                    "series_group": "vq_usage_entropy_level",
                    "series_label": f"level {k}",
                },
            }
            for k in range(8)
        },
        **{
            f"vq_dead_frac_l{k}": {
                "description": (
                    f"Level-{k} dead-code fraction: codes with EMA usage under one hit "
                    "per batch, the pool resets draw from."
                ),
                "chart": {
                    "title": "Dead Codes by Level",
                    "y_label": "fraction of K",
                    "y_scale": "linear",
                    "group": "coarse_to_fine",
                    "order": 5,
                    "series_group": "vq_dead_level",
                    "series_label": f"level {k}",
                },
            }
            for k in range(8)
        },
        **{
            f"vq_resets_l{k}": {
                "description": (
                    f"Level-{k} cumulative code resets. A steady slope = codes keep "
                    "changing meaning under the generator."
                ),
                "chart": {
                    "title": "Code Resets by Level (cumulative)",
                    "y_label": "codes reset",
                    "y_scale": "linear",
                    "group": "coarse_to_fine",
                    "order": 6,
                    "series_group": "vq_resets_level",
                    "series_label": f"level {k}",
                },
            }
            for k in range(8)
        },
    }

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    def decoding_method(self, generation_config=None):
        """Always coarse to fine: there is no next-byte distribution to sample
        until a chunk's codes exist."""
        return self.coarse_to_fine_decoding

    def _patches(self, tokens: Tensor) -> Tensor:
        """Patch lengths of ``tokens`` with the next-byte slot, cut as training
        cuts them. A prompt that ends mid-word leaves its last patch open, and
        the first generated bytes finish it: the decoder reads that patch's
        bytes so far, which a fresh patch would hide from it."""
        lengths, _ = self.patcher.patch(tokens, include_next_token=True)
        return lengths[:, lengths[0] > 0]

    def _patch_latents(self, tokens: Tensor, patch_lengths: Tensor) -> Tensor:
        """Normalized patch latents [1, P, D] for every patch of ``tokens``."""
        N = tokens.shape[1]
        patch_ids = patch_ids_from_lengths(patch_lengths, N)
        embeds = self.embeddings(tokens)
        (h_encoder, _), _ = self.encoder(
            tokens=tokens,
            embeds=embeds,
            patch_embeds=None,
            cross_mask=None,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
            mask=None,
            block_ids=None,
        )
        h = self._downsample(h_encoder, None, 1, patch_lengths, patch_ids)
        if self.token_proj is not None:
            h = self.token_proj(h)
        return F.rms_norm(h, (h.shape[-1],))

    def _reconstruct_context(self, z: Tensor) -> Tensor:
        """Tokenizer reconstructions [1, P, D] of existing patches, on the chunk
        grid a training row starting here would use."""
        P, C = z.shape[1], self.chunk
        n = max(1, -(-P // C))
        padded = F.pad(z, (0, 0, 0, n * C - P))
        valid = torch.zeros(1, n * C, dtype=torch.bool, device=z.device)
        valid[:, :P] = True
        _, _, sums = self.ladder(
            padded.view(1, n, C, -1), valid.view(1, n, C), learn=False
        )
        return sums[-1].reshape(1, n * C, -1)[:, :P]

    def _generation_layout(self, context: int, level: int, device) -> BlockLayout:
        length = self.ladder.lengths[level]
        span = self.chunk // length
        block = (
            context
            + torch.arange(length, device=device).float() * span
            + (span - 1) / 2
        )
        positions = torch.cat([torch.arange(context, device=device).float(), block])
        group = torch.cat(
            [
                torch.full((context,), -1, dtype=torch.long, device=device),
                torch.zeros(length, dtype=torch.long, device=device),
            ]
        )
        chunk = torch.cat(
            [
                torch.zeros(context, dtype=torch.long, device=device),
                torch.ones(length, dtype=torch.long, device=device),
            ]
        )
        doc = torch.zeros(1, context + length, dtype=torch.long, device=device)
        return BlockLayout(positions=positions, group=group, chunk=chunk, doc=doc)

    def _sample_chunk(
        self, trunk, context: Tensor, do_sample: bool, temperature: float
    ):
        """One chunk's reconstruction [1, chunk, D], level by level: ``levels``
        trunk passes over the context plus the current level's block."""
        C, D = self.chunk, self.dim
        P = context.shape[1]
        valid = torch.ones(1, 1, C, dtype=torch.bool, device=context.device)
        recon = context.new_zeros(1, 1, C, D)
        for k, length in enumerate(self.ladder.lengths):
            if k == 0:
                prior = context.new_zeros(1, 1, 1, D)
            else:
                prior, _ = self.ladder.pool(recon, valid, length)
            block = (self.level_inputs(prior) + self.level_embeddings[k]).view(
                1, length, D
            )
            SelfAttention.set_layout(
                trunk["root"], self._generation_layout(P, k, context.device)
            )
            try:
                hidden = trunk["forward"](torch.cat([context, block], dim=1))
            finally:
                SelfAttention.set_layout(trunk["root"], None)
            logits = self.code_logits(k, hidden[:, P:]).float()[0]
            if do_sample:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                codes = torch.multinomial(probs, 1).squeeze(-1)
            else:
                codes = logits.argmax(dim=-1)
            vectors = self.ladder.lookup(k, codes).view(1, 1, length, D)
            recon = recon + self.ladder.spread(vectors, length)
        return recon.view(1, C, D)

    def _next_byte_logits(
        self,
        model,
        tokens: Tensor,
        patch_lengths: Tensor,
        table: Tensor,
        origin: int = 0,
    ):
        """Logits for the byte after ``tokens``, from the latent of its patch in
        ``table`` and that patch's bytes so far (all ``_spell`` reads)."""
        N = tokens.shape[1]
        target_patch = patch_ids_from_lengths(patch_lengths, N + 1)[:, 1:]
        first = int((target_patch[0] < target_patch[0, -1]).sum())
        features = self._spell(
            tokens[:, first:], target_patch[:, first:], table, origin
        )
        return model.classifier(features[:, -1:])[:, -1]

    def _generate_row(
        self, model, prompt, logits_processor, criteria, config, streamer
    ):
        from praxis.inference.decoding import (
            is_halted,
            pick_next,
            stream_put,
            trunk_hooks,
        )

        trunk = {"root": model.decoder, "forward": trunk_hooks(model)["latent_forward"]}
        do_sample = bool(getattr(config, "do_sample", False))
        temperature = float(getattr(config, "temperature", None) or 1.0)
        seq = prompt
        lengths = self._patches(seq)
        start = lengths.shape[1] - 1  # the patch the next byte is in
        origin = start  # generated chunks count from here
        table = self._reconstruct_context(self._patch_latents(seq, lengths)[:, :start])
        while True:
            context = self._patch_latents(seq, lengths)[:, :start]
            table = torch.cat(
                [
                    table[:, :start],
                    self._sample_chunk(trunk, context, do_sample, temperature),
                ],
                dim=1,
            )
            while True:
                logits = self._next_byte_logits(model, seq, lengths, table, origin)
                byte = pick_next(logits, seq, logits_processor, do_sample)
                seq = torch.cat([seq, byte.view(1, 1).to(seq.dtype)], dim=1)
                stream_put(streamer, byte.view(1))
                if is_halted(seq, criteria):
                    return seq
                lengths = self._patches(seq)
                if lengths.shape[1] - 1 >= start + self.chunk:
                    break
            start += self.chunk

    @torch.no_grad()
    def coarse_to_fine_decoding(
        self,
        model,
        input_ids: Optional[Tensor] = None,
        logits_processor=None,
        stopping_criteria=None,
        generation_config=None,
        tokenizer: Any = None,
        streamer: Any = None,
        **model_kwargs: Any,
    ):
        """A transformers decoding method (see ``BaseEncoder.decoding_method``).

        Per chunk: ``levels`` trunk passes sample its codes coarse to fine, then
        the local decoder spells its bytes one at a time from the reconstructed
        latents, through the prepared processors and stopping criteria. A
        prompt that ends mid-word is continued inside its open patch.
        Rows are generated one at a time and right-padded to a common length.
        """
        from praxis.inference.decoding import stream_end

        if input_ids is None or input_ids.shape[1] == 0:
            raise ValueError("coarse-to-fine generation needs a non-empty prompt")
        rows = [
            self._generate_row(
                model,
                input_ids[b : b + 1],
                logits_processor,
                stopping_criteria,
                generation_config,
                streamer if input_ids.shape[0] == 1 else None,
            )
            for b in range(input_ids.shape[0])
        ]
        width = max(row.shape[1] for row in rows)
        fill = self.pad_token_id if self.pad_token_id is not None else 0
        sequences = torch.cat(
            [F.pad(row, (0, width - row.shape[1]), value=fill) for row in rows]
        )
        stream_end(streamer)
        if bool(getattr(generation_config, "return_dict_in_generate", False)):
            return GenerateDecoderOnlyOutput(sequences=sequences)
        return sequences
