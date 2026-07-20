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

import copy
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

# Speculative width control. Fixed, model-agnostic constants: the width itself
# is learned from the run's own accepted-run lengths (see MultiTokenPrediction).
# Decay is a slow EMA so a single unlucky commit cannot collapse the window;
# the margin is how far above the current run we keep probing, so a model whose
# drafts improve can widen again without any external signal.
_ACCEPT_EMA_DECAY: float = 0.9
_ACCEPT_WIDTH_MARGIN: int = 2


def _is_byte_latent(encoder_type) -> bool:
    """True for a byte-latent (BLT/Abstractinator) encoder. Lazy import: the
    encoders package pulls in blocks/heads, so a module-level import would
    cycle."""
    if not encoder_type:
        return False
    from praxis.encoders import is_byte_latent_encoder

    return is_byte_latent_encoder(encoder_type)


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

        # Byte-level MTP for byte-latent encoders: predict future BYTES (CE)
        # rather than future patch representations (MSE). This is the objective
        # that lets the depth modules DRAFT bytes for speculative decoding, the
        # inference speedup that makes byte-latent generation not print one byte
        # per full forward. A byte-latent model's shared head classifies the
        # byte-level decoder hidden (dim = embed_size = dim_token_emb), so the
        # depths run in embed_size space, not the trunk's hidden_size.
        self.byte_level = self.encoder_path and _is_byte_latent(config.encoder_type)

        depth_config = config
        if self.byte_level:
            # Depths operate on the byte head's input width. Build them from a
            # config view whose hidden_size == embed_size so the module's block,
            # norms, and projection all size to the byte space.
            depth_config = copy.copy(config)
            depth_config.hidden_size = config.embed_size

        # vear: one shared pool of light harmonic experts, sliding-window merged
        # per depth (praxis/heads/mtp/vear.py). Other types: K independent
        # per-depth modules from the registry.
        self.is_vear = config.mtp_type == "vear"
        if self.is_vear:
            from praxis.heads.mtp.vear import VearHarmonicMTPBank

            self.bank = VearHarmonicMTPBank(depth_config, self.num_depths)
            self.depths = None
        else:
            module_cls = MTP_REGISTRY[config.mtp_type]
            self.depths = nn.ModuleList(
                [module_cls(depth_config) for _ in range(self.num_depths)]
            )

        self._draft_accs: list = []

        # Speculative width adapts to the accepted-run length this model
        # actually delivers. Acceptance stops at the FIRST divergence, so every
        # drafted/verified candidate past that point is pure waste - and on the
        # byte-latent path the verify is a batch of one row PER candidate, so
        # that waste is linear in the width. Start optimistic (full depth) and
        # let the observed runs pull it down; the margin keeps a probe above the
        # current run so the width can climb back as the drafts improve.
        self._accept_ema: float = float(self.num_depths)
        self._accept_seen: int = 0

        # Non-byte encoder path (e.g. CALM) owns a projection + head for
        # patch-level MTP; byte-level and token paths reuse the model's head.
        if self.encoder_path and not self.byte_level:
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

    @property
    def draft_width(self) -> int:
        """How many drafts to actually produce per speculative step.

        Acceptance stops at the first divergence, so a step that commits a run
        of ``r`` bytes never uses candidate ``r+1`` onward - it only pays for
        them, in a sequential draft per depth and (byte-latent) one verify row
        per candidate. Width therefore tracks the measured run length rather
        than ``num_depths``: the useful window, not the trained one. Bounded by
        ``num_depths``, and never below 1 so drafting cannot switch itself off.
        """
        import math

        width = math.ceil(self._accept_ema) + _ACCEPT_WIDTH_MARGIN
        return max(1, min(self.num_depths, int(width)))

    def note_accepted(self, run_length: int) -> None:
        """Record one speculative commit's accepted-run length (EMA input)."""
        run = max(0, int(run_length))
        self._accept_ema = (
            _ACCEPT_EMA_DECAY * self._accept_ema + (1.0 - _ACCEPT_EMA_DECAY) * run
        )
        self._accept_seen += 1

    def prepare_inputs(
        self,
        hidden_states,
        input_ids,
        attention_mask,
        embed_fn,
        head,
        patch_embeds=None,
    ):
        """Build path-appropriate MTPInputs for the current execution path.


        Args:
            hidden_states: Decoder output [batch, seq, hidden_size]
            input_ids: Original input token IDs [batch, seq]
            attention_mask: Attention mask or None
            embed_fn: Input embedding function (nn.Embedding)
            head: LM head module (token path) — ignored on encoder path
            patch_embeds: Patch embeddings from encoder (encoder path only)
        """
        if self.byte_level:
            # Byte-latent: h_0 is the byte-level decoder hidden (embed_size),
            # position embeds are byte embeddings, targets are the byte IDs, and
            # the shared byte head classifies each drafted hidden. Same shape
            # contract as the token path, so the forward loop is unchanged.
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
        elif self.encoder_path:
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

    def _run_depth(self, k: int, h, e, mask):
        """Execute depth ``k``: a shared sliding-window merge (vear) or the k-th
        independent per-depth module (transformer/conv)."""
        if self.is_vear:
            return self.bank(h, e, mask, depth=k)
        return self.depths[k](h, e, mask)

    def training_metrics(self) -> dict:
        """Multi-token eligibility (per-depth draft accuracy) + vear harmonic
        field diagnostics."""
        out: dict = {}
        accs = getattr(self, "_draft_accs", []) or []
        for k, a in enumerate(accs):
            out[f"mtp_draft_acc_d{k}"] = a
        if accs:
            out["mtp_draft_acc"] = sum(accs) / len(accs)
        if self.is_vear:
            out.update(self.bank.training_metrics())
        return out

    def dashboard_snapshots(self) -> dict:
        return self.bank.dashboard_snapshots() if self.is_vear else {}

    def _draft_acc_descriptions(self) -> dict:
        """Per-depth draft-accuracy chart hints (one shared chart via
        series_group), plus the mean. Byte-latent MTP only."""
        if not self.byte_level:
            return {}
        out: dict = {
            f"mtp_draft_acc_d{k}": {
                "description": (
                    f"Fraction of positions where MTP depth {k} (drafting the "
                    f"byte at offset {k + 2}) predicts correctly. This depth's "
                    "speculative accept-rate ceiling; the profile across depths "
                    "is the expected accepted-run length, i.e. the achievable "
                    "multi-token speedup. NOTE: actual accept-rate <= this, since "
                    "byte-latent's non-causal patching can still reject a good "
                    "draft at verify time."
                ),
                "chart": {
                    "title": "MTP Draft Accuracy (per depth)",
                    "y_label": "draft accuracy",
                    "y_scale": "linear",
                    "group": "mtp_field",
                    "group_order": 45,
                    "order": 50,
                    "series_group": "mtp_draft_acc",
                    "series_label": f"depth {k} (+{k + 2})",
                },
            }
            for k in range(self.num_depths)
        }
        out["mtp_draft_acc"] = {
            "description": (
                "Mean per-depth draft accuracy - the headline multi-token "
                "inference eligibility number. Rising = the K drafts are getting "
                "predictable, so more bytes land per speculative step."
            ),
            "chart": {
                "title": "MTP Draft Accuracy (mean)",
                "y_label": "mean draft accuracy",
                "y_scale": "linear",
                "group": "mtp_field",
                "group_order": 45,
                "order": 45,
            },
        }
        return out

    def field_metric_descriptions(self) -> dict:
        """Chart hints for the MTP diagnostics (draft accuracy always for
        byte-latent; harmonic-field metrics for vear), for the descriptions
        walker."""
        out = dict(self._draft_acc_descriptions())
        if self.is_vear:
            from praxis.heads.mtp.vear import VearHarmonicMTPBank

            out.update(VearHarmonicMTPBank.metric_descriptions)
        return out

    def forward(self, inputs: MTPInputs):
        total_loss = 0.0
        h_prev = inputs.hidden_states
        depths_computed = 0
        # Per-depth draft accuracy: how often the depth-k head predicts the true
        # byte at offset k+2. The multi-token-inference eligibility signal - each
        # depth's accuracy is (an upper bound on) its speculative accept rate, so
        # the profile across depths is the expected accepted-run length -> the
        # achievable speedup. Discrete (CE) paths only; MSE has no argmax.
        discrete = not self.encoder_path or self.byte_level
        draft_accs: list = []

        for k in range(self.num_depths):
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

            # Run through the depth transform
            h_k = self._run_depth(k, h_trimmed, position_embeds, mask)

            # Predictions via head, trimmed for alignment
            preds = inputs.head(h_k)[:, :-1]

            # Targets at offset k+2
            targets = inputs.targets[:, offset + 1 :]

            # Align predictions and targets
            min_out = min(preds.size(1), targets.size(1))
            preds = preds[:, :min_out].contiguous()
            targets = targets[:, :min_out].contiguous()

            total_loss = total_loss + inputs.loss_fn(preds, targets)
            if discrete:
                with torch.no_grad():
                    draft_accs.append(
                        float((preds.argmax(-1) == targets).float().mean().item())
                    )

            # Chain: this depth's output becomes input for next depth
            h_prev = h_k
            depths_computed += 1

        self._draft_accs = draft_accs
        losses = LossContainer()
        if depths_computed > 0:
            losses.add_loss("mtp", total_loss / depths_computed)
            # Keep the vear pool's harmonic geometries distinct (training-only).
            if self.is_vear and self.training:
                losses.add_loss("mtp_vear_repulsion", self.bank.repulsion_loss())
        return losses

    @torch.no_grad()
    def draft_next_tokens(
        self, hidden_state, first_token_id, embed_fn, head_fn, max_depths=None
    ):
        """Draft N additional tokens greedily using MTP modules.

        Used at inference for speculative decoding. Each MTP depth takes the
        previous depth's hidden state and the last predicted token's embedding
        to produce a draft for the next position. Depths run sequentially, each
        paying a head evaluation, so ``max_depths`` (default: ``draft_width``)
        bounds the work to the window acceptance can actually use.

        Args:
            hidden_state: Hidden state at last position [batch, 1, hidden_size]
            first_token_id: Token predicted by main model [batch, 1]
            embed_fn: nn.Embedding for token embeddings
            head_fn: Head module for computing logits
            max_depths: Depths to run; ``None`` uses the adaptive draft width

        Returns:
            Tensor of drafted token IDs [batch, min(max_depths, num_depths)]
        """
        limit = self.draft_width if max_depths is None else int(max_depths)
        limit = max(0, min(self.num_depths, limit))

        drafted = []
        h_prev = hidden_state
        prev_token = first_token_id

        for k in range(limit):
            token_embeds = embed_fn(prev_token)
            h_k = self._run_depth(k, h_prev, token_embeds, None)
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
