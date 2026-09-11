"""
AbstractinatorEncoder: BLT encoder with a residual vector quantization bottleneck.

Subclasses ByteLatentEncoder and overrides only the _downsample and
_post_downsample hooks — no encode()/decode() duplication.

Codebook size defaults to config.vocab_size. Other defaults aligned with
the abstractinator project (beta=0.1, decay=0.999, EMA enabled, dead-code
resets every 250 steps).

The bottleneck's coordinate frame is selectable: the default quantizes raw
patch features, while the "harmonic" variants quantize amplitudes in the CALM
standing-wave basis (see praxis/encoders/quantization/harmonic_bottleneck.py).

``next_code`` gives the trunk an objective of its own, as the reference's top
model has: its output at patch ``p`` classifies patch ``p+1``'s code at every
residual stage. Without it the trunk learns only through the local decoder.

Based on: https://github.com/OilProducts/abstractinator
"""

import math
from typing import Dict, Optional, TypeVar

import torch
import torch.nn as nn

from praxis.encoders.byte_latent.encoder import ByteLatentEncoder
from praxis.encoders.quantization import (
    HarmonicResidualVQ,
    LearnedQueryAttention,
    MultiStageResidualVQ,
)
from praxis.classifiers.halo import HaloGeometry
from praxis.losses.halo import HALOLoss

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

NEXT_CODE_OBJECTIVES = (None, "halo")


class AbstractinatorEncoder(ByteLatentEncoder):
    """
    BLT encoder with a multi-stage residual VQ bottleneck between the local
    encoder and the global transformer.

    After downsampling byte-level representations to patch-level and projecting
    to hidden_size, vectors are quantized through a residual VQ. The VQ loss
    is added to the existing encoder aux_loss, requiring no changes to the
    training loop.
    """

    def __init__(
        self,
        config: ConfigType,
        *,
        # All ByteLatentEncoder kwargs
        patching_mode: str = "space",
        patching_threshold: float = 3.141592653589793,
        patch_size: int = 6,
        target_compression_ratio: float = 0.125,
        local_architecture: str = "conv",
        n_layers_encoder: int = 3,
        n_layers_decoder: int = 3,
        embeddings: str = "byte_hash",
        entropy_model_layers: int = 2,
        cross_attn_encoder: bool = False,
        cross_attn_decoder: bool = False,
        merge: str = "add",
        downsampling_method: str = "max",
        # Bottleneck coordinate frame: "rvq" quantizes raw patch features (the
        # abstractinator default); "harmonic" rotates them into the CALM
        # standing-wave basis first and quantizes the amplitudes;
        # "harmonic_serpent" adds a learned periodic nonlinearity to that
        # analysis transform. bottleneck_ratio sets the harmonic frame's
        # spectral budget as a fraction of hidden_size (lossy when < 1.0).
        bottleneck: str = "rvq",
        bottleneck_ratio: float = 0.5,
        # RVQ kwargs (defaults from abstractinator)
        vq_codebook_size: Optional[int] = None,
        vq_depth: int = 2,
        vq_beta: float = 0.1,
        vq_ema: bool = True,
        vq_decay: float = 0.99,
        vq_reset_codes: bool = True,
        vq_reset_interval: int = 250,
        vq_max_reset_pct: float = 0.1,
        vq_stale_after: int = 2000,
        # Learned query pooling
        use_learned_queries: bool = False,
        num_queries_per_segment: int = 1,
        # Trunk objective on the next patch's codes: None, or "halo".
        next_code: Optional[str] = None,
    ) -> None:
        super().__init__(
            config,
            patching_mode=patching_mode,
            patching_threshold=patching_threshold,
            patch_size=patch_size,
            target_compression_ratio=target_compression_ratio,
            local_architecture=local_architecture,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            embeddings=embeddings,
            entropy_model_layers=entropy_model_layers,
            cross_attn_encoder=cross_attn_encoder,
            cross_attn_decoder=cross_attn_decoder,
            merge=merge,
            downsampling_method=downsampling_method,
        )

        D = config.hidden_size
        # Bank size, most explicit source first: `config.codebook_size` (the
        # --codebook-size flag, set in the experiment file), then the encoder
        # profile's own constant, then vocab_size. The flag outranks the profile
        # for the same reason `config.embeddings` outranks the encoder's
        # embedding profile - a number written in the run's config should not be
        # overridden by a default written in a registry partial. The vocab_size
        # fallback is a coincidence of scale rather than a rule: this bank
        # indexes patch latents, not tokens.
        K = getattr(config, "codebook_size", None)
        if K is None:
            K = vq_codebook_size
        if K is None:
            K = config.vocab_size

        vq_kwargs = dict(
            K=K,
            depth=vq_depth,
            beta=vq_beta,
            ema=vq_ema,
            decay=vq_decay,
            reset_codes=vq_reset_codes,
            reset_interval=vq_reset_interval,
            max_codes_to_reset_pct=vq_max_reset_pct,
            stale_after=vq_stale_after,
        )
        if bottleneck == "rvq":
            self.quantizer = MultiStageResidualVQ(D=D, **vq_kwargs)
        elif bottleneck in ("harmonic", "harmonic_serpent", "harmonic_gdn"):
            self.quantizer = HarmonicResidualVQ(
                dim=D,
                latent_dim=max(1, int(D * bottleneck_ratio)),
                nonlinear=(bottleneck == "harmonic_serpent"),
                normalization=("gdn" if bottleneck == "harmonic_gdn" else "rms"),
                **vq_kwargs,
            )
        else:
            raise ValueError(f"Unknown abstractinator bottleneck: {bottleneck!r}")

        # Scored with HALO: per stage, a projection of the normalized trunk
        # output feeds a HaloGeometry over that stage's K codes. The projection
        # starts on the unit per-coordinate scale HALO's calibration assumes,
        # and is left unnormalized after that, as in the reference.
        if next_code not in NEXT_CODE_OBJECTIVES:
            raise ValueError(
                f"Unknown next_code objective {next_code!r}; "
                f"known: {list(NEXT_CODE_OBJECTIVES)}"
            )
        self.next_code = next_code
        self._pending: Dict[str, torch.Tensor] = {}
        self._next_code_diag: Dict[str, torch.Tensor] = {}
        if next_code == "halo":
            core = getattr(self.quantizer, "quantizer", self.quantizer)
            depth, codes = int(getattr(core, "depth", 1)), int(core.K)
            self.next_code_norm = nn.LayerNorm(D)
            self.next_code_proj = nn.ModuleList(
                [nn.Linear(D, D, bias=False) for _ in range(depth)]
            )
            for proj in self.next_code_proj:
                nn.init.normal_(proj.weight, std=D**-0.5)
            self.next_code_classifiers = nn.ModuleList(
                [HaloGeometry(D, codes) for _ in range(depth)]
            )
            self.next_code_loss = HALOLoss(vocab_size=codes, learn_gamma=False)

        # Optional learned query pooling
        self.use_learned_queries = use_learned_queries
        if use_learned_queries:
            max_queries = config.max_position_embeddings // patch_size
            # Ensure max_queries is a multiple of num_queries_per_segment
            max_queries = (
                max_queries // num_queries_per_segment
            ) * num_queries_per_segment
            self.learned_pooler = LearnedQueryAttention(
                embed_dim=D,
                num_queries_per_segment=num_queries_per_segment,
                max_queries=max_queries,
                num_heads=config.num_heads,
                use_flex_attention=True,
            )

    def _downsample(self, h_encoder, h_cross, bs, patch_lengths, patch_ids):
        """Override: use learned query pooling if enabled, else default."""
        if self.use_learned_queries:
            pooled, _ = self.learned_pooler(x=h_encoder, seg_id=patch_ids)
            return pooled
        return super()._downsample(h_encoder, h_cross, bs, patch_lengths, patch_ids)

    def _post_downsample(self, h, aux_loss):
        """Override: apply RVQ bottleneck after token projection."""
        z_q, vq_loss, vq_indices, vq_perplexity = self.quantizer(h)
        self._last_vq_indices = vq_indices
        self._last_vq_perplexity = vq_perplexity
        return z_q, aux_loss + vq_loss

    def _stage_indices(self) -> Optional[list]:
        """Per-stage code ids for the last forward, if the bank exposes them."""
        core = getattr(self.quantizer, "quantizer", self.quantizer)
        codec = getattr(core, "codec", None)
        idx = getattr(self, "_last_vq_indices", None)
        if codec is None or idx is None:
            return None
        digits, _ = codec.decompose(idx)
        return digits

    def decode(self, h, *args, **kwargs):
        """Register the next-code objective, then decode. ``h`` is the trunk
        output over patches, so position ``p`` conditions patch ``p+1``."""
        if (
            self.next_code is not None
            and self.training
            and torch.is_grad_enabled()
            and h.shape[1] >= 2
        ):
            self._register_next_code(h)
        return super().decode(h, *args, **kwargs)

    def _register_next_code(self, h: torch.Tensor) -> None:
        digits = self._stage_indices()
        if digits is None:
            return
        x = self.next_code_norm(h[:, :-1, :])
        total, correct, abstain, stages = None, [], [], 0
        for s, (proj, geometry) in enumerate(
            zip(self.next_code_proj, self.next_code_classifiers)
        ):
            if s >= len(digits) or tuple(digits[s].shape) != tuple(h.shape[:2]):
                break
            target = digits[s][:, 1:].reshape(-1)
            feats = proj(x).reshape(-1, x.shape[-1]).float()
            loss = self.next_code_loss.on_features(feats, target, geometry)
            total = loss if total is None else total + loss
            stages += 1
            with torch.no_grad():
                # Nearest centroid on the features as scored, not through the
                # geometry's RMS-normalized inference path.
                cen = geometry.centroids()
                score = 2.0 * feats.detach() @ cen.T - cen.pow(2).sum(-1)
                correct.append((score.argmax(-1) == target).float().mean())
                stats = self.next_code_loss._last_stats or {}
                abstain.append(stats.get("abstain_rate", 0.0))
        if total is None:
            return
        # Divided by log K, as CALM's code loss is, so the term's size does not
        # grow with the codebook. HALO's distance logits start sharper than a
        # chance classifier, so it opens at 2-3x this scale, not at 1.0.
        chance = math.log(max(2, self.next_code_classifiers[0].vocab_size))
        self._pending["next_code_halo"] = total / stages / chance
        self._next_code_diag = {
            "next_code_acc": torch.stack(correct).mean(),
            "next_code_abstain": sum(abstain) / stages,
        }

    def consume_pending_losses(self) -> Dict[str, torch.Tensor]:
        out, self._pending = self._pending, {}
        return out

    def training_metrics(self) -> dict:
        """VQ health for the dashboard (read by DynamicsLogger at the log
        interval - the same numbers the reset path prints to the terminal,
        now charted over time). The harmonic bottleneck wraps the core RVQ,
        so unwrap one level when present."""
        core = self.quantizer
        core = getattr(core, "quantizer", core)
        out = super().training_metrics()
        if hasattr(core, "telemetry"):
            out.update(core.telemetry())
        ppl = getattr(self, "_last_vq_perplexity", None)
        if ppl is not None:
            out["vq_perplexity"] = float(ppl)
        for key, value in self._next_code_diag.items():
            out[key] = float(value)
        # Compander anisotropy, when a GDN is in the frame. gamma starts flat at
        # 1/L (the isotropic sphere projection), so the coefficient of variation
        # over gamma is exactly 0 at init and rises only if the normalizer has
        # found a reason to treat directions differently. Without this the -j
        # run has no way to say whether the compander did anything or simply sat
        # at its initialization.
        gdn = getattr(self.quantizer, "gdn", None)
        if gdn is not None:
            g = gdn.gamma_sqrt.detach().pow(2)
            out["vq_gdn_anisotropy"] = float(g.std() / (g.mean() + 1e-12))
        return out

    # Chart hints for the metrics above. Per-stage keys are declared for up to
    # 4 residual stages; absent stages simply never emit their key.
    metric_descriptions = {
        **ByteLatentEncoder.metric_descriptions,
        "next_code_acc": {
            "description": (
                "Share of patches whose next code the trunk's HALO classifier "
                "names exactly, averaged over residual stages. Chance is 1/K. "
                "The direct read on whether the trunk predicts anything."
            ),
            "chart": {
                "title": "Next-Code Accuracy",
                "y_label": "accuracy",
                "y_scale": "logarithmic",
                "group": "next_code",
                "group_order": 75,
                "order": 0,
            },
        },
        "next_code_abstain": {
            "description": (
                "Probability mass the next-code HALO classifier puts on its abstain "
                "class. High = the trunk's features sit near the origin, unsure "
                "which code comes next."
            ),
            "chart": {
                "title": "Next-Code Abstain",
                "y_label": "abstain probability",
                "y_scale": "linear",
                "group": "next_code",
                "order": 1,
            },
        },
        "vq_gdn_anisotropy": {
            "description": (
                "Coefficient of variation of the GDN compander's gamma. 0 = still the "
                "isotropic init (inert); rising = resolution allocated unevenly across "
                "directions."
            ),
            "chart": {
                "title": "GDN Anisotropy",
                "y_label": "std / mean of gamma",
                "y_scale": "linear",
                "group": "vq",
                "group_order": 74,
                "order": 4,
            },
        },
        "vq_perplexity": {
            "description": (
                "Composed codebook perplexity across residual stages - distinct "
                "composed codes in use, out of K^depth. Falling toward 1 = codebook "
                "collapse."
            ),
            "chart": {
                "title": "VQ Perplexity (composed)",
                "y_label": "effective codes",
                "y_scale": "logarithmic",
                "group": "vq",
                "group_order": 74,
                "order": 0,
            },
        },
        **{
            f"vq_usage_entropy_s{s}": {
                "description": (
                    f"Stage-{s} codebook utilization: entropy of the EMA code-"
                    "usage distribution divided by log(K), so 1.0 = the whole "
                    "bank used uniformly and 0 = collapsed onto a single code. "
                    "Read this in preference to the stage perplexity beside it. "
                    "Perplexity is exp(entropy) of a PER-BATCH histogram, so it "
                    "is capped by the number of patches in a batch and that cap "
                    "moves with the sequence-length curriculum; this is an EMA "
                    "over many batches and is normalized by K, so it is "
                    "comparable both across time and across codebook sizes."
                ),
                "chart": {
                    "title": "VQ Codebook Utilization",
                    "y_label": "entropy / log(K)",
                    "y_scale": "linear",
                    "group": "vq",
                    "group_order": 74,
                    "order": 1,
                    "series_group": "vq_usage_entropy",
                    "series_label": f"stage {s}",
                },
            }
            for s in range(4)
        },
        **{
            f"vq_perplexity_s{s}": {
                "description": (
                    f"Stage-{s} codebook perplexity: effective codes in use "
                    "out of K at this residual stage. Later stages usually "
                    "sit lower (they quantize what earlier stages left)."
                ),
                "chart": {
                    "title": "VQ Stage Perplexity",
                    "y_label": "effective codes",
                    "y_scale": "logarithmic",
                    "group": "vq",
                    "order": 1,
                    "series_group": "vq_stage_ppl",
                    "series_label": f"stage {s}",
                },
            }
            for s in range(4)
        },
        **{
            f"vq_dead_frac_s{s}": {
                "description": (
                    f"Stage-{s} dead-code fraction: codes below the usage "
                    "threshold (EMA cluster size < 1), the population the "
                    "reset mechanism draws from. Persistently high = the "
                    "codebook is oversized or the encoder output has "
                    "collapsed onto few modes."
                ),
                "chart": {
                    "title": "VQ Dead-Code Fraction",
                    "y_label": "fraction of K",
                    "y_scale": "linear",
                    "group": "vq",
                    "order": 2,
                    "series_group": "vq_dead",
                    "series_label": f"stage {s}",
                },
            }
            for s in range(4)
        },
        **{
            f"vq_resets_s{s}": {
                "description": (
                    f"Stage-{s} cumulative dead-code resets (the count the "
                    "terminal 'VQ reset' log reports, accumulated). Read the "
                    "slope: a sustained slope = ongoing codebook churn; "
                    "flattening = the codebook has stabilized."
                ),
                "chart": {
                    "title": "VQ Code Resets (cumulative)",
                    "y_label": "codes reset",
                    "y_scale": "linear",
                    "group": "vq",
                    "order": 3,
                    "series_group": "vq_resets",
                    "series_label": f"stage {s}",
                },
            }
            for s in range(4)
        },
    }
