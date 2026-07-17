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

Based on: https://github.com/OilProducts/abstractinator
"""

from typing import Optional, TypeVar

from praxis.encoders.byte_latent.encoder import ByteLatentEncoder
from praxis.encoders.quantization import (
    HarmonicResidualVQ,
    LearnedQueryAttention,
    MultiStageResidualVQ,
)

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


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
            downsampling_method=downsampling_method,
        )

        D = config.hidden_size
        K = vq_codebook_size if vq_codebook_size is not None else 1024

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
        elif bottleneck in ("harmonic", "harmonic_serpent"):
            self.quantizer = HarmonicResidualVQ(
                dim=D,
                latent_dim=max(1, int(D * bottleneck_ratio)),
                nonlinear=(bottleneck == "harmonic_serpent"),
                **vq_kwargs,
            )
        else:
            raise ValueError(f"Unknown abstractinator bottleneck: {bottleneck!r}")

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
