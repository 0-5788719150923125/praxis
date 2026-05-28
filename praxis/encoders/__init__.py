from functools import partial

from praxis.encoders.abstractinator import AbstractinatorEncoder
from praxis.encoders.byte_latent import ByteLatentEncoder
from praxis.encoders.calm import CALMEncoder

# ByteLatent Encoder Profiles
# These provide convenient presets for different use cases

# Basic profiles
ByteLatentConv = partial(
    ByteLatentEncoder,
    local_architecture="conv",
    patching_mode="space",
    n_layers_encoder=3,
    n_layers_decoder=3,
    embeddings="byte_hash",
)

ByteLatentConvSmall = partial(
    ByteLatentConv,
    n_layers_encoder=2,
    n_layers_decoder=2,
)

ByteLatentTransformer = partial(
    ByteLatentEncoder,
    local_architecture="transformer",
    patching_mode="space",
    n_layers_encoder=1,
    n_layers_decoder=1,
    embeddings="byte_hash",
)

# BLT + residual VQ bottleneck (abstractinator defaults)
Abstractinator = partial(
    AbstractinatorEncoder,
    local_architecture="conv",
    patching_mode="space",
    n_layers_encoder=3,
    n_layers_decoder=3,
    embeddings="byte_hash",
    vq_codebook_size=16384,
)

# CALM profiles. Defaults track the paper (arXiv 2510.27688). Tokenizer-
# specific variants exist because K ("one word of meaning per latent")
# scales with tokenizer granularity: BPE=4, char=8, byte=16.
CALM = partial(
    CALMEncoder,
    chunk_size=8,
    latent_dim=128,
    ae_hidden=512,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.15,
    noise_dim=128,
    energy_blocks=3,
    energy_samples_n=8,
    energy_samples_m=100,
    energy_alpha=1.0,
)

# Small profiles scale relative to config.hidden_size (float dims), so the
# encoder tracks the model instead of pinning absolute widths. Paper-scale
# profiles below keep absolute ints to preserve the published capacities.
CALMSmall = partial(
    CALMEncoder,
    chunk_size=8,
    latent_dim=0.25,
    ae_hidden=1.0,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.1,
    noise_dim=0.25,
    energy_blocks=2,
    energy_samples_n=8,
    energy_samples_m=100,
    energy_alpha=1.0,
)

CALMByte = partial(
    CALMEncoder,
    chunk_size=16,
    latent_dim=128,
    ae_hidden=512,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.15,
    noise_dim=128,
    energy_blocks=3,
    energy_samples_n=8,
    energy_samples_m=100,
    energy_alpha=1.0,
)

CALMBpe = partial(
    CALMEncoder,
    chunk_size=4,
    latent_dim=128,
    ae_hidden=512,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.15,
    noise_dim=128,
    energy_blocks=3,
    energy_samples_n=8,
    energy_samples_m=100,
    energy_alpha=1.0,
)

# Byte K with a much smaller VAE + energy head, for compact experiments.
# Dims are fractions of hidden_size (0.25/1.0/0.25 == 64/256/64 at hidden=256).
# Joint VAE+LM training is unstable without a warmup phase (the paper trains
# the VAE first to convergence), so the small profile defaults to a ~2k-step
# energy hold and a ~1k-step KL anneal.
CALMByteSmall = partial(
    CALMEncoder,
    chunk_size=16,
    latent_dim=0.25,
    ae_hidden=1.0,
    kl_beta=1e-3,
    kl_clip=0.5,
    kl_warmup_steps=1000,
    ae_dropout=0.1,
    noise_dim=0.25,
    energy_blocks=2,
    energy_samples_n=4,
    energy_samples_m=16,
    energy_alpha=1.0,
    energy_warmup_steps=2000,
)


def is_byte_latent_encoder(encoder_type: str) -> bool:
    """Check if an encoder type is a ByteLatentEncoder or subclass."""
    encoder_cls = ENCODER_REGISTRY.get(encoder_type)
    if encoder_cls is None:
        return False
    actual_cls = getattr(encoder_cls, "func", encoder_cls)
    return issubclass(actual_cls, ByteLatentEncoder)


ENCODER_REGISTRY = dict(
    # Base class (use with explicit arguments)
    byte_latent=ByteLatentEncoder,
    # Recommended profiles
    byte_latent_conv=ByteLatentConv,
    byte_latent_conv_small=ByteLatentConvSmall,
    byte_latent_transformer=ByteLatentTransformer,
    # BLT + residual VQ bottleneck
    abstractinator=Abstractinator,
    # CALM: token-chunk VAE + energy head (arXiv 2510.27688).
    # Tokenizer-specific variants adjust K: BPE=4, char=8, byte=16.
    # calm_small is the smoke-test profile.
    calm=CALM,
    calm_small=CALMSmall,
    calm_byte=CALMByte,
    calm_byte_small=CALMByteSmall,
    calm_bpe=CALMBpe,
    # # Entropy-based patching
    # byte_latent_transformer_entropy=ByteLatentTransformerEntropy,
    # # Lightweight variants
    # byte_latent_transformer_light=ByteLatentTransformerLight,
    # byte_latent_recurrent=ByteLatentRecurrent,
    # byte_latent_entropy_conv=ByteLatentEntropyConv,
    # byte_latent_entropy_recurrent=ByteLatentEntropyRecurrent,
    # byte_latent_light_conv=ByteLatentLightConv,
    # byte_latent_light_recurrent=ByteLatentLightRecurrent,
    # # Experimental
    # byte_latent_cross_attn=ByteLatentCrossAttn,
)
