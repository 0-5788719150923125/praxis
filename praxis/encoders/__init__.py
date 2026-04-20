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
    use_hash_embeddings=True,
    hash_functions=1,
    hash_group_sizes=[3, 4, 5],
)

ByteLatentTransformer = partial(
    ByteLatentEncoder,
    local_architecture="transformer",
    patching_mode="space",
    n_layers_encoder=1,
    n_layers_decoder=1,
    use_hash_embeddings=True,
    hash_functions=1,
    hash_group_sizes=[3, 4, 5],
)

# BLT + residual VQ bottleneck (abstractinator defaults)
Abstractinator = partial(
    AbstractinatorEncoder,
    local_architecture="conv",
    patching_mode="space",
    n_layers_encoder=3,
    n_layers_decoder=3,
    use_hash_embeddings=True,
    hash_functions=1,
    hash_group_sizes=[3, 4, 5],
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

CALMSmall = partial(
    CALMEncoder,
    chunk_size=8,
    latent_dim=64,
    ae_hidden=256,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.1,
    noise_dim=64,
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

ENCODER_REGISTRY = dict(
    # Base class (use with explicit arguments)
    byte_latent=ByteLatentEncoder,
    # Recommended profiles
    byte_latent_conv=ByteLatentConv,
    byte_latent_transformer=ByteLatentTransformer,
    # BLT + residual VQ bottleneck
    abstractinator=Abstractinator,
    # CALM: token-chunk VAE + energy head (arXiv 2510.27688).
    # Tokenizer-specific variants adjust K: BPE=4, char=8, byte=16.
    # calm_small is the smoke-test profile.
    calm=CALM,
    calm_small=CALMSmall,
    calm_byte=CALMByte,
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
