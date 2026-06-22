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

# Byte K with a smaller VAE for compact experiments; energy head uses the
# paper's N/M/blocks so the gradient isn't sample-starved (the original 4/16
# combo gave high-variance estimates and was the main throttle on the energy
# head's learning curve). Dims are fractions of hidden_size
# (0.25/1.0/0.25 == 64/256/64 at hidden=256).
#
# Two-stage like the reference: train the codec alone until the freeze, with
# the KL annealed in over the same window so the final latent is smooth, then
# freeze it and train only the energy head against a stationary target. The
# freeze is convergence-driven: schedules are left unset, so the codec trains
# until its reconstruction plateaus (the window's linear trend drops below its
# own noise), then freezes - capped by ae_max_pretrain_steps as a backstop. Watch
# calm_recon_ce / calm_pretrain_flatness descend and calm_ae_frozen flip at
# the boundary. kl_beta/kl_clip/N/M/vote pool match the paper (arXiv 2510.27688):
# β=1e-3 with free-bits clip 0.5 keeps the latent modelable without
# over-regularizing; the ~500-sample vote pool is the paper's accuracy-diversity
# frontier (50 was far too noisy for patch-vote decoding).
CALMByteSmall = partial(
    CALMEncoder,
    chunk_size=8,
    latent_dim=0.5,
    ae_hidden=1.5,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.1,
    noise_dim=0.5,
    energy_blocks=3,
    energy_samples_n=8,
    energy_samples_m=100,
    energy_alpha=1.0,
    vote_num_samples=500,
)

# Baseline for the calm-a ablations: the published repo's dims (latent 128, AE
# hidden 512, noise 64, 4 head blocks, dropout 0.15) with ONE departure from the
# authors - a deeper residual codec (vae_depth=4). chunk_size=16 is the byte-level
# K (~16 bytes per latent); CALMTmRef overrides it to K=4 for subword tokenizers.
CALMByteRef = partial(
    CALMEncoder,
    chunk_size=16,
    latent_dim=0.5,
    ae_hidden=2.0,
    vae_depth=4,
    kl_beta=1e-3,
    kl_clip=0.5,
    ae_dropout=0.15,
    noise_dim=0.25,
    energy_blocks=4,
    energy_samples_n=8,
    energy_samples_m=100,
    energy_alpha=1.0,
    vote_num_samples=500,
    energy_prior="none",
    energy_anchor_weight=0.0,
)

# CALMByteRef with the energy head swapped for a flow-matching head: the probe
# showed the codec round-trips losslessly but the energy head never learns the
# conditional (acc 0 even teacher-forced), so the flow head's dense low-variance
# objective is the calm-a-2 intervention.
# K=4 (4:1 codec compression) not the reference's K=16: at 16:1 the codec
# manifold was a thin high-norm shell that the flow head couldn't hit at small
# scale (off-manifold -> gibberish). 4:1 doubled head token-acc (0.15 -> ~0.35),
# confirming aggressive patching was a real constraint. kl_beta stays at the
# reference 1e-3 (inherited): the earlier 1e-2 bump was diagnosing off-manifold
# gibberish that turned out to be the padding/seed generation bug, not the
# manifold geometry, so the more-faithful low beta is the default again.
# Cap stage 1 well under the 20k backstop so it can't pretrain for days; K=4
# recon converges fast, so the detector likely freezes before this anyway.
CALMByteFlow = partial(
    CALMByteRef,
    head_kind="flow",
    chunk_size=4,
    ae_max_pretrain_steps=3000,
)

# CALMByteFlow with the flow head's generic velocity net swapped for the harmonic
# latent head (head_kind="harmonic"): same flow-matching objective, but the flow
# runs in a compact harmonic coefficient space so each next-latent is a smooth
# low-frequency superposition. The bet (research/main.tex log-scaling) is that
# fewer effective output dims = lower head variance = faster convergence at small
# scale - the scale-wall lever the flow head can't pull. Not yet run; the fast
# proxy (calm-a-3) is the bench to A/B it against flow once that loop is trusted.
CALMByteHarmonic = partial(
    CALMByteFlow,
    head_kind="harmonic",
)

# CALMByteFlow with the learned VAE swapped for a FIXED deterministic codec
# (codec_kind="fixed"): the encoder is a frozen orthonormal byte transform, only
# the decoder learns. The latent target is stationary from step 0, so no codec
# freeze is needed - ae_freeze_steps=0 runs it single-stage (decoder + flow head
# train jointly, head active immediately against the fixed target). Tests the
# bet that a static codec is "good enough" at our tiny scale + 264-byte vocab,
# eliminating two-stage training entirely. Not yet run; inert option.
CALMByteFixed = partial(
    CALMByteFlow,
    codec_kind="fixed",
    ae_freeze_steps=0,
)

# CALMByteRef at the reference's true patch granularity: K=4 subword tokens
# (~15-20 bytes of text per latent) for a TokenMonster/BPE tokenizer. The
# calm-a-1 ablation uses this; calm-a-2 uses CALMByteRef (K=16) directly so the
# only moved variable is byte vs subword tokenization.
CALMTmRef = partial(
    CALMByteRef,
    chunk_size=4,
)

# CALMByteSmall with harmonic codec dropout: the scalar rate becomes a
# standing-wave field over (patch position, channel), n cycles per axis.
CALMByteSmallHarmonic = partial(
    CALMByteSmall,
    ae_dropout_mode="harmonic",
    ae_dropout_cycles=2,
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
    calm_byte_small_harmonic=CALMByteSmallHarmonic,
    calm_byte_ref=CALMByteRef,
    calm_byte_flow=CALMByteFlow,
    calm_byte_harmonic=CALMByteHarmonic,
    calm_byte_fixed=CALMByteFixed,
    calm_tm_ref=CALMTmRef,
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
