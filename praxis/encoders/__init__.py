from functools import partial

from praxis.encoders.abstractinator import (
    AbstractinatorCALM,
    AbstractinatorEncoder,
)
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

# Abstractinator with the residual VQ moved into the CALM harmonic coordinate
# frame: patch latents rotate into the standing-wave basis (the same
# harmonic_matrix the HarmonicCodec builds its mix from), are energy-normalized
# there, and the residual codes quantize harmonic AMPLITUDES - the paper
# addendum's conjecture (residual codes read as amplitudes in the harmonic
# basis) made concrete. The objective stays byte CE end-to-end: single-stage,
# never frozen, no KL - byte-latent convergence with a CALM-shaped latent
# geometry. bottleneck_ratio=0.5 keeps the frame lossy (a low-frequency
# spectral budget, the codec's latent_dim < K*E mechanism).
AbstractinatorHarmonic = partial(
    Abstractinator,
    bottleneck="harmonic",
    bottleneck_ratio=0.5,
)

# Serpent variant: a learned periodic nonlinearity after the analysis
# transform, mirroring codec_kind="harmonic_serpent" (the calm-d lineage
# codec) - the encode into the spectral frame is learnable rather than a
# fixed rotation. Still single-stage and never frozen.
AbstractinatorHarmonicSerpent = partial(
    AbstractinatorHarmonic,
    bottleneck="harmonic_serpent",
)

# Mean-pooled variant. The default "max" pooling (pooling_downsample ->
# patch_reduce with "amax") carries a systematic length bias: the expectation of
# a max over n vectors grows like sqrt(2 ln n), so under space patching - where
# n runs from 1 to 10+ - the patch vector's magnitude encodes how many bytes
# fell in the patch before any content does. HarmonicResidualVQ then
# RMS-normalizes onto the sphere, erasing the magnitude but not the direction
# bias it induced.
#
# Mean pooling is not magnitude-neutral either (a mean over n shrinks like
# 1/sqrt(n), the INVERSE bias), but the difference that matters is in DIRECTION,
# which survives the RMS normalization. A mean is an unbiased estimate of the
# patch's content direction at every n; only its variance depends on n. A max is
# an order statistic, so which coordinates it selects - and therefore where it
# points - shifts systematically as n grows.
#
# The pooling mode is matched by substring in pooling_downsample: the key is
# "avg", not "mean"; "mean" matches nothing and trips its assert.
AbstractinatorHarmonicSerpentAvg = partial(
    AbstractinatorHarmonicSerpent,
    downsampling_method="avg",
)

# Codebook sized to the token vocabulary instead of the inherited 16384. `None`
# selects AbstractinatorEncoder's documented default (config.vocab_size).
# Measured on -h/-i at step 6k-10k: roughly 470 and 218 effective codes in use
# at stages 0 and 1 against K=16384 - under 3% utilization, with ~140k
# cumulative dead-code resets, eight full turnovers of the bank.
#
# Not capacity-matched, deliberately: this REMOVES parameters. The codebook is
# an nn.Parameter (vector_quantizer.py), so the bank costs optimizer state as
# well as weights.
#
# Caveat: tying K to vocab_size is a coincidence of scale, not a principle - the
# bank indexes patch latents, not tokens. It lands right here because the
# measurements say ~1k AND vocab_size is 1024. If the tokenizer changes, K
# should be pinned explicitly.
AbstractinatorHarmonicSerpentVocabBank = partial(
    AbstractinatorHarmonicSerpent,
    vq_codebook_size=None,
)

# Both VQ-side fixes at once (abstractinator-i), bundled because with a fixed
# compute budget the efficient search is coarse-to-fine: group the changes that
# push the same direction on the same suspected fault - here, a bottleneck not
# earning its parameters.
#
#   1. K = config.vocab_size (1024) instead of 16384. See the entry above.
#   2. NO Serpent on the analysis transform (`AbstractinatorHarmonic`, not
#      ...Serpent). Serpent is PERIODIC and sits immediately before
#      quantization, and a periodic map is not injective: two distinct patch
#      latents can land on the same point, which attacks the one thing a
#      quantizer exists to do. Removing it makes the frame a pure rotation plus
#      the sphere projection.
#
# `bottleneck_ratio` stays 0.5: a smaller codebook wants a smaller latent space
# (N codes give ~N^(1/d) resolution per axis), and -h's telemetry exonerates the
# ratio anyway - dead fraction fell monotonically to 0.27/0.097 at ratio 0.5.
#
# The bundle stays readable because K's effect is PREDICTABLE. -h used ~470
# codes at stage 0, so K alone predicts vq_dead_frac_s0 settling near 0.5.
# Meaningfully below 0.5 means dropping Serpent contributed; at 0.5 means K did
# the work and Serpent was neutral.
AbstractinatorHarmonicVocabBank = partial(
    AbstractinatorHarmonic,
    vq_codebook_size=None,
)

# -i plus a learned compander in place of the fixed RMS normalization
# (abstractinator-j). The sphere projection is the ISOTROPIC SPECIAL CASE of GDN
# - gamma_ij = 1/L, beta = 1e-5 - so the module initializes bit-identical to -i
# and the run measures only whether anisotropy earns anything.
#
# Classical quantization theory says a fixed codebook should spend resolution
# where the source density is, and a monotonic warp is how you arrange that
# (mu-law; GDN is the learned multivariate version, and what neural image codecs
# put in front of their quantizer). The measured fault in this stack is that
# patch directions CONCENTRATE and the bank starves - -i's avg-pooling run drove
# dead fraction to 0.85 - and max pooling's only virtue is that its order
# statistic disperses them by accident. A compander does it deliberately.
#
# Curvature, not periodicity: companding needs INVERTIBILITY, so the warp must
# be monotonic. A periodic map is not injective and can alias two distinct patch
# latents onto one point, which is why Serpent stays out of the bottleneck.
AbstractinatorHarmonicGDNVocabBank = partial(
    AbstractinatorHarmonic,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
)

# The GDN vocab-bank profile on FIXED-SIZE patches instead of space patching.
#
# Static patching is the worst-performing mode in BLT's own ablations, so this
# is a deliberate contrarian test. The reason to run it: a harmonic model builds
# a PERIODIC latent, and patch-time is a resampling of byte-time. A fixed patch
# size makes that resampling UNIFORM, where space and entropy patching make the
# sampling lattice depend on content - non-uniform sampling, which smears the
# frequency axis. BLT measured bits-per-byte on a non-periodic transformer,
# where an irregular lattice costs nothing structural.
#
# PATCH_SIZE is compute-matched: space patching on this repo's own source
# measures a mean patch length of 7.25 bytes (median 6, p25 4, p75 9), so 8
# keeps the global decoder's sequence length within ~10% of what the space
# patcher produces; 6 would lengthen it by ~21% and confound a patching change
# with a compute change.
#
# Free of BOE tokens: see Patcher._static_patching and ByteLatentEncoder.nb_boe.
# Free of BOE tokens: see Patcher._static_patching and ByteLatentEncoder.nb_boe.
AbstractinatorHarmonicGDNVocabBankStatic = partial(
    AbstractinatorHarmonic,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
)

# abstractinator-p: the profile above, run BESIDE a second codec - CALM's
# autoencoder - into the same trunk. Two encoders side by side, each with its
# own objective and therefore independently optimizable:
#
#   patch features h
#     |
#     +-- analysis -> HarmonicResidualVQ -> synthesis --> z_q   discrete codec
#     |
#     +-- PatchVAE.encode -> (mu, logvar) -> z_c                continuous codec
#     |     +-- PatchVAE.decode -> h_hat, relative recon error
#     |     +-- free-bits KL
#     v
#   z = z_q + gate * z_c  ->  trunk
#     |
#     +-- EnergyHead: predict the next VAE LATENT (CALM's energy score)
#     +-- code CE:    predict the next RVQ code (the dense signal that pays)
#
# THIS IS ONLY WELL-POSED BECAUSE THE PATCHING IS STATIC. Both codecs emit
# exactly one latent per patch at `patch_size=8`, so `z_q + z_c` is an alignable
# merge. Two encoders with different chunking would emit different numbers of
# latents in different units.
#
# A real VAE and not an `nn.Linear(D, 2*D)`, because CALM's VAE does three jobs.
# Compressing K tokens into one vector is the one the Abstractinator's local
# encoder genuinely already does. Supplying a continuous, KL-regularized,
# unit-scale, STATIONARY latent space and a per-patch POSTERIOR for the energy
# score's target draws are the two an RVQ does not.
#
# Near-silent at step 0, not bit-identical: `arm_gate` starts at sigmoid(-6), so
# z_c enters at ~0.25% of ||z_q||. `calm_arm_ratio` is the receipt.
AbstractinatorHarmonicGDNVocabBankStaticCALM = partial(
    AbstractinatorCALM,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
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
# paper's N/M/blocks so the gradient isn't sample-starved. Dims are fractions of
# hidden_size (0.25/1.0/0.25 == 64/256/64 at hidden=256).
#
# Two-stage like the reference: train the codec alone until the freeze, with the
# KL annealed in over the same window, then freeze it and train only the energy
# head against a stationary target. The freeze is convergence-driven - schedules
# are left unset, so the codec trains until its reconstruction plateaus (the
# window's linear trend drops below its own noise), capped by
# ae_max_pretrain_steps as a backstop. Watch calm_recon_ce /
# calm_pretrain_flatness descend and calm_ae_frozen flip at the boundary.
# kl_beta/kl_clip/N/M/vote pool match the paper (arXiv:2510.27688).
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

# CALMByteFixed's static scaffold + a small never-frozen learned residual
# (codec_kind="hybrid"): the bias-variance midpoint between the fixed codec
# (pure stationary) and the VAE (two-stage learned). Starts identical to fixed
# (residual zero-init), then the latent slowly drifts toward better organization
# without ever freezing - "stable, yet always slightly improving". Single-stage.
# Tests whether a slow-moving target reclaims any VAE benefit at scale/large K
# where a learned latent might earn its keep. Not yet run; inert option.
CALMByteHybrid = partial(
    CALMByteFlow,
    codec_kind="hybrid",
    ae_freeze_steps=0,
)

# FixedCodec with harmonic (standing-wave) bases instead of random orthonormal
# ones (codec_kind="harmonic"): structured rather than arbitrary latent geometry,
# every feature coupled through a shared spectrum, per-vocab + per-K modulation.
# Deterministic/stationary encode, single-stage. K=8 (longer patches than
# CALMByteFlow's K=4): the separable 2D harmonic basis gives the patch-position
# axis its own frequency budget, so smooth-across-patch structure compresses
# gracefully as K grows - this codec is the one built to absorb larger K, so it
# carries the longer patch.
CALMByteHarmonicCodec = partial(
    CALMByteFlow,
    codec_kind="harmonic",
    ae_freeze_steps=0,
    chunk_size=8,
)
# Serpent variant: same harmonic codec + K=8, but the encode gains a learned
# periodic Serpent nonlinearity after the transform (codec_kind="harmonic_serpent").
# This makes encode learnable and NON-stationary - trading the deterministic
# fixed-latent property for expressiveness (still single-stage, never frozen).
# Derived from the codec profile so K stays in sync.
CALMByteHarmonicSerpent = partial(
    CALMByteHarmonicCodec,
    codec_kind="harmonic_serpent",
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
    # Residual codes as harmonic amplitudes (the CALM-bridge conjecture, run)
    abstractinator_harmonic=AbstractinatorHarmonic,
    abstractinator_harmonic_serpent=AbstractinatorHarmonicSerpent,
    abstractinator_harmonic_serpent_avg=AbstractinatorHarmonicSerpentAvg,
    abstractinator_harmonic_serpent_vocab_bank=AbstractinatorHarmonicSerpentVocabBank,
    abstractinator_harmonic_vocab_bank=AbstractinatorHarmonicVocabBank,
    abstractinator_harmonic_gdn_vocab_bank=AbstractinatorHarmonicGDNVocabBank,
    # Same, on a uniform patch lattice (see the profile's note).
    abstractinator_harmonic_gdn_vocab_bank_static=AbstractinatorHarmonicGDNVocabBankStatic,
    # ...plus a continuous CALM arm beside the discrete one (abstractinator-p).
    # Identical to the profile above at initialization - the posterior is
    # zero-initialized, so z_c starts at exactly 0 - which makes the swap an A/B
    # rather than a reroll. See praxis/encoders/abstractinator/calm.py for why
    # the discrete arm is what pays for the continuous one's conditional.
    abstractinator_harmonic_gdn_vocab_bank_static_calm=AbstractinatorHarmonicGDNVocabBankStaticCALM,
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
    calm_byte_hybrid=CALMByteHybrid,
    calm_byte_harmonic_codec=CALMByteHarmonicCodec,
    calm_byte_harmonic_serpent=CALMByteHarmonicSerpent,
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
