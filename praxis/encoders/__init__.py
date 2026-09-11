from functools import partial
from typing import Any, Dict

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

# Abstractinator profiles: BLT plus a residual-VQ bottleneck on the patch
# vectors, versioned. The registry lists the versioned names; descriptive names
# resolve through ENCODER_REGISTRY.unlisted.

# The reference Abstractinator: a plain residual VQ on the pooled patch vectors.
AbstractinatorRVQ = partial(
    AbstractinatorEncoder,
    local_architecture="conv",
    patching_mode="space",
    n_layers_encoder=3,
    n_layers_decoder=3,
    embeddings="byte_hash",
    vq_codebook_size=16384,
)

# The residual VQ in the CALM harmonic frame: patch latents rotate into the
# standing-wave basis (praxis/encoders/basis.py), are RMS-normalized there, and
# the residual codes quantize harmonic amplitudes. bottleneck_ratio=0.5 keeps a
# low-frequency spectral budget. Every versioned profile below builds on it.
AbstractinatorHarmonic = partial(
    AbstractinatorRVQ,
    bottleneck="harmonic",
    bottleneck_ratio=0.5,
)

# v0: a learned Serpent activation on the analysis transform, 16384 codes,
# space patching. Serpent is periodic, so it can alias two patch latents onto
# one point in front of the quantizer.
AbstractinatorV0 = partial(AbstractinatorHarmonic, bottleneck="harmonic_serpent")

# v1: no Serpent, a codebook sized from config (codebook_size, else
# vocab_size), fixed 8-byte patches - a uniform resampling of byte-time for a
# periodic latent, compute-matched to space patching's 7.25-byte mean - and a
# GDN compander in front of the quantizer.
AbstractinatorV1 = partial(
    AbstractinatorHarmonic,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
)

# v1 with a continuous CALM arm beside the discrete codec: a
# PatchVAE over the same patch features, z = z_q + gate * z_c into the trunk, an
# energy head on the next VAE latent and a CE on the next RVQ code. Static
# patching is what makes the two codecs' latents align one per patch. See
# praxis/encoders/abstractinator/calm.py.
AbstractinatorV1CALM = partial(
    AbstractinatorCALM,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
)

# v2: v1 with plain RMS normalization in front of the quantizer instead of the
# GDN compander - the commitment loss rewards a smaller quantizer input, and a
# fixed normalization leaves no scale to shrink - and the byte path and the
# trunk blended by a gate over RMS-normalized streams instead of added
# (praxis/encoders/byte_latent/merge.py).
AbstractinatorV2 = partial(
    AbstractinatorHarmonic,
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
    merge="gated",
)

# Unlisted profiles, built by descriptive name: v0 mean-pooled, v0 with a
# config-sized codebook, and the harmonic frame with a config-sized codebook
# under RMS and under GDN, both space-patched.
AbstractinatorV0Avg = partial(AbstractinatorV0, downsampling_method="avg")
AbstractinatorV0Bank = partial(AbstractinatorV0, vq_codebook_size=None)
AbstractinatorHarmonicBank = partial(AbstractinatorHarmonic, vq_codebook_size=None)
AbstractinatorHarmonicGDNBank = partial(
    AbstractinatorHarmonic, bottleneck="harmonic_gdn", vq_codebook_size=None
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


def encoder_traits(encoder_type: str) -> Dict[str, bool]:
    """What a profile is built from, for callers that must not parse its name:
    an Abstractinator at all, one whose bottleneck quantizes harmonic
    amplitudes, and one carrying the continuous CALM arm."""
    profile = ENCODER_REGISTRY.get(encoder_type)
    cls = getattr(profile, "func", profile)
    if not isinstance(cls, type) or not issubclass(cls, AbstractinatorEncoder):
        return {
            "abstractinator": False,
            "harmonic_bottleneck": False,
            "calm_arm": False,
        }
    bottleneck = getattr(profile, "keywords", {}).get("bottleneck", "rvq")
    return {
        "abstractinator": True,
        "harmonic_bottleneck": str(bottleneck).startswith("harmonic"),
        "calm_arm": issubclass(cls, AbstractinatorCALM),
    }


class EncoderRegistry(dict):
    """Profiles by name. ``unlisted`` maps further names - the ones checkpoints
    and configs may carry - to a listed name or to a profile of their own; they
    resolve on lookup but are not listed."""

    def __init__(self, profiles: Dict[str, Any], unlisted: Dict[str, Any]) -> None:
        super().__init__(profiles)
        self.unlisted = unlisted

    def __missing__(self, key: str) -> Any:
        target = self.unlisted[key]
        return self[target] if isinstance(target, str) else target

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or key in self.unlisted


ENCODER_REGISTRY = EncoderRegistry(
    dict(
        # Base class (use with explicit arguments)
        byte_latent=ByteLatentEncoder,
        # Recommended profiles
        byte_latent_conv=ByteLatentConv,
        byte_latent_conv_small=ByteLatentConvSmall,
        byte_latent_transformer=ByteLatentTransformer,
        # BLT + residual VQ bottleneck, by generation (see each profile's note).
        abstractinator_rvq=AbstractinatorRVQ,
        abstractinator_v0=AbstractinatorV0,
        abstractinator_v1=AbstractinatorV1,
        abstractinator_v1_calm=AbstractinatorV1CALM,
        abstractinator_v2=AbstractinatorV2,
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
    ),
    unlisted=dict(
        abstractinator="abstractinator_rvq",
        abstractinator_harmonic_serpent="abstractinator_v0",
        abstractinator_harmonic_gdn_vocab_bank_static="abstractinator_v1",
        abstractinator_harmonic_gdn_vocab_bank_static_calm="abstractinator_v1_calm",
        abstractinator_harmonic=AbstractinatorHarmonic,
        abstractinator_harmonic_serpent_avg=AbstractinatorV0Avg,
        abstractinator_harmonic_serpent_vocab_bank=AbstractinatorV0Bank,
        abstractinator_harmonic_vocab_bank=AbstractinatorHarmonicBank,
        abstractinator_harmonic_gdn_vocab_bank=AbstractinatorHarmonicGDNBank,
    ),
)
