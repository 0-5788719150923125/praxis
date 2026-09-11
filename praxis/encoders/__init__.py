from functools import partial
from typing import Dict

from praxis import registry
from praxis.encoders.abstractinator import (
    AbstractinatorCALM,
    AbstractinatorEncoder,
)
from praxis.encoders.byte_latent import ByteLatentEncoder
from praxis.encoders.calm import CALMEncoder
from praxis.registry import Alias, Entry, unwrap

# Profiles. What each one is lives in its ``encoders`` entry below.

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


AbstractinatorRVQ = partial(
    AbstractinatorEncoder,
    local_architecture="conv",
    patching_mode="space",
    n_layers_encoder=3,
    n_layers_decoder=3,
    embeddings="byte_hash",
    vq_codebook_size=16384,
)

AbstractinatorHarmonic = partial(
    AbstractinatorRVQ,
    bottleneck="harmonic",
    bottleneck_ratio=0.5,
)

AbstractinatorV0 = partial(AbstractinatorHarmonic, bottleneck="harmonic_serpent")

AbstractinatorV1 = partial(
    AbstractinatorHarmonic,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
)

AbstractinatorV1CALM = partial(
    AbstractinatorCALM,
    bottleneck="harmonic_gdn",
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
)

AbstractinatorV2 = partial(
    AbstractinatorHarmonic,
    vq_codebook_size=None,
    patching_mode="static",
    patch_size=8,
    merge="normalized",
)

AbstractinatorV2Additive = partial(AbstractinatorV2, merge="add")

AbstractinatorV0Avg = partial(AbstractinatorV0, downsampling_method="avg")
AbstractinatorV0Bank = partial(AbstractinatorV0, vq_codebook_size=None)
AbstractinatorHarmonicBank = partial(AbstractinatorHarmonic, vq_codebook_size=None)
AbstractinatorHarmonicGDNBank = partial(
    AbstractinatorHarmonic, bottleneck="harmonic_gdn", vq_codebook_size=None
)

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

CALMByteFlow = partial(
    CALMByteRef,
    head_kind="flow",
    chunk_size=4,
    ae_max_pretrain_steps=3000,
)

CALMByteHarmonic = partial(
    CALMByteFlow,
    head_kind="harmonic",
)

CALMByteFixed = partial(
    CALMByteFlow,
    codec_kind="fixed",
    ae_freeze_steps=0,
)

CALMByteHybrid = partial(
    CALMByteFlow,
    codec_kind="hybrid",
    ae_freeze_steps=0,
)

CALMByteHarmonicCodec = partial(
    CALMByteFlow,
    codec_kind="harmonic",
    ae_freeze_steps=0,
    chunk_size=8,
)
CALMByteHarmonicSerpent = partial(
    CALMByteHarmonicCodec,
    codec_kind="harmonic_serpent",
)

CALMTmRef = partial(
    CALMByteRef,
    chunk_size=4,
)

CALMByteSmallHarmonic = partial(
    CALMByteSmall,
    ae_dropout_mode="harmonic",
    ae_dropout_cycles=2,
)


def is_byte_latent_encoder(encoder_type: str) -> bool:
    """Check if an encoder type is a ByteLatentEncoder or subclass."""
    profile = registry.namespace("encoders").get(encoder_type)
    if profile is None:
        return False
    return issubclass(unwrap(profile)[0], ByteLatentEncoder)


def encoder_traits(encoder_type: str) -> Dict[str, bool]:
    """What a profile is built from, for callers that must not parse its name:
    an Abstractinator at all, one whose bottleneck quantizes harmonic
    amplitudes, and one carrying the continuous CALM arm."""
    cls, _, keywords = unwrap(registry.namespace("encoders").get(encoder_type))
    if not isinstance(cls, type) or not issubclass(cls, AbstractinatorEncoder):
        return {
            "abstractinator": False,
            "harmonic_bottleneck": False,
            "calm_arm": False,
        }
    bottleneck = keywords.get("bottleneck", "rvq")
    return {
        "abstractinator": True,
        "harmonic_bottleneck": str(bottleneck).startswith("harmonic"),
        "calm_arm": issubclass(cls, AbstractinatorCALM),
    }


registry.declare(
    "encoders",
    title="Input encoders",
    doc=(
        (
            "Front-end encoders between the tokenizer and the decoder: the byte-latent "
            "(BLT) family, the Abstractinator (BLT plus a residual VQ bottleneck on the "
            "patch vectors, versioned v0 to v2), and CALM (a token-chunk VAE plus a "
            "next-latent head; K, the tokens per latent, scales with tokenizer "
            "granularity: BPE 4, char 8, byte 16). Unset, tokens are embedded directly. "
            "Set, an encoder turns the input into the sequence the decoder sees, maps the "
            "decoder's output back and adds its own term to the loss; runs with an encoder "
            "skip the KV cache and cannot use the tied head. The byte-latent and "
            "Abstractinator encoders read raw bytes, so they switch the tokenizer to "
            "``byte_level`` on their own. Descriptive Abstractinator names resolve to the "
            "versioned profiles."
        )
    ),
    entries={
        "byte_latent": Entry(
            ByteLatentEncoder,
            (
                "The byte-latent (BLT) encoder at its constructor defaults, for use "
                "with explicit arguments. The profiles below are presets of it."
            ),
        ),
        "byte_latent_conv": Entry(
            ByteLatentConv,
            (
                "Byte-latent encoder with three convolutional local layers on each "
                "side of the trunk, space patching and hashed n-gram byte embeddings."
            ),
        ),
        "byte_latent_conv_small": Entry(
            ByteLatentConvSmall,
            "byte_latent_conv with two local layers on each side.",
        ),
        "byte_latent_transformer": Entry(
            ByteLatentTransformer,
            (
                "Byte-latent encoder with one transformer local layer on each side of "
                "the trunk, space patching and hashed n-gram byte embeddings."
            ),
        ),
        "abstractinator_rvq": Entry(
            AbstractinatorRVQ,
            (
                "The reference Abstractinator: byte_latent_conv plus a plain residual "
                "VQ on the pooled patch vectors, 16384 codes."
            ),
        ),
        "abstractinator_v0": Entry(
            AbstractinatorV0,
            (
                "The residual VQ in the harmonic frame (patch latents rotated into the "
                "standing-wave basis, RMS-normalized, residual codes on harmonic "
                "amplitudes) with a learned Serpent activation on the analysis "
                "transform, 16384 codes and space patching. Serpent is periodic, so it "
                "can alias two patch latents onto one point in front of the quantizer."
            ),
        ),
        "abstractinator_v1": Entry(
            AbstractinatorV1,
            (
                "The harmonic frame without Serpent: a codebook sized from config "
                "(--codebook-size, else --vocab-size), fixed 8-byte patches - a "
                "uniform resampling of byte time for a periodic latent, "
                "compute-matched to space patching's mean patch length - and a GDN "
                "compander in front of the quantizer."
            ),
        ),
        "abstractinator_v1_calm": Entry(
            AbstractinatorV1CALM,
            (
                "abstractinator_v1 with a continuous CALM arm beside the discrete "
                "codec: a patch VAE over the same patch features feeds z = z_q + gate "
                "* z_c into the trunk, with an energy head on the next VAE latent and "
                "a cross-entropy on the next code. Static patching is what aligns the "
                "two codecs' latents one per patch."
            ),
        ),
        "abstractinator_v2": Entry(
            AbstractinatorV2,
            (
                "abstractinator_v1 with plain RMS normalization in front of the "
                "quantizer instead of the GDN compander, since a fixed normalization "
                "leaves the commitment loss no scale to shrink, and with the byte path "
                "and the trunk each RMS-normalized before they are added, so neither "
                "can outgrow the other."
            ),
        ),
        "abstractinator_v2_additive": Entry(
            AbstractinatorV2Additive,
            (
                "abstractinator_v2 with BLT's plain sum of the byte path and the "
                "trunk, h = h_encoder + patch_embeds, and nothing holding the two on "
                "one scale."
            ),
        ),
        "calm": Entry(
            CALM,
            (
                "CALM (arXiv 2510.27688): a token-chunk VAE plus an energy head that "
                "predicts the next latent, at the paper's defaults and K=8 tokens per "
                "latent."
            ),
        ),
        "calm_small": Entry(
            CALMSmall,
            (
                "calm with its dimensions given as fractions of --hidden-size, so the "
                "encoder scales with the model. The smoke-test profile."
            ),
        ),
        "calm_byte": Entry(
            CALMByte,
            "calm at the byte-level K of 16 bytes per latent.",
        ),
        "calm_byte_small": Entry(
            CALMByteSmall,
            (
                "Byte-level CALM at K=8 with a smaller VAE sized from --hidden-size "
                "and the paper's energy-head sampling. Trains in two stages like the "
                "reference: the codec alone, with the KL annealed in, until its "
                "reconstruction plateaus (capped by ae_max_pretrain_steps), then the "
                "energy head against the frozen codec. Watch calm_recon_ce, "
                "calm_pretrain_flatness and calm_ae_frozen."
            ),
        ),
        "calm_byte_small_harmonic": Entry(
            CALMByteSmallHarmonic,
            (
                "calm_byte_small with harmonic codec dropout: the dropout rate becomes "
                "a standing-wave field over (patch position, channel), two cycles per "
                "axis."
            ),
        ),
        "calm_byte_ref": Entry(
            CALMByteRef,
            (
                "The published CALM repo's dimensions (at hidden_size 256: latent 128, "
                "AE hidden 512, noise 64, 4 head blocks, dropout 0.15) with one "
                "departure, a deeper residual codec (vae_depth=4), at the byte-level K "
                "of 16. The baseline the other CALM byte profiles vary."
            ),
        ),
        "calm_byte_flow": Entry(
            CALMByteFlow,
            (
                "calm_byte_ref with a flow-matching head in place of the energy head, "
                "whose dense, low-variance objective learns the conditional at small "
                "scale where the energy head does not, and K=4, where the codec "
                "manifold is a target the head can reach. The codec's pretraining "
                "stage is capped at 3000 steps."
            ),
        ),
        "calm_byte_harmonic": Entry(
            CALMByteHarmonic,
            (
                "calm_byte_flow with the harmonic latent head: the same flow-matching "
                "objective, run in a compact harmonic coefficient space so each next "
                "latent is a smooth low-frequency superposition. Fewer effective "
                "output dimensions lower the head's variance at small scale."
            ),
        ),
        "calm_byte_fixed": Entry(
            CALMByteFixed,
            (
                "calm_byte_flow with a fixed deterministic codec: the encoder is a "
                "frozen orthonormal byte transform and only the decoder learns. The "
                "latent target is stationary from step 0, so training is single-stage, "
                "with the head active immediately."
            ),
        ),
        "calm_byte_hybrid": Entry(
            CALMByteHybrid,
            (
                "calm_byte_fixed's static scaffold plus a small, never-frozen learned "
                "residual initialized at zero: the midpoint between the fixed codec "
                "and the VAE, starting identical to fixed and drifting toward a "
                "better-organized latent. Single-stage."
            ),
        ),
        "calm_byte_harmonic_codec": Entry(
            CALMByteHarmonicCodec,
            (
                "calm_byte_flow with a fixed codec over harmonic (standing-wave) bases "
                "instead of random orthonormal ones, and K=8: the separable 2D basis "
                "gives the patch-position axis its own frequency budget, so it absorbs "
                "longer patches. Deterministic encode, single-stage."
            ),
        ),
        "calm_byte_harmonic_serpent": Entry(
            CALMByteHarmonicSerpent,
            (
                "calm_byte_harmonic_codec with a learned Serpent nonlinearity after "
                "the transform: a learnable, non-stationary encode, still single-stage "
                "and never frozen."
            ),
        ),
        "calm_tm_ref": Entry(
            CALMTmRef,
            (
                "calm_byte_ref at K=4 tokens per latent, the reference's granularity "
                "for a TokenMonster or BPE tokenizer."
            ),
        ),
        "calm_bpe": Entry(
            CALMBpe,
            "calm at the BPE K of 4 tokens per latent.",
        ),
        "abstractinator_harmonic": Entry(
            AbstractinatorHarmonic,
            (
                "abstractinator_rvq in the CALM harmonic frame: patch latents rotate "
                "into the standing-wave basis, are RMS-normalized there, and the "
                "residual codes quantize harmonic amplitudes, with bottleneck_ratio "
                "0.5 keeping a low-frequency spectral budget. The base of every "
                "versioned profile."
            ),
            listed=False,
        ),
        "abstractinator_harmonic_serpent_avg": Entry(
            AbstractinatorV0Avg,
            "abstractinator_v0 with mean-pooled patches.",
            listed=False,
        ),
        "abstractinator_harmonic_serpent_vocab_bank": Entry(
            AbstractinatorV0Bank,
            "abstractinator_v0 with a codebook sized from config.",
            listed=False,
        ),
        "abstractinator_harmonic_vocab_bank": Entry(
            AbstractinatorHarmonicBank,
            "The harmonic frame with a codebook sized from config, space-patched.",
            listed=False,
        ),
        "abstractinator_harmonic_gdn_vocab_bank": Entry(
            AbstractinatorHarmonicGDNBank,
            (
                "The harmonic frame with a GDN compander and a codebook sized from "
                "config, space-patched."
            ),
            listed=False,
        ),
        "abstractinator": Alias("abstractinator_rvq"),
        "abstractinator_harmonic_serpent": Alias("abstractinator_v0"),
        "abstractinator_harmonic_gdn_vocab_bank_static": Alias("abstractinator_v1"),
        "abstractinator_harmonic_gdn_vocab_bank_static_calm": Alias(
            "abstractinator_v1_calm"
        ),
    },
)
