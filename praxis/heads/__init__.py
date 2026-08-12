from functools import partial
from typing import Optional

from praxis.heads.crystal import CrystalClassifier, CrystalHead, CrystalVearHead
from praxis.heads.forward import ForwardHead
from praxis.heads.halo import HaloClassifier, HaloHead
from praxis.heads.harmonic import HarmonicField, HarmonicHead
from praxis.heads.mtp import MTP_REGISTRY, MultiTokenPrediction
from praxis.heads.parallel import ParallelHead
from praxis.heads.stacked import SequentialHead
from praxis.heads.tied import TiedWeights


def _field(
    amp_modulation: str, build_classifier: bool = False, fast_weights: bool = False
):
    """A harmonic field builder; transform-only by default (no dead classifier),
    or terminal (its own linear readout) when ``build_classifier`` is set.
    ``fast_weights`` adds the bounded context-written overlay on the spectrum."""
    return partial(
        HarmonicHead,
        amp_modulation=amp_modulation,
        build_classifier=build_classifier,
        fast_weights=fast_weights,
    )


def _harmonic_crystal(amp_modulation: str) -> list:
    """Builders for a SequentialHead: the harmonic field (given modulation,
    transform-only so it allocates no dead classifier) feeding the crystal
    distance classifier."""
    return [_field(amp_modulation), CrystalHead]


def _prismatic2_branches() -> list:
    """The two prismatic2 arms: bias (learned field), variance (input field ->
    crystal). Shared by prismatic2 variants."""
    return [
        partial(SequentialHead, heads=[_field("learned", build_classifier=True)]),
        partial(SequentialHead, heads=[_field("input"), CrystalHead]),
    ]


def _prismatic3_branches() -> list:
    """The three prismatic3 arms: bias (learned field), variance (input field ->
    crystal), and a pure variance-only field. Each arm carries fast weights - a
    bounded test-time overlay on its spectrum foundation. Shared by all
    prismatic3 variants."""
    return [
        partial(
            SequentialHead,
            heads=[_field("learned", build_classifier=True, fast_weights=True)],
        ),
        partial(
            SequentialHead, heads=[_field("input", fast_weights=True), CrystalHead]
        ),
        partial(
            SequentialHead,
            heads=[_field("pure", build_classifier=True, fast_weights=True)],
        ),
    ]


def _prismatic4_branches() -> list:
    """prismatic3's arms, but the variance arm's single crystal becomes a
    VEAR-merged BANK of CrystalClassifiers: sharpened routing selects a discrete
    crystal per batch and inter-expert repulsion keeps the geometries unique (a
    population of output geometries instead of one). See CrystalVearHead."""
    return [
        partial(
            SequentialHead,
            heads=[_field("learned", build_classifier=True, fast_weights=True)],
        ),
        partial(
            SequentialHead,
            heads=[_field("input", fast_weights=True), CrystalVearHead],
        ),
        partial(
            SequentialHead,
            heads=[_field("pure", build_classifier=True, fast_weights=True)],
        ),
    ]


def _prismatic5_branches() -> list:
    """prismatic4's arms plus a HALO distance arm. The HALO head is a DIRECT
    branch (no harmonic field in front): HALOLoss scores the trunk embeddings,
    so the arm must score those same features at inference - putting a
    transform in front would train one feature space and score another (the
    mismatch that scrambled the borrowed-crystal wiring). Its logits are
    detached in the gate blend (see HaloHead.detach_in_blend), so the arm
    trains purely under HALO while the gate learns whether to trust it."""
    return _prismatic4_branches() + [HaloHead]


# Routing exponent for the prismatic6 crystal bank. 1.0 is SMEAR (the softmax
# untouched, every geometry contributes in proportion to its routing
# probability and every geometry receives gradient every step); the VEAR
# default of 4.0 sharpens toward a near-discrete pick. See CrystalVearHead's
# __init__ for the trade in both directions.
PRISMATIC6_SHARPEN: float = 1.0


def _prismatic6_branches(sharpen: Optional[float]) -> list:
    """Three arms over ONE shared field, all reading the same stem.

    prismatic2-5 give every arm its own HarmonicField, so three arms cost three
    field evaluations - measured at ~30% of total compute in abstractinator-j,
    against 53-69% dormant capacity in each of them. Here the field is the
    ParallelHead's ``stem``: evaluated once, shared by both readouts, so the
    arms differ only in HOW they read it. That is the distinction -j's gate
    actually rewarded (field->crystal 0.736, field->linear 0.234) against the
    separate-field bias arm it starved to 0.029.

    - Arm 0, GEOMETRIC: the crystal bank. ``sharpen`` decides whether the bank
      votes (VEAR) or blends (SMEAR).
    - Arm 1, DIRECT: a plain linear readout of the same field. The control that
      says whether the crystal geometry earns its cost.
    - Arm 2, HALO: ``reads_trunk`` keeps it on the raw hidden states rather
      than the stem, and it is now ATTACHED (``detach_in_blend=False``) where
      prismatic5 detached it. It is REQUIRED whenever ``loss_func: halo`` is
      set - ParallelHead.classifier looks for the ``is_halo`` arm to put
      HALOLoss in composite mode, and without one the loss silently falls back
      to its legacy side-loss path where the harmonic and gate machinery see
      almost no gradient.

      Why attach. Detaching bought a clean verdict on HALO's scoring function,
      and that verdict came back at 0.00125 gate share over 22k steps in
      abstractinator-j, never above its init - so the measurement is complete
      and the detachment now buys a number already known. Attached, CE reaches
      the arm and the question becomes whether it was CE-trainable all along.
      The cost is honest: a rising share can no longer distinguish "HALO's
      geometry is good" from "CE made this a decent CE head." Flip it back
      (``detach_in_blend=True``, or drop the kwarg) if halo_gamma runs away or
      halo_mean_radius drifts off halo_shell_radius, which is what CE pulling
      the geometric calibration would look like.
    """
    return [
        partial(CrystalVearHead, sharpen=sharpen),
        ForwardHead,
        partial(HaloHead, detach_in_blend=False),
    ]


HEAD_REGISTRY = dict(
    forward=ForwardHead,
    tied=TiedWeights,
    harmonic=HarmonicHead,
    crystal=CrystalHead,
    # Harmonic field feeding the crystal classifier, composed dynamically by
    # SequentialHead: bare grid (off) or a fixed single oscillation (static).
    crystal_harmonic=partial(SequentialHead, heads=_harmonic_crystal("off")),
    crystal_harmonic_static=partial(SequentialHead, heads=_harmonic_crystal("static")),
    # Prismatic: a top-level parallel split that makes the bias/variance axes two
    # physical branches. Branch 0 is a harmonic field (learned but static
    # envelope) read out by a plain linear head - the bias arm, a strong
    # structural prior. Branch 1 refracts an input-conditional field (its
    # envelope carries a per-sequence delta, identity at init) through the
    # crystal distance classifier - the variance arm, the expressive one. A
    # learned per-token gate weights the two logit streams, routing features to
    # whichever arm explains them. Each arm emits its own Bias/Variance Strands
    # card (#0 stays collapsed = bias; #1 separates as variance is learned):
    #   Parallel(Sequential(HarmonicField), Sequential(HarmonicField, CrystalClassifier))
    prismatic=partial(ParallelHead, branches=_prismatic2_branches()),
    # Prismatic + a third, variance-only arm: a "pure" field (no static
    # spectrum; the conditional delta alone, zero at init) with its own linear
    # readout - the mirror of the bias arm. Variance can arrive in bins the
    # bias arms never occupy; its strand card starts empty and grows pure red.
    prismatic3=partial(ParallelHead, branches=_prismatic3_branches()),
    # prismatic3 + level-repulsion on the gate: a pairwise log-gap penalty pushes
    # the three arms' mean weights to DISTINCT tiers (e.g. 70/20/10) and punishes
    # near-ties (70/15/15) where two arms become equally important. Watch the
    # "Parallel Gate Min Gap" card. Strength is baked here, not a config flag.
    prismatic3_repel=partial(
        ParallelHead, branches=_prismatic3_branches(), gate_repulsion=0.02
    ),
    # prismatic3 with the variance arm's crystal replaced by a VEAR-merged bank
    # of CrystalClassifiers (CRYSTAL_BANK_SIZE geometries): discrete, unique
    # output geometries voted per context. See CrystalVearHead, praxis/routers/vear.py.
    prismatic4=partial(ParallelHead, branches=_prismatic4_branches()),
    # prismatic4 + a fourth arm: the HALO hyperspherical distance classifier
    # (praxis/heads/halo.py). Pair with loss_func: halo - the loss detects the
    # arm and runs its honest composite mode (mixture CE for gate + other
    # arms, pure HALO geometry for this arm). The parallel gate-share card for
    # the last branch is the live verdict on whether HALO's scoring competes.
    prismatic5=partial(ParallelHead, branches=_prismatic5_branches()),
    # prismatic6: prismatic5's four arms cut to three, over ONE shared harmonic
    # field carried as the ParallelHead stem instead of one field per arm.
    # Reads as: Parallel(HarmonicField -> [CrystalSmearBank, Forward, Halo]).
    # Drops the separate bias arm (gate share 0.029 and falling in -j) and the
    # duplicate fields; keeps the two readouts the gate rewarded plus the HALO
    # arm that loss_func: halo requires. Crystal bank merges by SMEAR
    # (PRISMATIC6_SHARPEN = 1.0), so every geometry trains every step rather
    # than the bank voting for one. The HALO arm is ATTACHED here (CE reaches
    # it) rather than detached - the detached measurement is complete, so the
    # arm is given a chance to be useful instead of only being measured.
    prismatic6=partial(
        ParallelHead,
        stem=_field("input", fast_weights=True),
        branches=_prismatic6_branches(PRISMATIC6_SHARPEN),
    ),
    # The control for the one design choice prismatic6 changes beyond arm count:
    # identical, but the crystal bank keeps VEAR's discrete vote (sharpen=None
    # -> VEAR_SHARPEN). Run this against prismatic6 to attribute any delta to
    # the merge rather than to the shared stem.
    prismatic6_vear=partial(
        ParallelHead,
        stem=_field("input", fast_weights=True),
        branches=_prismatic6_branches(None),
    ),
)
