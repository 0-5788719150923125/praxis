from functools import partial
from typing import Optional

from praxis import registry
from praxis.classifiers.crystal import (
    CrystalClassifier,
    CrystalGeometry,
    CrystalSmearClassifier,
    CrystalVearClassifier,
)
from praxis.classifiers.halo import HaloClassifier, HaloGeometry
from praxis.classifiers.harmonic import HarmonicClassifier, HarmonicField
from praxis.classifiers.linear import LinearClassifier
from praxis.classifiers.mtp import MultiTokenPrediction
from praxis.classifiers.parallel import ParallelClassifier, SurgicalParallelClassifier
from praxis.classifiers.sequential import SequentialClassifier
from praxis.classifiers.tied import TiedClassifier
from praxis.registry import Entry


def _field(amp_modulation: str, build_scorer: bool = False, fast_weights: bool = False):
    """A harmonic field builder; transform-only by default (no dead scorer),
    or terminal (its own linear readout) when ``build_scorer`` is set.
    ``fast_weights`` adds the bounded context-written overlay on the spectrum."""
    return partial(
        HarmonicClassifier,
        amp_modulation=amp_modulation,
        build_scorer=build_scorer,
        fast_weights=fast_weights,
    )


def _harmonic_crystal(amp_modulation: str) -> list:
    """Builders for a SequentialClassifier: the harmonic field (given modulation,
    transform-only so it allocates no dead scorer) feeding the crystal
    distance classifier."""
    return [_field(amp_modulation), CrystalClassifier]


def _prismatic2_branches() -> list:
    """The two prismatic2 arms: bias (learned field), variance (input field ->
    crystal). Shared by prismatic2 variants."""
    return [
        partial(SequentialClassifier, stages=[_field("learned", build_scorer=True)]),
        partial(SequentialClassifier, stages=[_field("input"), CrystalClassifier]),
    ]


def _prismatic3_branches() -> list:
    """The three prismatic3 arms: bias (learned field), variance (input field ->
    crystal), and a pure variance-only field. Each arm carries fast weights - a
    bounded test-time overlay on its spectrum foundation. Shared by all
    prismatic3 variants."""
    return [
        partial(
            SequentialClassifier,
            stages=[_field("learned", build_scorer=True, fast_weights=True)],
        ),
        partial(
            SequentialClassifier,
            stages=[_field("input", fast_weights=True), CrystalClassifier],
        ),
        partial(
            SequentialClassifier,
            stages=[_field("pure", build_scorer=True, fast_weights=True)],
        ),
    ]


def _prismatic4_branches() -> list:
    """prismatic3's arms, but the variance arm's single crystal becomes a
    VEAR-merged BANK of crystal geometries: sharpened routing selects a discrete
    crystal per batch and inter-expert repulsion keeps the geometries unique (a
    population of output geometries instead of one). See CrystalVearClassifier."""
    return [
        partial(
            SequentialClassifier,
            stages=[_field("learned", build_scorer=True, fast_weights=True)],
        ),
        partial(
            SequentialClassifier,
            stages=[_field("input", fast_weights=True), CrystalVearClassifier],
        ),
        partial(
            SequentialClassifier,
            stages=[_field("pure", build_scorer=True, fast_weights=True)],
        ),
    ]


def _prismatic5_branches() -> list:
    """prismatic4's arms plus a HALO distance arm. The HALO classifier is a DIRECT
    branch (no harmonic field in front): HALOLoss scores the trunk embeddings,
    so the arm must score those same features at inference - putting a
    transform in front would train one feature space and score another (the
    mismatch that scrambled the borrowed-crystal wiring). Its logits are
    detached in the gate blend (see HaloClassifier.detach_in_blend), so the arm
    trains purely under HALO while the gate learns whether to trust it."""
    return _prismatic4_branches() + [HaloClassifier]


# Routing exponent for the prismatic6 crystal bank. 1.0 is SMEAR (the softmax
# untouched, every geometry contributes in proportion to its routing
# probability and every geometry receives gradient every step); the VEAR
# default of 4.0 sharpens toward a near-discrete pick. See CrystalVearClassifier's
# __init__ for the trade in both directions.
PRISMATIC6_SHARPEN: float = 1.0


def _prismatic6_branches(sharpen: Optional[float]) -> list:
    """Three arms over ONE shared field, all reading the same stem.

    prismatic2-5 give every arm its own HarmonicField, so three arms cost three
    field evaluations - ~30% of total compute in abstractinator-j, against 53-69%
    dormant capacity in each. Here the field is the ParallelClassifier's ``stem``:
    evaluated once, shared by both readouts, so the arms differ only in HOW they
    read it. That is the distinction -j's gate actually rewarded (field->crystal
    0.736, field->linear 0.234) against the separate-field bias arm it starved to
    0.029.

    - Arm 0, GEOMETRIC: the crystal bank. ``sharpen`` decides whether the bank
      votes (VEAR) or blends (SMEAR).
    - Arm 1, DIRECT: a plain linear readout of the same field. The control that
      says whether the crystal geometry earns its cost.
    - Arm 2, HALO: ``reads_trunk`` keeps it on the raw hidden states rather than
      the stem, and it is ATTACHED (``detach_in_blend=False``). Required whenever
      ``loss_func: halo`` is set - ParallelClassifier.scorer looks for the ``is_halo``
      arm to put HALOLoss in composite mode, and without one the loss silently
      falls back to its legacy side-loss path. See ``HaloClassifier`` for what attaching
      trades away and what flips it back.
    """
    return [
        partial(CrystalVearClassifier, sharpen=sharpen),
        LinearClassifier,
        partial(HaloClassifier, detach_in_blend=False),
    ]


def _prismatic7_branches() -> list:
    """prismatic6's three arms, with arm 0's bank merged per example.

    Only the geometric arm changes. The direct readout and the HALO arm are
    byte-identical to prismatic6, so a prismatic6 -> prismatic7 comparison
    attributes any delta to the merge and to nothing else.
    """
    return [
        CrystalSmearClassifier,
        LinearClassifier,
        partial(HaloClassifier, detach_in_blend=False),
    ]


def _prismatic8_branches() -> list:
    """prismatic7's arms with the crystal bank replaced by ONE crystal.

    No bank, no router, no experts on the geometric arm - a plain
    ``CrystalClassifier`` reading the shared stem, exactly as prismatic2/3 had it.

    THE ARGUMENT IS ABOUT WHERE DECISIONS BELONG. By the time features reach
    the classifier they have already been routed at every level the trunk
    offers: depth, residual, attention, memory. A classifier that also picks
    its own output geometry per example makes the last stage of the model the
    least predictable one, and predictability is the whole reason to put a
    distance classifier there. The three arms already give the classifier its
    choice; the arms themselves should be fixed functions.

    THE MEASUREMENT THAT SUPPORTS IT (abstractinator-m, step 17343). The
    routed bank did not perturb one geometry, it grew four unequal and
    partly degenerate ones: LoRA deviations at 0.36x / 2.61x / 1.27x / 0.85x
    the base's Frobenius norm, and per-expert effective_dim (90% variance) of
    13 / 4 / 9 / 21 against the base's 23. Expert 1 had collapsed to four
    dimensions with 84% of its variance in the top two PCs. That is not the
    crystal geometry the harmonic-loss result is about, and averaging four of
    them per example is not a way back to it.

    Two things follow for free. ``causal_readout`` returns to True on this arm
    (the bank pooled the sequence to route, which forced the speculative
    decoder to re-encode a row per candidate), and the arm's PCA card is a
    single panel fitting its own frame again - the pre-bank view, rather than
    four panels sharing a frame that the largest deviation stretches.

    The other two arms are byte-identical to prismatic7, so a prismatic7 ->
    prismatic8 comparison attributes any delta to the bank and nothing else.
    """
    return [
        CrystalClassifier,
        LinearClassifier,
        partial(HaloClassifier, detach_in_blend=False),
    ]


registry.declare(
    "classifiers",
    title="Classifiers",
    doc=(
        (
            "Classifiers map features to logits over a discrete vocabulary: untied and "
            "tied linear classifiers, the harmonic field, the crystal distance "
            "classifier, and the prismatic compositions of them, whose gate-combined "
            "parallel arms split the bias and variance axes. Multi-token prediction "
            "modules are a separate registry, ``mtp``; continuous-output modules live "
            "in ``generators``."
        )
    ),
    entries={
        "forward": LinearClassifier,
        "tied": TiedClassifier,
        "harmonic": HarmonicClassifier,
        "crystal": CrystalClassifier,
        "crystal_harmonic": Entry(
            partial(SequentialClassifier, stages=_harmonic_crystal("off")),
            (
                "A harmonic field feeding the crystal distance classifier, composed by "
                'SequentialClassifier. The field is the bare grid (``amp_modulation="off"``) '
                "and is transform-only, so it allocates no dead scorer."
            ),
        ),
        "crystal_harmonic_static": Entry(
            partial(SequentialClassifier, stages=_harmonic_crystal("static")),
            (
                "crystal_harmonic with a fixed single oscillation on the field's "
                'envelope (``amp_modulation="static"``).'
            ),
        ),
        "prismatic": Entry(
            partial(ParallelClassifier, branches=_prismatic2_branches()),
            (
                "A top-level parallel split that makes the bias and variance axes two "
                "physical branches. Branch 0 is a harmonic field with a learned but "
                "static envelope, read out by a plain linear classifier: the bias arm, a "
                "strong structural prior. Branch 1 refracts an input-conditional field "
                "(a per-sequence envelope delta, identity at init) through the crystal "
                "distance classifier: the variance arm, the expressive one. A learned "
                "per-token gate weights the two logit streams. Each arm emits its own "
                "Bias/Variance Strands card; #0 stays collapsed and #1 separates as "
                "variance is learned."
            ),
        ),
        "prismatic3": Entry(
            partial(ParallelClassifier, branches=_prismatic3_branches()),
            (
                "prismatic plus a third, variance-only arm: a ``pure`` field (no "
                "static spectrum; the conditional delta alone, zero at init) with its "
                "own linear readout, the mirror of the bias arm. Every arm carries "
                "fast weights, a bounded test-time overlay on its spectrum. Variance "
                "can arrive in bins the bias arms never occupy; the third arm's strand "
                "card starts empty and grows pure red."
            ),
        ),
        "prismatic3_repel": Entry(
            partial(
                ParallelClassifier, branches=_prismatic3_branches(), gate_repulsion=0.02
            ),
            (
                "prismatic3 with level repulsion on the gate: a pairwise log-gap "
                "penalty (strength 0.02, fixed here rather than a flag) pushes the "
                "three arms' mean weights to distinct tiers such as 70/20/10 and "
                "punishes near-ties such as 70/15/15, where two arms become equally "
                'important. Watch the "Parallel Gate Min Gap" card.'
            ),
        ),
        "prismatic4": Entry(
            partial(ParallelClassifier, branches=_prismatic4_branches()),
            (
                "prismatic3 with a VEAR-merged bank of crystal geometries "
                "(CRYSTAL_BANK_SIZE geometries) in place of the variance arm's single "
                "crystal: sharpened routing votes a discrete output geometry per "
                "context, and inter-expert repulsion keeps the geometries distinct. "
                "See CrystalVearClassifier and praxis/routers/vear.py."
            ),
        ),
        "prismatic5": Entry(
            partial(ParallelClassifier, branches=_prismatic5_branches()),
            (
                "prismatic4 plus a fourth arm: the HALO hyperspherical distance "
                "classifier (praxis/classifiers/halo.py), a direct branch on the trunk "
                "features whose logits are detached in the gate blend, so it trains "
                "purely under HALO while the gate learns whether to trust it. Pair "
                "with ``loss_func: halo``: the loss detects the arm and runs its "
                "composite mode (mixture CE for the gate and the other arms, pure HALO "
                "geometry for this one). The last branch's gate-share card says "
                "whether HALO's scoring competes."
            ),
        ),
        "prismatic6": Entry(
            partial(
                ParallelClassifier,
                stem=_field("input", fast_weights=True),
                branches=_prismatic6_branches(PRISMATIC6_SHARPEN),
            ),
            (
                "prismatic5's four arms cut to three over ONE shared harmonic field, "
                "carried as the ParallelClassifier stem and evaluated once: "
                "Parallel(HarmonicField -> [crystal bank, linear, HALO]). The arms "
                "differ only in how they read the field. The plain linear readout is "
                "the control that says whether the crystal geometry earns its cost, "
                "and the HALO arm reads the raw trunk, attached so cross-entropy "
                "reaches it; ``loss_func: halo`` requires it. The bank merges by SMEAR "
                "(PRISMATIC6_SHARPEN = 1.0), so every geometry trains every step "
                "rather than the bank voting for one."
            ),
        ),
        "prismatic6_vear": Entry(
            partial(
                ParallelClassifier,
                stem=_field("input", fast_weights=True),
                branches=_prismatic6_branches(None),
            ),
            (
                "prismatic6 with the crystal bank keeping VEAR's discrete vote "
                "(``sharpen=None``, so VEAR_SHARPEN). The control for the one choice "
                "prismatic6 makes beyond arm count: against prismatic6 it attributes "
                "any delta to the merge rather than to the shared stem."
            ),
        ),
        "prismatic7": Entry(
            partial(
                ParallelClassifier,
                stem=_field("input", fast_weights=True),
                branches=_prismatic7_branches(),
            ),
            (
                "prismatic6's arms with the crystal bank merged the way the SMEAR "
                "paper merges: per example rather than on the batch mean, over one "
                "shared geometry plus low-rank deviations rather than N independent "
                "center sets (CrystalSmearClassifier). A batch-mean merge makes a constant "
                "router the design's fixed point, which ``smear_input_dependence`` "
                "near 0 shows. Identical to prismatic6 at initialization (LoRA init) "
                "and in its other two arms, so the swap is a clean A/B."
            ),
        ),
        "prismatic8": Entry(
            partial(
                ParallelClassifier,
                stem=_field("input", fast_weights=True),
                branches=_prismatic8_branches(),
            ),
            (
                "prismatic7 with the crystal bank collapsed to a single "
                "CrystalClassifier reading the shared stem. The trunk has already "
                "routed features at every level it offers - depth, residual, "
                "attention, memory - so a classifier that also picks its own geometry "
                "per example makes the last stage the least predictable one, which "
                "defeats the point of a distance classifier. The three arms give the "
                "classifier its choice; each arm is a fixed function of the stem, and "
                "``causal_readout`` holds on the geometric arm. The other two arms "
                "match prismatic7, so the comparison isolates the bank."
            ),
        ),
        "prismatic9": Entry(
            partial(
                SurgicalParallelClassifier,
                stem=_field("input", fast_weights=True),
                branches=_prismatic8_branches(),
            ),
            (
                "prismatic8's three arms, trained differently: each arm gets its own "
                "cross-entropy instead of the mixture's residual, and the trunk "
                "receives one PCGrad-combined gradient over those three objectives "
                "rather than their plain sum. The blend is unchanged and still makes "
                "every prediction; it stops deciding how much each arm is trained. The "
                "cost is the division of labour: arms trained alone all learn the "
                "whole task, and val NLL is where that shows. See SurgicalParallelClassifier "
                "and _pcgrad in praxis/classifiers/parallel.py."
            ),
        ),
        "prismatic10": Entry(
            partial(
                SurgicalParallelClassifier,
                branches=[CrystalClassifier, CrystalClassifier],
            ),
            (
                "Two CrystalClassifiers reading the trunk directly: no harmonic stem "
                "and no HALO arm, trained as prismatic9 trains its arms. Each arm "
                "learns from its own harmonic cross-entropy, the trunk from their "
                "PCGrad-combined gradient, and the gate from the mixture. The arms "
                "differ only by initialization, so the gate's split and "
                "arm_cos_01 say whether two geometries are worth more than one."
            ),
        ),
    },
)
