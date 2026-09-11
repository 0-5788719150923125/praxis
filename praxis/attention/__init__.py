from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis import registry
from praxis.attention.arc import ArcAttention, ArcNoMemAttention
from praxis.attention.arc_ssog import ArcSSOGAttention
from praxis.attention.causal import CausalAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.infini import InfiniAttention
from praxis.attention.kaleidoscope import KaleidoscopeAttention
from praxis.attention.modular import ModularAttention
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.single import (
    SingleHeadArcAttention,
    SingleHeadArcNoMemAttention,
)
from praxis.attention.ssog import SSOGAttention
from praxis.attention.syntaxes import SyntaxesAttention
from praxis.registry import Entry

registry.declare(
    "attention",
    title="Attention mechanisms",
    doc=(
        (
            "Self-attention variants, from vanilla causal MHA to compressive-memory and "
            "per-depth-biased variants, plus the Q/K-free kaleidoscope and SSOG fields. "
            "Entries may be ``functools.partial`` profiles of one class; a profile is the "
            "variant, not a flag."
        )
    ),
    entries={
        "modular": ModularAttention,
        "vanilla": VanillaMHA,
        "pk": ProductKeyAttention,
        "syntaxes": SyntaxesAttention,
        "causal": CausalAttention,
        "infini": InfiniAttention,
        "arc": ArcAttention,
        "arc_dropoff": Entry(
            partial(ArcAttention, dropoff="warp"),
            (
                "Arc with the dropoff ablation: the ``warp`` value sink withholds the "
                "causal tip at step ``depth - num_layers``, so the model leans on "
                "delayed context for that beat and the remaining layers recorrect. "
                "Under KL halting the training depth budget is sampled, so this step "
                "is rarely reached; ``arc_dropoff_always`` is the arm that applies the "
                "ablation at a real rate. Training only. See CausalAttention.__init__."
            ),
        ),
        "arc_dropoff_always": Entry(
            partial(ArcAttention, dropoff="warp", dropoff_every=True),
            (
                "arc_dropoff with the sink at every recurrent pass. It fires about 44x "
                "as often as the single-pass schedule, which makes dropoff a real "
                "intervention rather than a rounding error. The cost is that the tip "
                "is absent from the value path at every depth, a recency prior rather "
                "than an ablation."
            ),
        ),
        "arc_single": Entry(
            SingleHeadArcAttention,
            (
                "Arc with one head: a shared Q/K representation, per-dimension affine "
                "reads of it, and a SiLU output gate (Mega, arXiv:2209.10655) in place "
                "of Arc's sigmoid. Overrides ``num_heads`` and ``num_queries`` to 1; "
                "head width is ``head_size`` as usual. See praxis/attention/single.py."
            ),
        ),
        "arc_single_dropoff": Entry(
            partial(SingleHeadArcAttention, dropoff="warp"),
            "arc_single with the single-pass ``warp`` dropoff sink of arc_dropoff.",
        ),
        "arc_nomem": Entry(
            ArcNoMemAttention,
            (
                "Arc with Infini's segment-level compressive memory removed "
                "(NoCompressiveMemory), so attention sees the whole sequence in one "
                "flex call. Per-depth biases, ghostmax, ArcHoPE, the output gate and "
                "the dropoff ablation are inherited unchanged, so an A/B against "
                "``arc`` isolates the memory and nothing else. The memory branch is "
                "identically zero where a sequence fits in one segment, and costs "
                "wall-clock that grows with length where it spans several, since the "
                "segment loop is serial Python. On the memory arm, "
                "``attn_memory_share`` at its 0.5 init says the blend is not using it."
            ),
        ),
        "arc_single_nomem": Entry(
            SingleHeadArcNoMemAttention,
            (
                "arc_single without the compressive memory; an A/B against "
                "``arc_single`` isolates the memory."
            ),
        ),
        "arc_single_dropoff_nomem": Entry(
            partial(SingleHeadArcNoMemAttention, dropoff="warp"),
            (
                "arc_single_dropoff without the compressive memory; an A/B against "
                "``arc_single_dropoff`` isolates the memory."
            ),
        ),
        "arc_single_dropoff_always_nomem": Entry(
            partial(SingleHeadArcNoMemAttention, dropoff="warp", dropoff_every=True),
            (
                "arc_single_dropoff_nomem with the sink at every recurrent pass, so "
                "the every-pass schedule can be compared on the single-head, "
                "memory-free profile and not only on ``arc``."
            ),
        ),
        "kaleido": Entry(
            KaleidoscopeAttention,
            (
                "Kaleidoscope: N frozen ``[T, T]`` mixing matrices, blended per token "
                "by a router, with a per-depth rank-1 deformation on the mirrors "
                "themselves. There is no Q or K - the matrix is the parameter, so "
                "there is nothing to project from. Synthesizer (arXiv:2005.00743) "
                "covers one frozen matrix, one trained matrix, and N mixed by static "
                "learned scalars; an input-conditional blend is the cell it leaves "
                "open. See praxis/attention/kaleidoscope.py for why the mix is "
                "pre-softmax and why the per-depth bias goes on the mirrors rather "
                "than on the inputs."
            ),
        ),
        "kaleido_dropoff": Entry(
            partial(KaleidoscopeAttention, dropoff="warp"),
            (
                "kaleido with the ``warp`` dropoff sink at the first layer of the last "
                "recurrent pass, the schedule ``arc_dropoff`` runs. Only ``warp``: the "
                "``shift`` mode moves K as well as V, and there is no K here to move."
            ),
        ),
        "kaleido_dropoff_always": Entry(
            partial(KaleidoscopeAttention, dropoff="warp", dropoff_every=True),
            (
                "kaleido with the ``warp`` dropoff sink at every recurrent pass, like "
                "ghostmax. See CausalAttention.__init__ for the argument on both sides "
                "of the schedule."
            ),
        ),
        "kaleido_pink": Entry(
            partial(KaleidoscopeAttention, alpha=1.0),
            (
                "kaleido with a 1/k^alpha envelope (alpha 1) over the dictionary, the "
                "pink-noise prior HarmonicField puts on its frequency grid. The flat "
                "dictionary is the alpha=0 corner of the paper's interference-capacity "
                "proposition; this is the corner where the prior costs capacity unless "
                "the blend spends amplitude against it, which "
                "``kaleido_envelope_fight`` measures. Same seed and same draw as "
                "``kaleido``, so the A/B isolates the envelope."
            ),
        ),
        "kaleido_pink_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention, alpha=1.0, dropoff="warp", dropoff_every=True
            ),
            "kaleido_pink with the ``warp`` dropoff sink at every recurrent pass.",
        ),
        "kaleido_split": Entry(
            partial(KaleidoscopeAttention, coords="split"),
            (
                "kaleido with the dictionary split across two coordinate systems: half "
                "in (query fraction, key fraction), half in (query fraction, log lag). "
                'The ratio half spans "attend to the start"; the lag half spans '
                '"attend one token back" and resolves lag 1 to width 1 at any length, '
                "where the ratio half smears it across 16 positions. "
                "``kaleido_lag_share`` says whether the lag half earns its place (0.5 "
                "is parity). The two systems address the same positions unless T well "
                "exceeds MIRROR_RES, so the comparison needs long sequences. "
                "Ratio-only stays the default because its uniform grid is exactly "
                "scale-equivariant."
            ),
        ),
        "kaleido_split_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                coords="split",
                dropoff="warp",
                dropoff_every=True,
            ),
            "kaleido_split with the ``warp`` dropoff sink at every recurrent pass.",
        ),
        "kaleido_pink_split_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                alpha=1.0,
                coords="split",
                dropoff="warp",
                dropoff_every=True,
            ),
            (
                "kaleido_split with the pink envelope of ``kaleido_pink`` (ranked "
                "within each coordinate group) and the ``warp`` dropoff sink at every "
                "recurrent pass."
            ),
        ),
        "kaleido_12_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                num_mirrors=12,
                dropoff="warp",
                dropoff_every=True,
            ),
            (
                "kaleido_dropoff_always with a 12-mirror dictionary. With no Q/K, N "
                "frozen patterns and a signed router over them are all the expressive "
                "capacity the block has, and the default 4 is very few. The merge is "
                "linear in N and the block is launch-overhead-bound at small widths, "
                "so the wider dictionary costs little step time. Read "
                "``kaleido_turn_modes`` against N: it separates a router that "
                "saturates at 2-3 patterns however many it is offered from four iid "
                "draws being redundant."
            ),
        ),
        "kaleido_24_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                num_mirrors=24,
                dropoff="warp",
                dropoff_every=True,
            ),
            "kaleido_12_dropoff_always at 24 mirrors.",
        ),
        "kaleido_zoom_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention, zoom=True, dropoff="warp", dropoff_every=True
            ),
            (
                "kaleido_dropoff_always with a per-mirror spatial zoom: the ratio "
                "mirrors are read at a ladder of harmonic and sub-harmonic zoom "
                "factors centred on 1 (1/2, 1, 2, 3 at N=4), so the dictionary spans "
                "coarser and finer relative-position periods. See ``zoom_ladder`` for "
                "why granularity and not amplitude. The ladder derives from the group "
                "size, so a wider dictionary buys finer rungs rather than more draws. "
                "This is the redundancy fix, not the capacity fix: N iid draws from "
                "one distribution stay redundant however many are taken. "
                "``kaleido_zoom_mean`` says which rung the router buys, as a 0-1 "
                "position comparable across N."
            ),
        ),
        "kaleido_12_zoom_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                num_mirrors=12,
                zoom=True,
                dropoff="warp",
                dropoff_every=True,
            ),
            (
                "kaleido_zoom_dropoff_always at 12 mirrors, spanning zoom 1/6 to 7. "
                "With a derived ladder, dictionary width and granularity are one axis, "
                'so this tests "more dictionary" and "more granularity" as the single '
                "thing they are."
            ),
        ),
        "kaleido_norm_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention, mix_norm=True, dropoff="warp", dropoff_every=True
            ),
            (
                "kaleido_dropoff_always with 1/sqrt(N) on the mix. ``scores`` sums N "
                "mirror terms with no normalization, so at equal per-mirror router "
                "magnitude a wider dictionary opens sharper, not richer: effective "
                "attention support falls from 181 to 65 to 25 positions going N = 4, "
                "12, 24 at T=257. The scale also divides the per-token modulation "
                "ceiling, so this is the N=4 matched control, and an N sweep needs it "
                "on in every arm or in none."
            ),
        ),
        "kaleido_12_norm_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                num_mirrors=12,
                mix_norm=True,
                dropoff="warp",
                dropoff_every=True,
            ),
            (
                "kaleido_12_dropoff_always with the 1/sqrt(N) mix scale of "
                "``kaleido_norm_dropoff_always``."
            ),
        ),
        "kaleido_24_norm_dropoff_always": Entry(
            partial(
                KaleidoscopeAttention,
                num_mirrors=24,
                mix_norm=True,
                dropoff="warp",
                dropoff_every=True,
            ),
            (
                "kaleido_24_dropoff_always with the 1/sqrt(N) mix scale of "
                "``kaleido_norm_dropoff_always``."
            ),
        ),
        "ssog": Entry(
            SSOGAttention,
            (
                "Query-steered sum-of-Gaussians field over causal lag, with no Q/K "
                "(Pisoni's SSOG, ported to 1D). Position-addressed only; see "
                "praxis/attention/ssog.py."
            ),
        ),
        "arc_ssog": Entry(
            ArcSSOGAttention,
            (
                "The SSOG field with a per-depth axis, a warm steering gate and a "
                "small populated atom bank - the reference's own per-layer geometry, "
                "which a depth-shared field cannot express. ``ssog`` is the faithful "
                "port; this is the variant to modify. See praxis/attention/arc_ssog.py "
                "for the reason behind each deviation."
            ),
        ),
        "arc_ssog_wide": Entry(
            partial(ArcSSOGAttention, num_atoms=12, mu_init_max=128.0),
            (
                "arc_ssog with twelve atoms over lag 0.5 to 128 instead of four over "
                "0.5 to 32. Attention weights are the mixture normalized over causal "
                "keys, so twelve atoms dilute each other to about 0.083 per atom "
                "against 0.25, and atoms centred beyond the live window are truncated "
                "and renormalized onto the oldest tokens. It keeps the bank-size and "
                "ladder-span question measurable against the small bank."
            ),
        ),
        "arc_ssog_null": Entry(
            partial(ArcSSOGAttention, null_atom=True),
            (
                "arc_ssog plus a per-depth null atom: one learned logit per pass whose "
                "value is zero, so a query can decline to contribute. Nothing below "
                "the head knows absolute position - the logit is a function of lag "
                "alone - so without it a query near the start has an atom's truncated "
                "tail renormalized onto the oldest token, a sink that looks exactly "
                "like a real long-range read. See ArcSSOGAttention._apply_null."
            ),
        ),
    },
)


def patch_attention_config(config: Any, args: Any = None) -> None:
    """Let the selected attention mechanism correct the config describing it.

    A mechanism that does not build what the config asks for has to say so
    here, because the config is not a private argument list: it is serialized
    to ``config.json``, rendered in the blueprint tab, and read by every other
    module in the stack. ``arc_single`` runs ONE head whatever ``num_heads``
    says - so without this the config would advertise a head count no module
    ever built. Every rewrite is printed, because the failure mode this exists
    to prevent is precisely a number changing where nobody can see it.

    Pass ``args`` (the parsed CLI namespace) to carry the same correction back
    there. It is a SECOND record of the same numbers, not a copy of the config:
    the Arguments card on the Architecture tab serializes the namespace
    directly (praxis/web/spec_data.py::_serialise_args), so a config-only fix
    leaves that card reporting the head count the run did not use. Only keys
    the patch actually changed are written back, and only when the namespace
    already carries them, so nothing is invented on it. The run hash is
    computed from ``sys.argv`` rather than the namespace
    (praxis/cli/core/logger.py::log_command), so this cannot move a run's
    directory out from under a resume.

    A no-op for every mechanism that builds exactly what it was asked for,
    which is all of them but one. Registry entries may be ``functools.partial``
    profiles, so the hook is read off the underlying class.
    """
    entry = registry.namespace("attention").get(getattr(config, "attention_type", None))
    if entry is None:
        return
    patch = getattr(getattr(entry, "func", entry), "patch_config", None)
    if patch is None:
        return

    # Diff rather than naming fields, so a future patch_config that corrects
    # some other knob is reported and mirrored without editing this function.
    before = dict(vars(config))
    patch(config)
    changed = {
        key: (old, getattr(config, key, old))
        for key, old in before.items()
        if getattr(config, key, old) != old
    }
    if not changed:
        return

    # Say so out loud. A silent rewrite is how a config ends up describing a
    # model nobody built - and how an edit that looks like a no-op turns out
    # not to be one.
    summary = ", ".join(f"{k} {o!r} -> {n!r}" for k, (o, n) in sorted(changed.items()))
    print(f"[CONFIG] {config.attention_type} overrides {summary}")

    if args is not None:
        for key, (_, new) in changed.items():
            if hasattr(args, key):
                setattr(args, key, new)
