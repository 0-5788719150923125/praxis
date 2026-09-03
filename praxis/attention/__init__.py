from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.arc import ArcAttention, ArcNoMemAttention
from praxis.attention.arc_ssog import ArcSSOGAttention
from praxis.attention.modular import ModularAttention
from praxis.attention.causal import CausalAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.infini import InfiniAttention
from praxis.attention.kaleidoscope import KaleidoscopeAttention
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.single import (
    SingleHeadArcAttention,
    SingleHeadArcNoMemAttention,
)
from praxis.attention.ssog import SSOGAttention
from praxis.attention.syntaxes import SyntaxesAttention

# Registry of available attention mechanisms
ATTENTION_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "modular": ModularAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
    "syntaxes": SyntaxesAttention,
    "causal": CausalAttention,
    "infini": InfiniAttention,
    "arc": ArcAttention,
    # Arc + the dropoff ablation (next/dropoff.md): withhold the causal tip
    # via the "warp" value sink at step ``depth - num_layers``, so the model
    # leans on delayed context for that beat and the remaining layers
    # recorrect. NOTE this schedule is very nearly inert under KL halting,
    # whose TRAINING depth budget is sampled: measured 0.07 firings per
    # step at depth 6, against 3.06 for the _always sibling. Prefer the ``_always`` sibling for an arm that actually
    # applies the ablation. See CausalAttention.__init__.
    "arc_dropoff": partial(ArcAttention, dropoff="warp"),
    # The same sink at EVERY pass. Not a stylistic variant of the entry above:
    # it is ~44x the exposure (3.06 firings per step against 0.07), and
    # the first schedule under which dropoff is a real intervention rather than
    # a rounding error.
    "arc_dropoff_always": partial(ArcAttention, dropoff="warp", dropoff_every=True),
    # Arc with ONE head: shared Q/K representation, per-dimension affine
    # reads of it, and a SiLU output gate (Mega, arXiv:2209.10655) in place
    # of Arc's sigmoid. Overrides num_heads/num_queries to 1; head width is
    # head_size as usual. See praxis/attention/single.py.
    "arc_single": SingleHeadArcAttention,
    "arc_single_dropoff": partial(SingleHeadArcAttention, dropoff="warp"),
    # The same three, with Infini's segment-level compressive memory removed
    # (praxis.attention.infini.NoCompressiveMemory). Attention sees the whole
    # sequence in one flex call; the per-depth biases, ghostmax, ArcHoPE, the
    # SiLU gate and the dropoff ablation are all inherited unchanged, so an
    # A/B against the entry above it isolates the memory and nothing else.
    #
    # The memory is not free and, measurably, has not been earning it. On
    # abstractinator-t at step 450 `attn_memory_share` read 0.4999 over a
    # [0.4998, 0.5] range - the blend never left its zero init, which with a
    # sequence that fits in one segment means half the attention output was
    # multiplied by a branch that is identically zero. Where the sequence DOES
    # span segments the cost is wall-clock: measured at beta.yml's dimensions
    # (hidden 96, depth 6, batch 16, fused flex, fwd+bwd), `arc` at
    # window_size 64 / T 256 ran 178 ms/step against 84 ms with the memory
    # inert and 48 ms for plain `causal`, and the gap WIDENS with length
    # because the segment loop is serial Python.
    "arc_nomem": ArcNoMemAttention,
    "arc_single_nomem": SingleHeadArcNoMemAttention,
    "arc_single_dropoff_nomem": partial(SingleHeadArcNoMemAttention, dropoff="warp"),
    # The every-pass schedule on the profile the abstractinator line actually
    # runs, so the schedule question is answerable there and not only on arc.
    "arc_single_dropoff_always_nomem": partial(
        SingleHeadArcNoMemAttention, dropoff="warp", dropoff_every=True
    ),
    # Kaleidoscope: N frozen [T, T] mixing matrices, blended per TOKEN by a
    # router, with a per-depth rank-1 deformation on the mirrors themselves.
    # No Q, no K - the matrix is the parameter, so there is nothing to project
    # from. Synthesizer (arXiv:2005.00743) covered every neighbouring cell
    # (one frozen matrix, one trained matrix, N mixed by STATIC learned
    # scalars); an input-conditional blend is the one it left empty. See
    # praxis/attention/kaleidoscope.py for why the mix is pre-softmax and why
    # the per-depth bias goes on the mirrors rather than on the inputs.
    "kaleido": KaleidoscopeAttention,
    # ... plus the dropoff ablation, the same "warp" value sink at the first
    # layer of the last recurrent pass that `arc_dropoff` runs. Only `warp`:
    # the `shift` mode moves K as well as V and there is no K here to move.
    "kaleido_dropoff": partial(KaleidoscopeAttention, dropoff="warp"),
    # ... at every pass, like ghostmax. See CausalAttention.__init__ for the
    # argument on both sides of the schedule.
    "kaleido_dropoff_always": partial(
        KaleidoscopeAttention, dropoff="warp", dropoff_every=True
    ),
    # Query-steered Gaussian field over causal lag, no Q/K (Pisoni's SSOG,
    # ported to 1D). Position-addressed only; see praxis/attention/ssog.py.
    "ssog": SSOGAttention,
    # The same field with a PER-DEPTH axis, a warm steering gate and a
    # populated atom bank - the reference's own per-layer geometry, which a
    # depth-shared field cannot express. `ssog` stays the faithful port; this
    # is the one we hack on. See praxis/attention/arc_ssog.py for why each
    # deviation exists (every one of them is a measurement off -r).
    "arc_ssog": ArcSSOGAttention,
    # The bank is a profile, not a flag: it IS the variant. This is exactly
    # what -r ran from 2026-08-18, kept reproducible rather than deleted -
    # twelve atoms over 0.5..128 diluted the softmax (0.083 per atom against
    # 0.25, with three atoms centred outside the ×1 window) and `far_mass`
    # decayed at every depth for 11.8k steps (measured with the centre-indicator
    # far_mass; see _tail_mass for why those numbers do not compare to today's).
    # `arc_ssog` came back down to the
    # faithful ladder; this stays so the measurement can be repeated.
    "arc_ssog_wide": partial(ArcSSOGAttention, num_atoms=12, mu_init_max=128.0),
    # ... plus a per-depth NULL ATOM: one learned logit per pass whose "value"
    # is zero, so a query can decline to contribute. It matters here because
    # nothing below the head knows absolute position - the logit is a function
    # of lag alone - so without it a query near the start has an atom's
    # truncated tail renormalised onto the oldest token, a sink that looks
    # exactly like a real long-range read. See ArcSSOGAttention._apply_null.
    "arc_ssog_null": partial(ArcSSOGAttention, null_atom=True),
}


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
    entry = ATTENTION_REGISTRY.get(getattr(config, "attention_type", None))
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
