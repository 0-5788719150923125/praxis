from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.arc import ArcAttention
from praxis.attention.arc_ssog import ArcSSOGAttention
from praxis.attention.modular import ModularAttention
from praxis.attention.causal import CausalAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.infini import InfiniAttention
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.single import SingleHeadArcAttention
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
    # via the "warp" value sink at the first layer of the last recurrent pass
    # (heuristic: depth - num_layers; e.g. 2 layers x 4 loops -> step 6), so
    # the model leans on delayed context for that beat and the remaining
    # layers recorrect.
    "arc_dropoff": partial(ArcAttention, dropoff="warp"),
    # Arc with ONE head: shared Q/K representation, per-dimension affine
    # reads of it, and a SiLU output gate (Mega, arXiv:2209.10655) in place
    # of Arc's sigmoid. Overrides num_heads/num_queries to 1; head width is
    # head_size as usual. See praxis/attention/single.py.
    "arc_single": SingleHeadArcAttention,
    "arc_single_dropoff": partial(SingleHeadArcAttention, dropoff="warp"),
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
    # decayed at every depth for 11.8k steps. `arc_ssog` came back down to the
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
