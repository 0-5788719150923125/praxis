from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.arc import ArcAttention
from praxis.attention.modular import ModularAttention
from praxis.attention.causal import CausalAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.infini import InfiniAttention
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.single import SingleHeadArcAttention
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
    # of Arc's sigmoid. num_heads keeps setting the head's WIDTH; only the
    # count drops. See praxis/attention/single.py.
    "arc_single": SingleHeadArcAttention,
    "arc_single_dropoff": partial(SingleHeadArcAttention, dropoff="warp"),
}


def patch_attention_config(config: Any, args: Any = None) -> None:
    """Let the selected attention mechanism correct the config describing it.

    A mechanism that does not build what the config asks for has to say so
    here, because the config is not a private argument list: it is serialized
    to ``config.json``, rendered in the blueprint tab, and read by every other
    module in the stack. ``arc_single`` runs ONE head whatever ``num_heads``
    says, and reads ``num_heads`` as the head's WIDTH instead - so without this
    the config would advertise a head count no module ever built.

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

    before = dict(vars(config)) if args is not None else None
    patch(config)
    if before is None:
        return

    # Diff rather than naming fields, so a future patch_config that corrects
    # some other knob stays reported correctly without editing this function.
    for key, old in before.items():
        new = getattr(config, key, old)
        if new != old and hasattr(args, key):
            setattr(args, key, new)
