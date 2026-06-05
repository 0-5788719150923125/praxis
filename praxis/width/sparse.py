"""Helical width deflation, truly sparse: slice the weights so the matmul
shrinks, rather than masking a full-size one.

Two recipes run each step, both over a depth-precessing helical window:

* **FFN** - a generic resizer walks the experts for GLU-style ``up``/``down``
  projection pairs (ArcGLU, GatedLinearMLP, anything naming them that way) and
  patches their forwards to compute over a window of the intermediate dimension.
  Keyed by attribute name, so retargeting is a one-line ``PAIR_NAMES`` change.

* **Attention** - the bulk of the compute. Whole KV heads (and their GQA query
  groups) are dropped via the attention's own ``head_budget`` context, which
  slices the fused QKV, output, gate, per-depth biases and Infini memory in
  lockstep. The dataflow-entangled slicing lives in the attention module that
  owns it; this policy only chooses how many heads survive.

Both genuinely shrink the matmul, and gradients reach only the rows/heads that
ran - rank-sparse and conditional, the same shape as mixture-of-depths hard
gating but on the feature axis. Because the budget is a function of depth, only
a handful of shapes ever occur (one per recurrent step), so the inner torch
recompiles are bounded and predictable - a multiplier over the steps, not an
open-ended blowup.
"""

from contextlib import ExitStack, contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.width.helical import GOLDEN, HelicalWidth, is_compiling, width_fraction

try:  # lazy (uninitialized) params/buffers: never slice these
    from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

    _LAZY = (UninitializedParameter, UninitializedBuffer)
except ImportError:  # older torch
    from torch.nn.parameter import UninitializedParameter

    _LAZY = (UninitializedParameter,)

# (producer, consumer) attribute names whose shared inner dimension we slice.
# ``up``/``down`` is the GLU convention across the dense registry.
PAIR_NAMES = [("up", "down")]


def _helix_window(size, frac, depth, device):
    """``r`` indices of ``size`` on a golden-stride window that precesses with
    depth, or ``None`` when the budget keeps everything (nothing to slice)."""
    r = max(1, min(size, round(frac * size)))
    if r >= size:
        return None
    start = round(size * GOLDEN * depth) % size
    return (start + torch.arange(r, device=device)) % size


def _attention_modules(experts):
    """Attention submodules that expose a ``head_budget`` recipe."""
    for expert in experts:
        if not isinstance(expert, nn.Module):
            continue
        for module in expert.modules():
            if hasattr(module, "head_budget") and hasattr(module, "num_heads"):
                yield module


def _glu_modules(experts):
    """Every GLU-style module (``up``/``down`` Linear pair) under these experts,
    with its activation - so a parametric activation can be sliced in step."""
    for expert in experts:
        if not isinstance(expert, nn.Module):
            continue
        for module in expert.modules():
            for up_name, down_name in PAIR_NAMES:
                up = getattr(module, up_name, None)
                down = getattr(module, down_name, None)
                if isinstance(up, nn.Linear) and isinstance(down, nn.Linear):
                    yield module, up, down, getattr(module, "act", None)


def _activation_channel_tensors(act, inner):
    """``(store, name)`` for every per-channel activation tensor (param or
    buffer) sized to the intermediate width - what must be sliced alongside the
    GLU so a parametric activation (Serpent, Snake, PReLU, ...) stays in sync.

    Returns ``None`` if any is still lazy (uninitialized): we must not slice the
    GLU then, or the activation would materialize at the sliced width."""
    out = []
    if act is None:
        return out
    for sub in act.modules():
        for store in (sub._parameters, sub._buffers):
            for name, tensor in store.items():
                if tensor is None:
                    continue
                if isinstance(tensor, _LAZY):
                    return None
                if tensor.dim() >= 1 and tensor.shape[-1] == inner:
                    out.append((store, name))
    return out


@contextmanager
def _glu_slice(up, down, act, idx, inner):
    """Compute the GLU over the kept intermediate window ``idx`` for one step,
    then restore. ``up`` emits ``2 * len(idx)`` channels ordered ``[a(idx),
    b(idx)]`` so the block's ``chunk(2)`` still splits gate and value correctly;
    ``down`` consumes the same ``idx`` columns; and any per-channel activation
    parameter (Serpent's a/b/g, ...) is sliced to the same window so it stays
    aligned. All slices are differentiable gathers, so the backward scatters
    into the full tensors - only the used slice trains.

    A no-op (full GLU) when the activation is still lazy, so its params
    materialize at full width before they are ever sliced."""
    act_chans = _activation_channel_tensors(act, inner)
    if act_chans is None:
        yield
        return

    up_rows = torch.cat([idx, idx + inner])
    up_w = up.weight[up_rows]
    up_b = up.bias[up_rows] if up.bias is not None else None
    down_w = down.weight[:, idx]
    up_orig, down_orig = up.forward, down.forward
    up.forward = lambda x: F.linear(x, up_w, up_b)
    down.forward = lambda x: F.linear(x, down_w, down.bias)
    swaps = []  # (store, name, original) for activation params/buffers
    for store, name in act_chans:
        swaps.append((store, name, store[name]))
        store[name] = store[name][..., idx]
    try:
        yield
    finally:
        up.forward, down.forward = up_orig, down_orig
        for store, name, original in swaps:
            store[name] = original


class HelicalSparseWidth(HelicalWidth):
    """Truly-sparse helical deflation: same arch schedule as ``HelicalWidth``,
    but it slices the GLU and attention-head weights so the matmuls actually
    shrink instead of masking them."""

    def extra_repr(self) -> str:
        return f"floor={self.floor}, peak={self.peak}, sparse=True"

    @contextmanager
    def scope(self, experts, current_depth, max_depth):
        if is_compiling():  # runtime weight/forward slicing is untraceable
            yield
            return
        frac = width_fraction(current_depth, max_depth, self.floor, self.peak)
        with ExitStack() as stack:
            # FFN: slice each GLU's intermediate rank (and its activation).
            for module, up, down, act in _glu_modules(experts):
                inner = down.in_features
                idx = _helix_window(inner, frac, current_depth, up.weight.device)
                if idx is not None:
                    stack.enter_context(_glu_slice(up, down, act, idx, inner))
            # Attention: drop whole KV heads (the dominant compute).
            for attn in _attention_modules(experts):
                kv = _helix_window(
                    attn.num_heads, frac, current_depth, attn.qkv.weight.device
                )
                if kv is not None:
                    stack.enter_context(attn.head_budget(kv))
            yield
