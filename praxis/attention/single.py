"""
SingleHeadArcAttention: Arc with Mega-style single-head gated attention.

ArcAttention runs ``num_heads * num_queries`` heads through Infini's segment
loop, and every one of them pays twice - an S x S score matrix per segment,
plus a ``head_dim x head_dim`` compressive-memory state to retrieve from and
update on every segment boundary. This variant keeps exactly ONE head, which
divides both costs by ``num_query_heads`` and removes two of the block's three
projections.

Read that as a PARAMETER saving, not a wall-clock one, unless the sequence is
long. Measured (5060 Ti, uncompiled flex_attention, which is what the arc
configs run): the flex call costs a flat ~2.5 ms at seq 64 and 256 whether it
is handed 1 head or 16 - it is fixed-overhead bound, and head count only bites
past roughly a thousand positions, where 16 heads cost 10.3x one. Below that
the head count is free and this class is a wash on speed; the win it reliably
delivers is 5.5x fewer attention parameters at abstractinator-g's dimensions.

Two results say the capacity cut is not the loss it looks like.

Liu et al. (arXiv:2106.09650) find that multi-head attention's advantage over
single-head is not that it attends to several positions at once - multi-LAYER
single-head attention does that too, and does it better - but that it is
shallower, and therefore easier to optimize. With modern stabilization the
deeper single-head variant wins outright, without hyperparameter tuning. This
stack is already built in the regime their result needs: ``depth`` recurrent
passes over ``num_layers`` physical blocks, held under SandwichNorm precisely
because additive signals compound across a recurrent unroll.

Mega (Ma et al., arXiv:2209.10655) supplies the other half, as Theorem 1: if
the transformation G is a universal approximator, then for every X there is a
gate ``gamma = G(X)`` such that ``gamma * O_single == O_multihead``. One head
plus a learned elementwise gate on its output spans what the heads spanned.
Mega's other two ingredients are already present here in different clothing -
its damped EMA exists to inject position-aware local structure into a
position-agnostic attention, which ArcHoPE's warped rotary phase and Infini's
segment memory already do - so only the attention core is adopted:

    Z     = silu(X W_z + b_z)                      shared representation
    Q, K  = kappa_q * Z + mu_q,  kappa_k * Z + mu_k
    V     = silu(X W_v + b_v)
    O     = attn(Q K^T / sqrt(d)) V
    Y     = W_o (gamma * O),   gamma = silu(X W_gamma + b_gamma)

Q and K are per-dimension affine reads of one shared Z, which is where the
parameters go: the block holds a single input projection where multi-head Arc
holds a fused three-way one, and the score matmul runs once rather than
``num_query_heads`` times.

The gate is the one place this deliberately departs from ArcAttention. Arc
follows Qiu et al. (arXiv:2505.06708), whose head-specific SIGMOID gate after
SDPA is the right answer for a multi-head block - it introduces non-linearity
over the low-rank softmax mapping, applies query-dependent sparsity, and damps
the attention sink. It is the wrong answer here: sigmoid lands in (0, 1), so
it can only attenuate, while Mega's theorem needs a gate that can also amplify
and flip sign to reach an output the removed heads could have produced. SiLU
can do both; sigmoid cannot. The two classes gate differently on purpose, and
``arc_gate_negative`` reports whether that freedom is ever used.

Head width is ``head_size``, and only ``head_size``. ``patch_config`` below
corrects the head COUNT to 1 and touches nothing else, so width still falls out
of the standing rule - ``head_size or hidden_size // num_heads`` - which with
the count already corrected means an unset ``head_size`` gives one head
spanning the full hidden size. Set ``head_size`` to narrow it; at
abstractinator-g's width the flex kernel needs it narrowed, since a head_dim of
90 asks for more Triton shared memory than the card has.

Rewriting the count is what keeps the config honest: config.json, the
blueprint tab, the Arguments card and every module reading ``config.num_heads``
all report the 1 head that was actually built, and the inherited __init__ chain
sizes the output projection, the blend gate and the memory buffers from it with
no head-count overrides in this class.
"""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.attention.arc import ArcAttention
from praxis.attention.infini import NoCompressiveMemory


class SingleHeadArcAttention(ArcAttention):
    """ArcAttention with one head, a shared Q/K representation, and a SiLU gate.

    Everything else - the per-depth biases, the segment memory, ghostmax, the
    cached decode path, the dropoff ablation - is inherited unchanged. Only
    the three subclass hooks and the tensors whose shapes depend on the head
    count are replaced.
    """

    metric_descriptions = {
        **ArcAttention.metric_descriptions,
        "arc_gate_negative": {
            "description": (
                "Fraction of SiLU gate values below zero - the sign flips a sigmoid "
                "gate cannot make. Pinned at 0 = Arc's sigmoid gate would have done as "
                "well."
            ),
            "chart": {
                "title": "Arc Gate",
                "y_label": "Value",
                "y_scale": "linear",
                "group": "arc",
                "order": 30,
                "series_group": "arc_gate",
                "series_label": "negative fraction",
            },
        },
        "arc_gate_magnitude": {
            "description": (
                "Mean absolute SiLU gate value. Near 0 = the attention branch is "
                "closed off; near 1 = passed through; well above 1 = amplified."
            ),
            "chart": {
                "title": "Arc Gate",
                "y_label": "Value",
                "y_scale": "linear",
                "group": "arc",
                "order": 31,
                "series_group": "arc_gate",
                "series_label": "mean magnitude",
            },
        },
    }

    @classmethod
    def patch_config(cls, config) -> None:
        """Correct the head COUNT, and only the count.

        ``head_size`` is deliberately not touched. It is already the knob for
        head width, and an earlier version of this hook derived a width from
        ``num_heads`` before overwriting it - which quietly made ``num_heads``
        a second width control, so editing it from 3 to 1 tripled ``head_dim``
        from 30 to 90 and blew the Triton shared-memory limit on the flex
        kernel. One knob, one meaning: ``num_heads`` is a count and is
        overridden here; ``head_size`` is a width and belongs to the config.

        Width therefore falls out of the standing rule
        (``head_size or hidden_size // num_heads``, as CausalAttention and the
        encoding modules both apply it) with ``num_heads`` already corrected to
        1 - so an unset ``head_size`` gives one head spanning the full hidden
        size, exactly what that rule predicts for a single head. Set
        ``head_size`` to narrow it. Both this module and the positional
        encoding read the same patched config, so they agree either way with
        no special-casing.

        Idempotent, because it runs both from the CLI (so the config object is
        truthful from the moment it exists) and from ``__init__`` (so a config
        built directly, as tests do, still yields a correctly shaped module).
        """
        config.num_heads = 1
        config.num_queries = 1

    def __init__(self, config, **kwargs) -> None:
        self.patch_config(config)
        super().__init__(config, **kwargs)

        hidden_size = config.hidden_size

        # One projection for the block: [Z | V]. Mega reads Z from the EMA
        # output X' and V from X, but there is no EMA branch here (ArcHoPE and
        # the segment memory carry that role), so both read X - and both pass
        # through SiLU. Two Linears on the same input with the same activation
        # is one Linear with a wider output, so this is the same function in
        # one matmul rather than two. This is the only projection whose width
        # the head counts do not already imply, so it is the only one rebuilt.
        self.qkv_dim = 2 * self.head_dim
        self.qkv = nn.Linear(hidden_size, self.qkv_dim, bias=True)
        self.depth_bias = nn.Embedding(self.depth, self.qkv_dim + hidden_size)
        nn.init.zeros_(self.depth_bias.weight)

        # Q and K are per-dimension affine reads of Z - free, and the reason
        # the block is cheap. kappa is small-random (Mega's own init) rather
        # than 1: at kappa = 1, mu = 0 the two reads are identical, Q K^T is a
        # symmetric Gram matrix with a dominant diagonal, and attention starts
        # collapsed onto the query's own position. Small kappa starts the
        # scores near zero instead, which under ghostmax is a near-uniform
        # read that the gate and the memory can immediately work against.
        self.kappa = nn.Parameter(torch.empty(2, self.head_dim))
        nn.init.normal_(self.kappa, mean=0.0, std=0.02)
        self.mu = nn.Parameter(torch.zeros(2, self.head_dim))

        self._gate_negative: Optional[Tensor] = None
        self._gate_magnitude: Optional[Tensor] = None

    @contextmanager
    def head_budget(self, kv_keep: Tensor):
        """No-op: there is one head, and a budget cannot remove it.

        The inherited version slices the fused QKV rows by a per-head channel
        layout this class does not have - its projection emits [Z | V], not
        [Q | K | V] - so inheriting it would gather the wrong rows silently. A
        mixture-of-widths policy therefore leaves this block at full width.
        """
        yield

    def _project_qkv(self, inputs: Tensor, current_depth: int) -> Tensor:
        depth_idx = torch.tensor(current_depth, device=inputs.device)
        base = self.qkv(inputs) + self.depth_bias(depth_idx)[..., : self.qkv_dim]
        z, v = F.silu(base).split([self.head_dim, self.head_dim], dim=-1)
        q = self.kappa[0] * z + self.mu[0]
        k = self.kappa[1] * z + self.mu[1]
        return torch.cat([q, k, v], dim=-1)

    def _finalize_output(
        self, output: Tensor, inputs: Tensor, current_depth: int
    ) -> Tensor:
        gate = F.silu(self.gate(inputs))
        self._record_gate(gate)
        output = output * gate
        depth_idx = torch.tensor(current_depth, device=inputs.device)
        return self.output(output) + self.depth_bias(depth_idx)[..., self.qkv_dim :]

    def _record_gate(self, gate: Tensor) -> None:
        """Stash the gate's shape for training_metrics(), on-device.

        Two reductions per forward, no host sync: the ``.item()`` happens in
        training_metrics(), which runs on the logging cadence. Each call
        overwrites the last, so what surfaces is the most recent recurrent
        pass through this layer rather than an average over depth. Skipped
        under torch.compile, where mutating module attributes forces a graph
        break.
        """
        if not self.training or torch.compiler.is_compiling():
            return
        with torch.no_grad():
            self._gate_negative = (gate < 0).to(gate.dtype).mean()
            self._gate_magnitude = gate.abs().mean()

    def training_metrics(self) -> dict:
        out = super().training_metrics()
        if self._gate_negative is not None:
            out["arc_gate_negative"] = self._gate_negative.item()
            out["arc_gate_magnitude"] = self._gate_magnitude.item()
        return out


class SingleHeadArcNoMemAttention(NoCompressiveMemory, SingleHeadArcAttention):
    """SingleHeadArcAttention with the compressive memory removed. See
    :class:`~praxis.attention.infini.NoCompressiveMemory`."""
