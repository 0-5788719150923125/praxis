"""In-house NeuralMemory: a Titans-style memory that learns at test time.

The memory network's weights are updated online by gradient descent on a
"surprise" loss - how badly the network reconstructs the value from the key -
modulated per token by a learned learning rate, with per-chunk momentum
(surprise carries forward) and weight decay (adaptive forgetting). Memory is
re-initialized per sequence (no learnable init state), which sidesteps the
collapse mode that bit the Infini/Arc memories.

Following the paper, every chunk's surprise gradient is taken against the
frozen segment-start weights, so all chunks are differentiated in one batched
pass and the per-chunk momentum/decay recurrence collapses to a parallel
associative scan (``_affine_scan``) over the chunk axis.

With ``use_energy=True`` the whole test-time update runs detached: the surprise
gradient is a purely local learning rule and no scan trajectory / second-order
graph is retained (that trajectory dominates VRAM). The learned update gates are
replaced by a fixed Adam-style rule - per-chunk EMAs of the surprise (1st/2nd
moment) give a direction, which is then scaled by the segment-start weights' own
RMS so the write is a constant RELATIVE perturbation. That second half is not
cosmetic: ``m_hat / sqrt(v_hat)`` is sign-like, so without it the step is a fixed
ABSOLUTE ``max_lr`` while W0 is a trained parameter that grows unchecked (the
readout is behind ``out_norm``, so nothing constrains the memory net's output
magnitude), and the test-time update decays into irrelevance over a run - see
``_step_scale``, which also divides by ``sqrt(num_chunks)`` so the update grid
is not a second, undeclared learning rate. There are no untrained gate heads
either way. Training the encoder on the reconstruction
energy would collapse (value -> 0), so instead the key projection is tied to the
query projection - the shared addressing map learns on the task through
retrieval - and the value side is fixed to identity, leaving ``combine`` to adapt
content. The backbone connects through that retrieval and the residual skip.
The reconstruction is measured on RMS-normalized vectors (matching the
out_norm'd readout), so the memory net's free output-scale mode can't dominate
the energy.
"""

import contextlib
import logging
from contextlib import contextmanager
from typing import Any, Dict, NamedTuple, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad_and_value, vmap

from praxis.activations import ACT2CLS

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

Weights = Dict[str, Tensor]

# Reverse map from activation class -> registry name, for a readable repr.
_ACTIVATION_NAMES = {}
for _name, _entry in ACT2CLS.items():
    _ACTIVATION_NAMES.setdefault(
        _entry[0] if isinstance(_entry, tuple) else _entry, _name
    )


_log = logging.getLogger("praxis.memory")

# Static shapes, matching praxis.trainers.compile. `dynamic=True` was measured
# here and is WORSE on the only axis that separates them: steady state is a
# wash (29.1s vs 29.8s on a 128-byte turn) but it warms in at 155s against 86s
# and leaves the first real turn at 109s against 55s. Decode walks a range of
# sequence lengths, so symbolic shapes looked like the obvious fit; they are
# not, and the reason to write that down is so the next person does not
# re-derive the obvious guess.
_DECODE_COMPILE_KWARGS = dict(mode="default", fullgraph=False, dynamic=False)

# Init bias for the standard-mode forgetting gate (alpha_t, paper Eq. 13), which
# compounds once per chunk. sigmoid(-5) = 0.0067 forgetting per chunk, so the
# meta-learned init survives the horizons this repo actually runs (90% over 16
# chunks, 42% over the reference's 128) instead of being erased by it. Chosen for
# the shape of the compounding, not tuned per experiment: any value here is a
# judgement call, but the direction is not - a gate that erases by default
# inverts what the paper's alpha means.
_DECAY_GATE_BIAS: float = -5.0


@contextmanager
def decode_compiled(model: nn.Module, enabled: bool = True):
    """Install compiled bodies on every :class:`NeuralMemory` under ``model``
    for the duration of the block, then restore eager.

    Compilation happens once per module and is cached on the instance, so only
    the first generation of a run pays for it (~3-4 minutes, well inside the
    stall watchdog's 600s). A failure to compile is not worth losing a
    generation over, so it degrades to eager and stays there.

    Scoped rather than permanent because the model object is shared with the
    trainer: outside this block the module tree must be byte-for-byte what
    training compiled against.
    """
    if not enabled:
        yield
        return
    installed = []
    for module in model.modules():
        if not isinstance(module, NeuralMemory):
            continue
        fn = module._compiled_body
        if fn is None:
            try:
                fn = torch.compile(module._forward_impl, **_DECODE_COMPILE_KWARGS)
            except Exception:
                _log.debug("Could not compile NeuralMemory for decode", exc_info=True)
                continue
            module._compiled_body = fn
        module._decode_forward = fn
        installed.append(module)
    try:
        yield
    finally:
        for module in installed:
            module._decode_forward = None


def _shift_chunks(t: Tensor, d: int, fill: float) -> Tensor:
    """Shift ``t`` along the chunk axis (dim=1) by ``d``, front-filled."""
    pad = t.new_full((t.shape[0], d, *t.shape[2:]), fill)
    return torch.cat([pad, t[:, :-d]], dim=1)


def _affine_scan(a: Tensor, b: Tensor, prev: Tensor) -> Tensor:
    """Inclusive scan of ``x_t = a_t * x_{t-1} + b_t`` over the chunk axis, with
    ``x_{-1} = prev``. Parallel (Hillis-Steele), O(log nc) depth.

    ``a`` is ``(B, nc)`` per-chunk scalars; ``b`` is ``(B, nc, *p)``; ``prev``
    is ``(B, *p)``. Composes affine maps rather than dividing by the cumulative
    product, so it stays stable as the decay/momentum factors shrink.
    """
    nc = b.shape[1]
    A = a.reshape(a.shape + (1,) * (b.dim() - 2))  # (B, nc, 1, ...)
    # Fold the initial carry into the first element: x_0 = a_0 * prev + b_0.
    b0 = A[:, :1] * prev.unsqueeze(1) + b[:, :1]
    B = torch.cat([b0, b[:, 1:]], dim=1)
    d = 1
    while d < nc:
        A_prev = _shift_chunks(A, d, 1.0)
        B_prev = _shift_chunks(B, d, 0.0)
        B = A * B_prev + B
        A = A * A_prev
        d *= 2
    return B


class NeuralMemState(NamedTuple):
    """Per-sequence memory state, threaded across chunks and decode steps."""

    seq_index: int
    weights: Weights  # fast weights, leading batch dim
    momentum: Weights  # 1st-moment accumulator, leading batch dim
    second_moment: Weights  # 2nd-moment accumulator (Adam/energy mode only)


def mem_state_detach(state: Optional[NeuralMemState]) -> Optional[NeuralMemState]:
    """Detach memory state from the graph (truncates BPTT across segments)."""
    if state is None:
        return None
    return NeuralMemState(
        state.seq_index,
        {k: v.detach() for k, v in state.weights.items()},
        {k: v.detach() for k, v in state.momentum.items()},
        {k: v.detach() for k, v in state.second_moment.items()},
    )


class NeuralMemory(nn.Module):
    """Test-time-learned associative memory (Titans, Behrouz et al. 2024)."""

    # Opt out of parameter-merging routers (praxis/routers/targeting.py). These
    # weights are meta-learned INITIAL CONDITIONS for a test-time update rule,
    # not a geometry a router should be choosing between: the object that
    # actually processes a sequence is the one this decays into after the
    # in-context updates, and the per-forward state that governs those updates
    # (a depth bank's occupancy, a band smear's reward EMA) lives in buffers,
    # which no parameter merge carries anyway. The expert-bank path reached the
    # same conclusion by sharing one memory across the copies
    # (praxis/decoders/base.py); this states it once, on the class.
    MERGE_OPAQUE: bool = True

    # Cadence for the readout probe (see ``_readout_delta``), in forward calls.
    # The dynamics logger reads on a far slower cadence than this module runs
    # (once per depth, per microbatch), so probing every call would compute a
    # value that is overwritten hundreds of times before anything reads it.
    PROBE_EVERY: int = 8

    def __init__(
        self,
        dim: int,
        model: nn.Module,
        chunk_size: int = 64,
        max_lr: float = 1e-2,
        momentum: bool = True,
        use_energy: bool = False,
        segment: bool = False,
        segment_block: int = 16,
        segment_gamma: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        parallel_scan: bool = True,
        write_objective: str = "recon",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.max_lr = max_lr
        self.use_momentum = momentum
        self.use_energy = use_energy
        # Write target for the test-time update. "recon": auto-associative, the
        # value is the (normalized) input itself - the memory reconstructs the
        # stream, which is redundant with the residual it's added to, so the
        # model learns to route around it (gain -> 0 at every scale). "predictive"
        # (NextLat, Liu et al. 2025): the value is the *next* latent stream_{t+1},
        # so retrieval carries belief-state info the residual doesn't already
        # hold. The target is stop-gradded (the detached update ctx) and the loss
        # is Huber, both per NextLat. Only meaningful in energy mode (recon mode
        # there fixes the value side to identity).
        # The predictive target is stop-gradded EXPLICITLY (see the forward), not
        # merely by sitting inside energy mode's no_grad region - so it is
        # available in standard mode too. Without that detach the encoder could
        # minimize its own surprise by making the next latent trivially
        # predictable, the BYOL/SimSiam collapse.
        assert write_objective in ("recon", "predictive"), write_objective
        self.write_objective = write_objective
        self.predictive = write_objective == "predictive"
        # True: differentiate every chunk in one batched pass and collapse the
        # per-chunk recurrence to a parallel scan (fast, materializes the full
        # (b, nc, *p) trajectory). False: a sequential loop carrying running
        # state, so the trajectory never exists - lower peak VRAM (energy mode
        # only), ~1.5x slower, same numerics.
        self.parallel_scan = parallel_scan
        # Surprise-based event segmentation (EM-LLM, Fountas et al. 2024), energy
        # mode only: split the update grid at surprise spikes instead of fixed
        # chunks. The grid becomes segment_block tokens; consecutive blocks merge
        # into an event until surprise exceeds mu+gamma*sigma (causal window) or
        # the event reaches chunk_size (the cap, bounding VRAM). Boundaries reset
        # the update's momentum so a context shift starts a fresh memory write.
        self.segment = segment and use_energy
        self.segment_block = segment_block
        self.segment_gamma = segment_gamma
        if self.segment:
            assert (
                chunk_size % segment_block == 0
            ), "chunk_size must be a multiple of segment_block"
        self._cap_blocks = chunk_size // segment_block
        # Energy mode: Adam-style adaptive update constants (replace the learned
        # gates). 1st/2nd-moment EMAs make the step scale-invariant, so the fixed
        # max_lr stays safe; weight_decay is optional forgetting.
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay

        # The memory network: any ``dim -> dim`` module whose weights are
        # updated at test time. Built from a praxis.dense variant.
        self.memory_model = model
        self._param_names = [n for n, _ in self.memory_model.named_parameters()]

        self.retrieve_norm = nn.RMSNorm(dim)
        # Energy mode shares the norm between store and retrieve (it would
        # otherwise be a frozen store-only param); standard mode keeps them apart.
        self.store_norm = self.retrieve_norm if use_energy else nn.RMSNorm(dim)
        self.out_norm = nn.RMSNorm(dim)

        self.to_queries = nn.Linear(dim, dim, bias=False)
        if use_energy:
            # Tie the key projection to the query projection so the shared
            # addressing map learns on the task through retrieval (training the
            # encoder on the reconstruction energy would collapse), and fix the
            # value side to identity - combine adapts the readout.
            self.to_keys = self.to_queries
            self.to_values = nn.Identity()
        else:
            self.to_keys = nn.Linear(dim, dim, bias=False)
            self.to_values = nn.Linear(dim, dim, bias=False)
        self.combine = nn.Linear(dim, dim, bias=False)

        # Data-dependent update controls (standard mode only): per-token lr,
        # per-chunk momentum and decay. Energy mode replaces these with the
        # parameter-free Adam-style rule, so it carries no untrained gate heads.
        if not use_energy:
            self.to_lr = nn.Linear(dim, 1)
            self.to_momentum = nn.Linear(dim, 1)
            self.to_decay = nn.Linear(dim, 1)
            # The forgetting gate must START AT RETAIN and learn to forget.
            # alpha_t is the paper's Eq. 13 gate: "it can update the memory
            # without affecting the past abstraction by letting alpha -> 0, and
            # can clear the entire memory by letting alpha -> 1". A default
            # Linear init sits at sigmoid(~0) = 0.5, i.e. exactly halfway to
            # "clear the entire memory" - and it compounds per chunk, so the
            # meta-learned W0 is annihilated before the sequence ends: measured
            # ||W_T||/||W0|| of 0.22 at 2 chunks, 0.056 at 4, 0.0029 at 8 and
            # 0.00059 at 16. The memory was being erased rather than written to,
            # worse the more chunks it got - the exact axis a longer horizon
            # moves along. Biasing the gate to near-zero forgetting matches how
            # every other gate in this repo enters (MAG's -3, ReZero): start at
            # identity, let the task open it.
            nn.init.zeros_(self.to_decay.weight)
            nn.init.constant_(self.to_decay.bias, _DECAY_GATE_BIAS)

        # Diagnostics from the last store pass, logged as metrics: cold-start
        # surprise, the memory's output magnitude relative to the stream, the
        # relative size of the test-time weight update, and what that update
        # changed in the readout (the same write, measured in function space).
        self.last_surprise: Optional[Tensor] = None
        self.last_surprise_norm: Optional[Tensor] = None
        self.last_gain: Optional[Tensor] = None
        self.last_write: Optional[Tensor] = None
        self.last_adapt: Optional[Tensor] = None
        # How many chunks the last call's sequence resolved to. Retrieval reads
        # PRE-write weights, so a chunk's own update is only ever visible to the
        # chunks after it: the effective number of in-context writes is
        # ``num_chunks - 1``, and at one chunk the module degenerates to a static
        # readout at W0 with the update computed and thrown away. That is a
        # silent failure - gain and write both look healthy while adapt is
        # exactly 0 - so the count is surfaced rather than left to be inferred.
        self.last_num_chunks: Optional[int] = None
        self._warned_single_chunk: bool = False
        # -1 so the very first call probes, rather than PROBE_EVERY calls in.
        self._probe_tick: int = -1
        # Event-size stats from the last segmented store pass (tokens per event).
        self.last_event_mean: Optional[Tensor] = None
        self.last_event_min: Optional[Tensor] = None
        self.last_event_max: Optional[Tensor] = None

    # Compiled copy of ``_forward_impl``, installed for the duration of a
    # generation by ``decode_compiled``. None (eager) everywhere else.
    #
    # Why this module and not the whole decoder: measured on abstractinator-t,
    # one forward issues ~8.6k aten dispatches and ~59% of them come from here,
    # on tensors small enough that every one is dispatch cost rather than
    # arithmetic. Compiling this module alone took a 128-byte generation from
    # 42.5s to 25.0s (1.7x) with byte-identical output. Compiling the whole
    # decoder instead is 100x SLOWER: the recurrent loop passes current_depth
    # as a python int and KL halting varies the loop count per input, so Dynamo
    # re-traces on nearly every call. This module's shapes are stable by
    # construction - it pads to a whole number of chunks internally - which is
    # exactly why it is the piece that compiles cleanly.
    _decode_forward = None
    _compiled_body = None

    def _activation_name(self) -> str:
        for module in self.memory_model.modules():
            name = _ACTIVATION_NAMES.get(type(module))
            if name is not None:
                return name
        return "none"

    def __repr__(self) -> str:
        # One line - the submodules would otherwise spam print(model).
        return (
            f"{type(self).__name__}(dim={self.dim}, chunk_size={self.chunk_size}, "
            f"model={type(self.memory_model).__name__}, "
            f"activation={self._activation_name()}, momentum={self.use_momentum}, "
            f"energy={self.use_energy}, segment={self.segment}, "
            f"write_objective={self.write_objective}, "
            f"parallel_scan={self.parallel_scan})"
        )

    def _update_ctx(self):
        """Context for the test-time update: detached in energy mode so neither
        the scan trajectory nor the second-order surprise graph is retained."""
        return torch.no_grad() if self.use_energy else contextlib.nullcontext()

    # --- readout probe -------------------------------------------------------

    def _probe_due(self) -> bool:
        """Cadence gate for ``_readout_delta``: one call in ``PROBE_EVERY``,
        training only.

        The counter is deliberately NOT consulted under compile. A host-side
        branch on a mutating Python int inside a traced region guards on that
        int, so every call would fail the guard and recompile - far worse than
        the probe it was meant to save. ``is_compiling()`` folds to a constant
        during tracing, so the compiled graph simply probes every call and pays
        the extra memory-net forward (a few percent of this module) rather than
        dropping the metric from every default run. Eager keeps the cadence.
        """
        if not self.training:
            return False
        if torch.compiler.is_compiling():
            return True
        self._probe_tick += 1
        return self._probe_tick % self.PROBE_EVERY == 0

    def _readout_delta(
        self, retrieved: Tensor, base_weights: Weights, queries: Tensor, n: int
    ) -> Tensor:
        """How much this call's writes changed the READOUT, relative.

        ``last_write`` measures the same update in weight space, against a
        ``||W0||`` that grows as the trunk trains - so a step of unchanged size
        reads as a shrinking ratio, and "the update is inert" is indistinguish-
        able from "the update is small next to weights that outgrew it". This
        re-retrieves the call's own queries at the weights it STARTED from and
        reports ``||read(W_T) - read(W0)|| / ||read(W0)||``: 0 means the writes
        changed nothing the layer above can see, which is the claim the weight
        ratio is usually read as making.

        Cheap relative to the store pass it rides along with: one extra memory
        net forward, vmapped over the batch only (every position reads the same
        start weights, unlike the real retrieval's per-chunk weights), no
        autograd, and only on the ``PROBE_EVERY`` cadence.
        """
        b = retrieved.shape[0]
        base = vmap(lambda w, q: functional_call(self.memory_model, w, (q,)))(
            base_weights, queries.reshape(b, -1, self.dim)
        )
        base = self.combine(self.out_norm(base))[:, :n]
        return (retrieved - base).norm() / (base.norm() + self.eps)

    # --- state ---------------------------------------------------------------

    def _init_weights(self, batch: int) -> Weights:
        """Expand the meta-learned init weights (W0) to a per-sequence batch."""
        return {
            n: p.unsqueeze(0).expand(batch, *p.shape)
            for n, p in self.memory_model.named_parameters()
        }

    def init_state(self, batch: int, device=None) -> NeuralMemState:
        weights = self._init_weights(batch)
        zeros = lambda: {n: torch.zeros_like(w) for n, w in weights.items()}
        return NeuralMemState(0, weights, zeros(), zeros())

    def _segment(self, s_blocks: Tensor) -> Tuple[Tensor, Tensor]:
        """Surprise-based event boundaries over the block axis (EM-LLM rule).

        ``s_blocks`` is the per-block surprise ``(b, nb)``. A block starts a new
        event when its surprise exceeds a causal ``mu + gamma*sigma`` threshold
        (running stats over prior blocks) or the running event reaches the cap.
        Returns a boolean ``reset_mask`` and the 1-indexed per-event position
        ``t_event`` (both ``(b, nb)``), for resetting the Adam EMAs.
        """
        b, nb = s_blocks.shape
        # Causal running mean/std over strictly prior blocks (parameter-free
        # window). Block 0 has no history, so it can only be a forced start.
        idx = torch.arange(nb, device=s_blocks.device)
        csum = s_blocks.cumsum(1) - s_blocks
        csq = (s_blocks * s_blocks).cumsum(1) - s_blocks * s_blocks
        count = idx.clamp(min=1).to(s_blocks.dtype)
        mean = csum / count
        var = (csq / count - mean * mean).clamp(min=0.0)
        # Relative tolerance on the threshold, matching _forward_sequential: the
        # variance is a difference of two large cumulative sums, so on a
        # near-constant stream it rounds to a spurious spike and cuts an event
        # every block. Both paths must agree or ``parallel_scan`` stops being a
        # pure perf knob.
        thresh = mean + self.segment_gamma * var.sqrt() + 1e-5 * mean.abs()
        spike = s_blocks > thresh
        spike[:, 0] = False  # no history at block 0

        # Walk the blocks to apply the cap, which is a stateful recurrence
        # (a spike resets the counter, shifting later forced boundaries). Cheap:
        # nb vector ops over (b,), no autograd, no model calls.
        reset = torch.zeros_like(spike)
        t_event = torch.ones(b, nb, dtype=torch.long, device=s_blocks.device)
        run = torch.zeros(b, dtype=torch.long, device=s_blocks.device)
        for j in range(nb):
            is_b = spike[:, j] | (run >= self._cap_blocks)
            if j == 0:
                is_b = torch.ones_like(is_b)
            reset[:, j] = is_b
            run = torch.where(is_b, torch.ones_like(run), run + 1)
            t_event[:, j] = run
        return reset, t_event

    def _step_scale(
        self, w0: Weights, name: str, ndim: int, b: int, num_chunks: int = 1
    ) -> Tensor:
        """What one chunk's update step is scaled by, shaped to broadcast over a
        surprise/update tensor of ``ndim`` dims. Two normalizations, so that
        ``max_lr`` means the same thing regardless of how big the memory net's
        weights have grown and regardless of how finely the pass is chunked.

        ``u = m_hat / (sqrt(v_hat) + eps)`` is sign-like, so without this the
        test-time step is a FIXED ABSOLUTE ``max_lr`` per element while ``W0``
        is a trained parameter free to grow - and it does, because the readout
        sits behind ``out_norm`` (RMSNorm, exactly scale-invariant), leaving the
        memory net's output magnitude a mode the outer loss cannot see and a
        sign-based optimizer random-walks upward. abstractinator-x measured the
        consequence: raw surprise up 85,935x (an output scale of ~185x),
        ``memory_write`` down 14x and ``memory_adapt`` down 78x to 0.010, so the
        module ended the run as a large static nonlinearity that the gate still
        wanted (0.53) but that no longer learned anything in context. Scaling by
        the parameter's own RMS makes the write a constant RELATIVE
        perturbation, which is the invariance this rule always claimed: measured
        across a 185x weight-scale sweep, write holds at 0.031 and adapt at
        0.07-0.09 where the absolute step decayed to 0.0027 and 0.0087.

        Taken from the SEGMENT-START weights, not the running ones, so every
        chunk in a pass is scaled identically and the parallel and sequential
        paths cannot drift apart.

        The second normalization is ``1/sqrt(num_chunks)``, and it exists so the
        UPDATE GRID stops acting as a hidden learning rate. ``u`` is sign-like
        and roughly decorrelated across chunks, so a pass's total write
        accumulates as ``max_lr * sqrt(nc)``: measured, ``write / sqrt(nc)``
        sits at 0.0104-0.0115 against a ``max_lr`` of 0.01 across grids from 2
        to 32 chunks. Halving ``segment_block`` therefore multiplied the
        effective step by 1.41 as a silent side effect, so two profiles sharing
        a ``max_lr`` but differing in grid were not running the same rule - the
        kind of undeclared knob this repo's no-tuning rule exists to remove.
        With this, ``max_lr`` is the TOTAL RELATIVE WRITE PER PASS and the grid
        sets only granularity, not strength.

        NOTE this REDEFINES ``max_lr`` rather than merely tidying it. Runs
        before this were writing ``max_lr * sqrt(nc)`` per pass - for
        abstractinator-y, whose median grid is 5-9 chunks, an effective
        0.022-0.030 against the nominal 0.01. A later run at the same nominal
        value is a genuinely gentler one, and is not comparable to -y on that
        axis.
        """
        rms = w0[name].flatten(1).pow(2).mean(-1).sqrt().clamp(min=self.eps)
        rms = rms / max(1.0, float(num_chunks)) ** 0.5
        return rms.reshape((b,) + (1,) * (ndim - 1))

    def _adam_update(
        self,
        weights: Weights,
        momentum: Weights,
        second_moment: Weights,
        surprise: Weights,
        num_chunks: int,
        reset_mask: Optional[Tensor] = None,
        t_event: Optional[Tensor] = None,
    ) -> Tuple[Weights, Weights, Weights, Weights]:
        """Detached Adam-style test-time update. Per-chunk EMAs of the surprise
        (1st/2nd moment, bias-corrected) give a scale-invariant step, so the
        fixed ``max_lr`` is safe; the parallel scans run over the chunk axis.

        With segmentation, ``reset_mask`` zeroes the EMA carry at event starts
        (fresh moments) and ``t_event`` re-bases the bias correction per event;
        the weights themselves persist across events (long-term memory)."""
        ref = surprise[self._param_names[0]]
        b = ref.shape[0]
        beta1 = ref.new_full((b, num_chunks), self.beta1)
        beta2 = ref.new_full((b, num_chunks), self.beta2)
        keep = ref.new_full((b, num_chunks), 1.0 - self.weight_decay)
        if reset_mask is not None:
            beta1 = beta1.masked_fill(reset_mask, 0.0)
            beta2 = beta2.masked_fill(reset_mask, 0.0)
        if t_event is None:
            t_event = torch.arange(1, num_chunks + 1, device=ref.device).expand(b, -1)
        c1 = 1.0 - self.beta1 ** t_event.to(ref.dtype)  # bias-correction, (b, nc)
        c2 = 1.0 - self.beta2 ** t_event.to(ref.dtype)

        chunk_weights, new_weights, new_m, new_v = {}, {}, {}, {}
        for name in self._param_names:
            s = surprise[name]
            bshape = (b, num_chunks) + (1,) * (s.dim() - 2)
            v = _affine_scan(beta2, (1.0 - self.beta2) * s * s, second_moment[name])
            if self.use_momentum:
                m = _affine_scan(beta1, (1.0 - self.beta1) * s, momentum[name])
                m_hat = m / c1.reshape(bshape)
            else:
                m, m_hat = s, s
            u = m_hat / ((v / c2.reshape(bshape)).sqrt() + self.eps)
            step = (
                self.max_lr
                * u
                * self._step_scale(weights, name, s.dim(), b, num_chunks)
            )
            w_t = _affine_scan(keep, step, weights[name])  # (b, nc, *p)
            chunk_weights[name] = w_t
            new_weights[name] = w_t[:, -1]
            new_m[name] = m[:, -1] if self.use_momentum else momentum[name]
            new_v[name] = v[:, -1]
        return chunk_weights, new_weights, new_m, new_v

    # --- functional grad of the surprise loss --------------------------------

    def _note_chunks(self, num_chunks: int, n: int, c: int) -> None:
        """Record the call's chunk count, and warn once if the memory cannot
        actually adapt in context at this sequence length."""
        if torch.compiler.is_compiling():
            return
        self.last_num_chunks = int(num_chunks)
        if num_chunks > 1 or self._warned_single_chunk:
            return
        self._warned_single_chunk = True
        _log.warning(
            "NeuralMemory: %d tokens on a %d-token grid is a single chunk, so "
            "retrieval reads the cold weights and the test-time update is "
            "discarded - memory_adapt will be exactly 0. Shorten the update "
            "grid (segment_block/chunk_size) or lengthen the sequence.",
            n,
            c,
        )

    def _valid_grid(self, seq: Tensor, num_chunks: int, c: int, pad: int) -> Tensor:
        """``(1, nc, c)`` mask marking the real (non-pad) grid positions.

        Broadcasts over the batch: the pad is a tail of the sequence, so every
        row shares it.

        The masking is MULTIPLICATIVE (``lr`` scales each token's contribution
        to the surprise sum), so the pad is still forwarded through the memory
        net and its loss is still computed - only its weight is zero. Any FINITE
        pad value is therefore equivalent, which the invariance test pins by
        refilling the pad and requiring bit-identical weights. A non-finite one
        is not: ``0 * inf`` is nan and would poison the carried state. The pad
        comes from ``F.pad``'s default 0.0 on a hidden-state tensor - not a
        token id, so the ``IGNORE_INDEX``/-100 label convention does not apply
        here - and zero is the one fill that is finite through any memory net.
        """
        valid = seq.new_ones(num_chunks * c)
        if pad:
            valid[-pad:] = 0.0
        return valid.reshape(1, num_chunks, c)

    def _shift_targets(self, stored: Tensor, n: int) -> Tensor:
        """Next-latent write targets over the padded grid.

        Position ``t`` targets ``t + 1`` for ``t < n - 1``; the last REAL token
        targets itself (no successor exists). Naively shifting the whole padded
        tensor would hand that token a zero pad as its target - a write that
        teaches the memory to forecast nothing at the end of every sequence.
        The pad's own targets are arbitrary; ``_valid_grid`` zeroes their lr.
        """
        return torch.cat([stored[:, 1:n], stored[:, n - 1 : n], stored[:, n:]], dim=1)

    def _recon_per_token(self, pred: Tensor, v: Tensor, normalize: bool) -> Tensor:
        """Per-token surprise loss against the write target. Energy mode compares
        RMS-normalized (directional) vectors - matching the out_norm'd readout -
        so the memory net's free output-scale mode can't dominate; standard mode
        uses raw MSE. The predictive arm (next-latent target) uses Smooth L1
        (Huber, NextLat), which bounds the surprise on outlier latents so the
        write rule stays stable without a tuned clip.
        """
        if normalize:
            pred = pred * torch.rsqrt(pred.pow(2).mean(-1, keepdim=True) + self.eps)
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.predictive:
            return F.smooth_l1_loss(pred, v, reduction="none", beta=1.0).mean(dim=-1)
        return ((pred - v) ** 2).mean(dim=-1)

    def _surprise_grads(
        self, weights: Weights, keys: Tensor, values: Tensor, lr: Tensor
    ) -> Tuple[Weights, Tensor, Tensor]:
        """Per-sequence gradient of the lr-weighted reconstruction loss.

        keys/values: (b, c, d); lr: (b, c); weights leaves: (b, ...). Returns
        grads (batched like weights), the per-token loss that drives them
        (normalized in energy mode), and the raw per-token loss (for the
        scale-sensitive metric).
        """

        def loss_single(w: Weights, k: Tensor, v: Tensor, step: Tensor):
            pred = functional_call(self.memory_model, w, (k,))
            raw = self._recon_per_token(pred, v, False)
            normed = self._recon_per_token(pred, v, True)
            # Energy mode OPTIMIZES the scale-free form; standard mode optimizes
            # the paper's raw MSE (Eq. 12). Both are reported either way: the
            # readout sits behind out_norm in both modes, so the memory net's
            # output magnitude is a free mode in both, and without the scale-free
            # line a drifting raw surprise is indistinguishable from a memory
            # that stopped learning. That ambiguity cost a run once already.
            driver = normed if self.use_energy else raw
            return (step * driver).sum(), (raw, normed)

        grads, (_, (raw, normed)) = vmap(grad_and_value(loss_single, has_aux=True))(
            weights, keys, values, lr
        )
        driver = normed if self.use_energy else raw
        return grads, driver, raw, normed

    # --- forward -------------------------------------------------------------

    def forward(
        self, seq: Tensor, state: Optional[NeuralMemState] = None
    ) -> Tuple[Tensor, NeuralMemState]:
        """Store ``seq`` into memory and retrieve causally. Returns (out, state).

        Dispatches to a compiled copy of the body during generation, when one
        has been installed (see ``decode_compiled`` and
        ``praxis.generation.decode_backend``). Training and validation always
        take the eager body: the trainer already compiles the whole model, and
        a nested compiled region inside that is a different and unmeasured
        proposition.
        """
        fn = self._decode_forward
        if fn is not None:
            return fn(seq, state)
        return self._forward_impl(seq, state)

    def _forward_impl(
        self, seq: Tensor, state: Optional[NeuralMemState] = None
    ) -> Tuple[Tensor, NeuralMemState]:
        b, n, d = seq.shape
        # Segmentation runs the update on a finer base grid; events merge blocks.
        c = self.segment_block if self.segment else self.chunk_size

        if state is None:
            state = self.init_state(b, seq.device)
        weights, momentum = dict(state.weights), dict(state.momentum)
        second_moment = dict(state.second_moment)

        # Pad to a whole number of chunks; the tail is truncated from the output.
        pad = (c - n % c) % c
        if pad:
            seq = F.pad(seq, (0, 0, 0, pad))
        num_chunks = seq.shape[1] // c
        # Which grid positions are real. The pad is zeros, and a zero token is
        # not a no-op for the write: RMS-normalizing it gives the zero vector,
        # so the surprise loss reads a full-magnitude "predict nothing from
        # nothing" error and the update chases it. Masking the per-token lr is
        # what makes the pad inert - it is the only term multiplying every
        # token's contribution to the surprise, in both modes.
        valid = self._valid_grid(seq, num_chunks, c, pad)
        self._note_chunks(num_chunks, n, c)

        if not self.parallel_scan:
            return self._forward_sequential(
                seq,
                weights,
                momentum,
                second_moment,
                state.seq_index,
                n,
                c,
                num_chunks,
                pad,
            )

        bn = b * num_chunks

        # Test-time update. In energy mode the whole region is detached: no scan
        # trajectory / second-order graph is retained (that trajectory dominates
        # VRAM). Retrieval below stays differentiable.
        with self._update_ctx():
            stored = self.store_norm(seq)
            keys = self.to_keys(stored).unflatten(1, (num_chunks, c))  # (b, nc, c, d)
            # Predictive: store key_t -> stream_{t+1} (next latent), so retrieval
            # is a forecast rather than an echo of the residual. Reading pre-write
            # weights (each chunk sees only earlier chunks' writes) keeps this
            # causal: an interior token's own target is written by its own chunk,
            # invisible at retrieval. The LAST REAL token has no successor, so it
            # falls back to itself (a single self-recon token, harmless) - the
            # shift is taken over the real prefix, never off the end into the
            # pad, which would train it to predict zero.
            # The predictive target is detached explicitly. In energy mode the
            # surrounding no_grad already did it; in standard mode nothing would,
            # and a differentiable next-latent target lets the encoder minimize
            # surprise by collapsing the stream instead of by memorizing it. The
            # recon target is NOT detached - `to_values` is a W_V the paper
            # trains in the outer loop (Eq. 12).
            tgt = (
                self._shift_targets(stored, n).detach()
                if self.predictive
                else self.to_values(stored)
            )
            values = tgt.unflatten(1, (num_chunks, c))
            # Energy mode takes the raw surprise (lr=1) and applies a fixed lr in
            # the Adam step; standard mode weights it by a learned per-token lr.
            # Both are zeroed on the pad, which is what keeps it out of the write.
            if self.use_energy:
                lr = valid.expand(b, num_chunks, c)
            else:
                lr = (self.to_lr(stored).squeeze(-1).sigmoid() * self.max_lr).unflatten(
                    1, (num_chunks, c)
                ) * valid  # (b, nc, c)

            # Surprise for every chunk, taken against the frozen segment-start
            # weights, in one batched pass over (b * nc).
            w0_rep = {
                k: v.repeat_interleave(num_chunks, dim=0) for k, v in weights.items()
            }
            grads, per_token, per_token_raw, per_token_norm = self._surprise_grads(
                w0_rep,
                keys.reshape(bn, c, d),
                values.reshape(bn, c, d),
                lr.reshape(bn, c),
            )
            surprise = {
                k: -g.reshape(b, num_chunks, *g.shape[1:]) for k, g in grads.items()
            }

            new_second = second_moment
            reset_mask = t_event = None
            if self.segment:
                # Per-block surprise (real tokens only), then event boundaries.
                pt = per_token.reshape(b, num_chunks, c)
                s_blocks = (pt * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
                reset_mask, t_event = self._segment(s_blocks)
            if self.use_energy:
                chunk_weights, new_weights, new_momentum, new_second = (
                    self._adam_update(
                        weights,
                        momentum,
                        second_moment,
                        surprise,
                        num_chunks,
                        reset_mask,
                        t_event,
                    )
                )
            else:
                # Learned momentum then weight-decay, each a scan over chunks.
                # The chunk summary the gates read averages REAL positions only:
                # a mean over the padded tail is diluted toward zero, so the
                # trailing chunk's forget/momentum gates would be driven by how
                # far the sequence happened to sit from a chunk boundary.
                chunk_rep = (
                    stored.unflatten(1, (num_chunks, c)) * valid.unsqueeze(-1)
                ).sum(dim=2) / valid.sum(-1).clamp(min=1.0).unsqueeze(
                    -1
                )  # (b, nc, d)
                eta = self.to_momentum(chunk_rep).sigmoid().squeeze(-1)  # (b, nc)
                alpha = self.to_decay(chunk_rep).sigmoid().squeeze(-1)  # (b, nc)
                chunk_weights, new_weights, new_momentum = {}, {}, {}
                for name in self._param_names:
                    if self.use_momentum:
                        s = _affine_scan(eta, surprise[name], momentum[name])
                    else:
                        s = surprise[name]
                    w_t = _affine_scan(1 - alpha, s, weights[name])  # (b, nc, *p)
                    chunk_weights[name] = w_t
                    new_weights[name] = w_t[:, -1]
                    new_momentum[name] = (
                        s[:, -1] if self.use_momentum else momentum[name]
                    )

        # Retrieve: each chunk reads the state *before* its own writes (causal),
        # i.e. W0 for chunk 0 and the previous chunk's weights thereafter.
        retrieve_w = {
            k: torch.cat([weights[k].unsqueeze(1), chunk_weights[k][:, :-1]], dim=1)
            for k in self._param_names
        }
        queries = self.to_queries(self.retrieve_norm(seq)).unflatten(1, (num_chunks, c))
        retrieved = vmap(lambda w, q: functional_call(self.memory_model, w, (q,)))(
            {k: v.reshape(bn, *v.shape[2:]) for k, v in retrieve_w.items()},
            queries.reshape(bn, c, d),
        )  # (bn, c, d)
        retrieved = retrieved.reshape(b, num_chunks * c, d)
        retrieved = self.combine(self.out_norm(retrieved))[:, :n]

        with torch.no_grad():
            # Raw surprise (scale-sensitive, kept for continuity) and, in energy
            # mode, the scale-free surprise the update actually optimizes. Both
            # are averaged over REAL positions only: the pad no longer drives the
            # update, so letting it into the mean would report a surprise the
            # memory is not being asked to reduce, biased by how far the sequence
            # sat from a chunk boundary.
            vmask = valid.reshape(1, -1)
            vsum = vmask.sum().clamp(min=1.0)
            self.last_surprise = (per_token_raw.reshape(b, -1) * vmask).sum() / (
                vsum * b
            )
            self.last_surprise_norm = (per_token_norm.reshape(b, -1) * vmask).sum() / (
                vsum * b
            )
            # Output magnitude relative to the stream: catches the model routing
            # around the memory (combine -> 0). Per-sequence write magnitude:
            # confirms the test-time update is doing real work (not collapsing).
            self.last_gain = retrieved.norm() / (seq[:, :n].norm() + self.eps)
            wnum = sum(
                (new_weights[p] - weights[p]).pow(2).sum() for p in self._param_names
            )
            wden = sum(weights[p].pow(2).sum() for p in self._param_names)
            self.last_write = (wnum / (wden + self.eps)).sqrt()
            # The same write in function space - see _readout_delta. Pairs with
            # last_write: identical numerator (this call's update), read out
            # through the memory net instead of measured against ||W0||.
            if self._probe_due():
                self.last_adapt = self._readout_delta(retrieved, weights, queries, n)

            # Event sizes: inter-boundary spans at the base-block grid, reported
            # in tokens (whole blocks * segment_block), so they're bounded by
            # [segment_block, chunk_size]. A padded trailing block still counts
            # as a full grid block - the update masks its pad, but the segment
            # granularity is the grid, so min never dips below one block.
            if self.segment and reset_mask is not None:
                sizes = []
                for bi in range(b):
                    bounds = reset_mask[bi].nonzero().flatten().tolist() + [num_chunks]
                    sizes += [(e - a) * c for a, e in zip(bounds, bounds[1:])]
                sizes = seq.new_tensor(sizes)
                self.last_event_mean = sizes.mean()
                self.last_event_min = sizes.min()
                self.last_event_max = sizes.max()

        new_state = NeuralMemState(
            state.seq_index + n, new_weights, new_momentum, new_second
        )
        return retrieved, new_state

    def _forward_sequential(
        self, seq, weights, momentum, second_moment, seq_index, n, c, num_chunks, pad
    ):
        """Chunk-at-a-time equivalent of the parallel store+retrieve. Carries the
        running weights/EMAs so the full (b, nc, *p) surprise and weight
        trajectories are never materialized. Numerics match the parallel path:
        every chunk's surprise is still taken against the frozen W0, retrieval
        still reads the pre-write weights, and segmentation/Adam are the same
        recurrence walked in order instead of scanned."""
        b = seq.shape[0]
        W0 = weights  # frozen segment-start weights; surprise grads taken against these
        w = dict(weights)  # running weights (retrieval reads these pre-write)
        m, v = dict(momentum), dict(second_moment)

        valid = self._valid_grid(seq, num_chunks, c, pad)[0]  # (nc, c) tail-pad mask
        queries = self.to_queries(self.retrieve_norm(seq)).unflatten(1, (num_chunks, c))

        # Predictive write target: the next-latent stream, stop-gradded, sliced
        # per chunk in the loop. Materializing the (b, N, d) stream is cheap (it
        # is the size of seq, not the (b, nc, *p) weight trajectory the sequential
        # path exists to avoid), so the low-VRAM property is preserved.
        if self.predictive:
            with torch.no_grad():
                sn = self.store_norm(seq)
                pred_target = self._shift_targets(sn, n)

        retrieved_chunks, reset_list = [], []
        raw_sum = drv_sum = seq.new_zeros(())
        raw_cnt = drv_cnt = 0
        if self.segment:
            csum, csq = seq.new_zeros(b), seq.new_zeros(b)
            run = torch.zeros(b, dtype=torch.long, device=seq.device)

        for i in range(num_chunks):
            # Retrieve chunk i against the pre-write weights (W0 for i=0). Stays
            # differentiable (query path always; the update path too in standard).
            retrieved_chunks.append(
                vmap(lambda wi, qi: functional_call(self.memory_model, wi, (qi,)))(
                    w, queries[:, i]
                )
            )

            with self._update_ctx():
                stored = self.store_norm(seq[:, i * c : (i + 1) * c])
                k_i = self.to_keys(stored)
                val_i = (
                    pred_target[:, i * c : (i + 1) * c]
                    if self.predictive
                    else self.to_values(stored)
                )
                # Pad-masked exactly as the parallel path: valid[i] is this
                # chunk's row of the tail mask.
                lr_i = (
                    valid[i].expand(b, c)
                    if self.use_energy
                    else self.to_lr(stored).squeeze(-1).sigmoid()
                    * self.max_lr
                    * valid[i]
                )
                grads, driver, raw, normed = self._surprise_grads(W0, k_i, val_i, lr_i)
                surprise = {k: -g for k, g in grads.items()}
                # Real positions only, matching the parallel path: the pad is
                # masked out of the update, so it stays out of the mean too.
                raw_sum = raw_sum + (raw * valid[i]).sum()
                drv_sum = drv_sum + (normed * valid[i]).sum()
                raw_cnt += int(valid[i].sum()) * b
                drv_cnt += int(valid[i].sum()) * b

                # Event boundary for this chunk, from causal stats over prior
                # blocks (matches _segment). Resets the EMAs and re-bases t_event.
                if self.segment:
                    s_block = (driver * valid[i]).sum(-1) / valid[i].sum().clamp(
                        min=1.0
                    )
                    mean = csum / max(i, 1)
                    std = (csq / max(i, 1) - mean * mean).clamp(min=0.0).sqrt()
                    if i == 0:
                        is_b = torch.ones(b, dtype=torch.bool, device=seq.device)
                    else:
                        # Relative tolerance guards the per-block variance against
                        # catastrophic cancellation: on a near-constant stream it
                        # would otherwise round to a spurious spike.
                        thresh = mean + self.segment_gamma * std + 1e-5 * mean.abs()
                        is_b = (s_block > thresh) | (run >= self._cap_blocks)
                    run = torch.where(is_b, torch.ones_like(run), run + 1)
                    t_event = run
                    csum, csq = csum + s_block, csq + s_block * s_block
                    reset_list.append(is_b)
                else:
                    is_b = None
                    t_event = torch.full((b,), i + 1, device=seq.device)

                self._update_chunk(
                    w,
                    m,
                    v,
                    surprise,
                    is_b,
                    t_event,
                    b,
                    stored,
                    valid[i],
                    W0,
                    num_chunks,
                )

        retrieved = torch.cat(retrieved_chunks, dim=1)
        retrieved = self.combine(self.out_norm(retrieved))[:, :n]

        with torch.no_grad():
            self.last_surprise = raw_sum / max(raw_cnt, 1)
            self.last_surprise_norm = drv_sum / max(drv_cnt, 1)
            self.last_gain = retrieved.norm() / (seq[:, :n].norm() + self.eps)
            wnum = sum((w[p] - W0[p]).pow(2).sum() for p in self._param_names)
            wden = sum(W0[p].pow(2).sum() for p in self._param_names)
            self.last_write = (wnum / (wden + self.eps)).sqrt()
            if self._probe_due():
                self.last_adapt = self._readout_delta(retrieved, W0, queries, n)
            if self.segment and reset_list:
                reset_mask = torch.stack(reset_list, dim=1)
                sizes = []
                for bi in range(b):
                    bounds = reset_mask[bi].nonzero().flatten().tolist() + [num_chunks]
                    sizes += [(e - a) * c for a, e in zip(bounds, bounds[1:])]
                sizes = seq.new_tensor(sizes)
                self.last_event_mean = sizes.mean()
                self.last_event_min = sizes.min()
                self.last_event_max = sizes.max()

        return retrieved, NeuralMemState(seq_index + n, w, m, v)

    def _update_chunk(
        self,
        w,
        m,
        v,
        surprise,
        is_b,
        t_event,
        b,
        stored,
        valid=None,
        w0=None,
        num_chunks=1,
    ):
        """One chunk of the test-time weight update, in place on the running
        ``w``/``m``/``v`` dicts. Energy mode = the Adam-style rule; standard mode
        = learned momentum/decay gates. Mirrors ``_adam_update`` / the standard
        branch for a single chunk."""
        if self.use_energy:
            beta1 = stored.new_full((b,), self.beta1)
            beta2 = stored.new_full((b,), self.beta2)
            if is_b is not None:  # fresh moments at an event start
                beta1 = beta1.masked_fill(is_b, 0.0)
                beta2 = beta2.masked_fill(is_b, 0.0)
            c1 = 1.0 - self.beta1 ** t_event.to(stored.dtype)
            c2 = 1.0 - self.beta2 ** t_event.to(stored.dtype)
            for name in self._param_names:
                s = surprise[name]
                shp = (b,) + (1,) * (s.dim() - 1)
                v[name] = beta2.reshape(shp) * v[name] + (1.0 - self.beta2) * s * s
                if self.use_momentum:
                    m[name] = beta1.reshape(shp) * m[name] + (1.0 - self.beta1) * s
                    m_hat = m[name] / c1.reshape(shp)
                else:
                    m_hat = s
                u = m_hat / ((v[name] / c2.reshape(shp)).sqrt() + self.eps)
                # Relative step, from the SEGMENT-START weights - see
                # _step_scale. w0 is None only for direct calls in tests.
                scale = (
                    1.0
                    if w0 is None
                    else self._step_scale(w0, name, s.dim(), b, num_chunks)
                )
                w[name] = (1.0 - self.weight_decay) * w[name] + self.max_lr * u * scale
        else:
            # Real positions only, matching the parallel path: a mean over the
            # padded tail is diluted toward zero and would drive the trailing
            # chunk's forget/momentum gates by the pad length.
            if valid is None:
                chunk_rep = stored.mean(dim=1)
            else:
                chunk_rep = (stored * valid.unsqueeze(-1)).sum(
                    dim=1
                ) / valid.sum().clamp(min=1.0)
            eta = self.to_momentum(chunk_rep).sigmoid().squeeze(-1)
            alpha = self.to_decay(chunk_rep).sigmoid().squeeze(-1)
            for name in self._param_names:
                s = surprise[name]
                shp = (b,) + (1,) * (s.dim() - 1)
                if self.use_momentum:
                    m[name] = eta.reshape(shp) * m[name] + s
                    s = m[name]
                w[name] = (1.0 - alpha.reshape(shp)) * w[name] + s

    # --- introspection -------------------------------------------------------

    @torch.no_grad()
    def memory_loss(self, seq: Tensor, weights: Weights) -> Tensor:
        """Mean reconstruction loss of ``weights`` on ``seq``'s associations,
        in the same (normalized in energy mode) space the update optimizes.
        Lower means the memory has better memorized the sequence.
        """
        stored = self.store_norm(seq)
        keys, values = self.to_keys(stored), self.to_values(stored)
        pred = vmap(lambda w, k: functional_call(self.memory_model, w, (k,)))(
            weights, keys
        )
        return self._recon_per_token(pred, values, self.use_energy).mean()
