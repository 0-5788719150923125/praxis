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
moment) give a scale-invariant step, so a constant learning rate stays safe and
there are no untrained gate heads. Training the encoder on the reconstruction
energy would collapse (value -> 0), so instead the key projection is tied to the
query projection - the shared addressing map learns on the task through
retrieval - and the value side is fixed to identity, leaving ``combine`` to adapt
content. The backbone connects through that retrieval and the residual skip.
The reconstruction is measured on RMS-normalized vectors (matching the
out_norm'd readout), so the memory net's free output-scale mode can't dominate
the energy.
"""

import contextlib
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
        assert write_objective in ("recon", "predictive"), write_objective
        assert (
            write_objective == "recon" or use_energy
        ), "predictive write_objective requires use_energy"
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

        # Diagnostics from the last store pass, logged as metrics: cold-start
        # surprise, the memory's output magnitude relative to the stream, and
        # the relative size of the test-time weight update.
        self.last_surprise: Optional[Tensor] = None
        self.last_surprise_norm: Optional[Tensor] = None
        self.last_gain: Optional[Tensor] = None
        self.last_write: Optional[Tensor] = None
        # Event-size stats from the last segmented store pass (tokens per event).
        self.last_event_mean: Optional[Tensor] = None
        self.last_event_min: Optional[Tensor] = None
        self.last_event_max: Optional[Tensor] = None

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
        spike = s_blocks > mean + self.segment_gamma * var.sqrt()
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
            w_t = _affine_scan(keep, self.max_lr * u, weights[name])  # (b, nc, *p)
            chunk_weights[name] = w_t
            new_weights[name] = w_t[:, -1]
            new_m[name] = m[:, -1] if self.use_momentum else momentum[name]
            new_v[name] = v[:, -1]
        return chunk_weights, new_weights, new_m, new_v

    # --- functional grad of the surprise loss --------------------------------

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
            driver = self._recon_per_token(pred, v, self.use_energy)  # (c,)
            raw = self._recon_per_token(pred, v, False)
            return (step * driver).sum(), (driver, raw)

        grads, (_, (driver, raw)) = vmap(grad_and_value(loss_single, has_aux=True))(
            weights, keys, values, lr
        )
        return grads, driver, raw

    # --- forward -------------------------------------------------------------

    def forward(
        self, seq: Tensor, state: Optional[NeuralMemState] = None
    ) -> Tuple[Tensor, NeuralMemState]:
        """Store ``seq`` into memory and retrieve causally. Returns (out, state)."""
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
            # invisible at retrieval. The global last token has no successor, so it
            # falls back to itself (a single self-recon token, harmless).
            tgt = (
                torch.cat([stored[:, 1:], stored[:, -1:]], dim=1)
                if self.predictive
                else self.to_values(stored)
            )
            values = tgt.unflatten(1, (num_chunks, c))
            # Energy mode takes the raw surprise (lr=1) and applies a fixed lr in
            # the Adam step; standard mode weights it by a learned per-token lr.
            if self.use_energy:
                lr = stored.new_ones(b, num_chunks, c)
            else:
                lr = (self.to_lr(stored).squeeze(-1).sigmoid() * self.max_lr).unflatten(
                    1, (num_chunks, c)
                )  # (b, nc, c)

            # Surprise for every chunk, taken against the frozen segment-start
            # weights, in one batched pass over (b * nc).
            w0_rep = {
                k: v.repeat_interleave(num_chunks, dim=0) for k, v in weights.items()
            }
            grads, per_token, per_token_raw = self._surprise_grads(
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
                valid = per_token.new_ones(num_chunks * c)
                if pad:
                    valid[-pad:] = 0.0
                valid = valid.reshape(num_chunks, c)
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
                chunk_rep = stored.unflatten(1, (num_chunks, c)).mean(dim=2)  # (b,nc,d)
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
            # mode, the scale-free surprise the update actually optimizes.
            self.last_surprise = per_token_raw.mean()
            self.last_surprise_norm = per_token.mean() if self.use_energy else None
            # Output magnitude relative to the stream: catches the model routing
            # around the memory (combine -> 0). Per-sequence write magnitude:
            # confirms the test-time update is doing real work (not collapsing).
            self.last_gain = retrieved.norm() / (seq[:, :n].norm() + self.eps)
            wnum = sum(
                (new_weights[p] - weights[p]).pow(2).sum() for p in self._param_names
            )
            wden = sum(weights[p].pow(2).sum() for p in self._param_names)
            self.last_write = (wnum / (wden + self.eps)).sqrt()

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

        valid = seq.new_ones(num_chunks * c).reshape(num_chunks, c)  # tail-pad mask
        if pad:
            valid.view(-1)[-pad:] = 0.0
        queries = self.to_queries(self.retrieve_norm(seq)).unflatten(1, (num_chunks, c))

        # Predictive write target: the next-latent stream, stop-gradded, sliced
        # per chunk in the loop. Materializing the (b, N, d) stream is cheap (it
        # is the size of seq, not the (b, nc, *p) weight trajectory the sequential
        # path exists to avoid), so the low-VRAM property is preserved.
        if self.predictive:
            with torch.no_grad():
                sn = self.store_norm(seq)
                pred_target = torch.cat([sn[:, 1:], sn[:, -1:]], dim=1)

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
                lr_i = (
                    stored.new_ones(b, c)
                    if self.use_energy
                    else self.to_lr(stored).squeeze(-1).sigmoid() * self.max_lr
                )
                grads, driver, raw = self._surprise_grads(W0, k_i, val_i, lr_i)
                surprise = {k: -g for k, g in grads.items()}
                raw_sum = raw_sum + raw.sum()
                drv_sum = drv_sum + driver.sum()
                raw_cnt += raw.numel()
                drv_cnt += driver.numel()

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

                self._update_chunk(w, m, v, surprise, is_b, t_event, b, stored)

        retrieved = torch.cat(retrieved_chunks, dim=1)
        retrieved = self.combine(self.out_norm(retrieved))[:, :n]

        with torch.no_grad():
            self.last_surprise = raw_sum / max(raw_cnt, 1)
            self.last_surprise_norm = (
                drv_sum / max(drv_cnt, 1) if self.use_energy else None
            )
            self.last_gain = retrieved.norm() / (seq[:, :n].norm() + self.eps)
            wnum = sum((w[p] - W0[p]).pow(2).sum() for p in self._param_names)
            wden = sum(W0[p].pow(2).sum() for p in self._param_names)
            self.last_write = (wnum / (wden + self.eps)).sqrt()
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

    def _update_chunk(self, w, m, v, surprise, is_b, t_event, b, stored):
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
                w[name] = (1.0 - self.weight_decay) * w[name] + self.max_lr * u
        else:
            chunk_rep = stored.mean(dim=1)
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
