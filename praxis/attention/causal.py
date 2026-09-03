"""
CausalAttention module for causal language modeling using PyTorch's FlexAttention.
Based on the efficient attention implementation with block masking support.
"""

import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.encoding import ENCODING_REGISTRY

# Suppress verbose compile-time logs
os.environ["TORCHDYNAMO_EXTENDED_ADVICE"] = "0"


class CausalAttention(nn.Module):
    """
    Causal self-attention using PyTorch's FlexAttention API.
    Provides efficient attention computation with customizable block masking.
    Supports optional sliding window for efficient long-sequence inference.
    """

    def __init__(
        self,
        config,
        dropoff: str = None,
        dropoff_every: bool = False,
    ) -> None:
        """
        Initialize CausalAttention module.

        Args:
            config: Configuration object containing attention parameters.
                ``config.encoding`` selects any entry from ENCODING_REGISTRY
                (rope, alibi, hope, nope, ...). Optional ``config.window_size``
                (int) enables sliding window attention.
        """
        super().__init__()

        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.head_dim = getattr(config, "head_size") or hidden_size // self.num_heads
        self.dropout_p = config.dropout
        self.causal = config.causal
        self.window_size = getattr(config, "window_size", None)
        # Dropoff ablation (next/dropoff.md): withhold the causal tip so the
        # model must lean on delayed context. ``dropoff`` is the mode - None
        # (off), "shift" (uniform K/V delay) or "warp" (feature-dependent value
        # sink at the tip) - owned by ATTENTION_REGISTRY profiles (e.g.
        # arc_dropoff), not config.
        #
        # ``dropoff_every`` is the SCHEDULE, orthogonal to the mode:
        #
        #   False (default) - fires only when ``current_depth`` reaches
        #     ``depth - num_layers``. The stated mechanism (next/dropoff.md)
        #     is that the surrounding beats recorrect for the sunk one.
        #
        #     THIS SCHEDULE IS VERY NEARLY INERT, and the comment here used to
        #     claim otherwise. Under KL halting the TRAINING depth budget is
        #     sampled, not fixed: ``KLHalting.get_depth`` returns
        #     ``_sample_loop_count() * num_layers`` from a log-normal Poisson
        #     while training. (``check`` is the inference-only half; reading
        #     that one alone hides this.) At depth 6 / num_layers 1, so
        #     max_loops 6 and r_bar 2.5, the loop count is 6 on 7.0% of steps
        #     and only then is step 5 reached: 0.07 firings per step, 2.3% of
        #     the passes that execute (E[loops] is 3.06, not 6). Whatever the
        #     arc configs have been measuring, it is not this ablation.
        #   True - every pass, like ghostmax. ~44x the exposure above, and the first schedule under which dropoff is
        #     a real intervention rather than a rounding error. It also removes
        #     the chosen step, which is worth something on its own: the default
        #     is a hand-picked beat whose realized rate turns out to be set by
        #     an unrelated sampling distribution.
        #
        # The cost of True, stated so a null result stays readable: the tip is
        # then permanently absent from the value path at every depth - a recency
        # PRIOR rather than an ablation - and the envelope is anchored to the
        # current forward's T, so under a sequence curriculum its shape moves
        # with the batch.
        #
        # Neither is measured, and given the exposure numbers above, dropoff has
        # never really been measured at all.
        self.dropoff_mode = dropoff
        self.dropoff_every = bool(dropoff_every)
        if dropoff is None:
            self.dropoff_step = None
        else:
            layers = getattr(config, "num_layers", None) or config.depth
            self.dropoff_step = max(0, config.depth - layers)

        # Positional encoding lives entirely in the registry-built module:
        # before_scores mutates Q/K (RoPE/HoPE), build_score_mod returns the
        # FlexAttention closure (ALiBi), after_scores adds bias on materialized
        # scores (the CPU/ghost-aware path). NoPE/RoPE/HoPE return None from
        # build_score_mod, so the FlexAttention path skips the closure entirely.
        self.encoding = ENCODING_REGISTRY[config.encoding](config)
        # Plain-string introspection field (used by Prismatic router tests
        # and diagnostics to check which encoding an expert was built with).
        self.pos_type = config.encoding

        # QKV projection - separate sizes for Q (with num_queries) and K/V
        # Q: num_query_heads * head_dim
        # K: num_heads * head_dim
        # V: num_heads * head_dim
        qkv_dim = (
            self.num_query_heads * self.head_dim + 2 * self.num_heads * self.head_dim
        )
        self.qkv = nn.Linear(hidden_size, qkv_dim, bias=False)

        # Output projection
        self.output = nn.Linear(
            self.num_query_heads * self.head_dim, hidden_size, bias=False
        )

        # Dropout layers
        self.dropout = nn.Dropout(self.dropout_p)

        # Try to import FlexAttention components
        self.flex_attention = None
        self.create_block_mask = None
        self.and_masks = None
        self._import_flex_attention()

        # Cache for block masks at different sequence lengths
        # Key is (seq_len, kv_len, device_str) tuple
        self.block_mask_cache = {}

    @classmethod
    def patch_config(cls, config) -> None:
        """Correct the config fields that would otherwise misdescribe this module.

        Most mechanisms build exactly what the config asks for, so this is a
        no-op for them. It exists for the ones that cannot - a mechanism that
        fixes its own head count, say - because the config is not a private
        argument list: it is serialized to ``config.json``, rendered in the
        blueprint tab, and read by every other module in the stack. A field
        the attention silently disagrees with is a field that describes a
        model nobody built.

        Called for the selected ``attention_type`` by
        :func:`praxis.attention.patch_attention_config`, and again from the
        implementing class's own ``__init__``, so it MUST be idempotent.
        """

    @contextmanager
    def head_budget(self, kv_keep: Tensor):
        """Coherently restrict attention to a subset of KV heads (and their GQA
        query-head groups) for one forward, then restore - the mixture-of-widths
        recipe for attention.

        Every per-head tensor is sliced in lockstep so the forward stays well
        formed: the fused QKV rows, the output columns, and (where present) the
        gate, the per-depth QKV bias, and the Infini ``betas`` / memory buffers.
        The head counts the ``.view``s read are dropped to match. Weight slices
        are differentiable gathers, so only the heads that ran get gradient. The
        per-depth output bias is per-hidden, not per-head, so it is left whole.
        """
        nq, hd = self.num_queries, self.head_dim
        H, QH = self.num_heads, self.num_query_heads
        dev = self.qkv.weight.device
        kv_keep = kv_keep.to(dev).sort().values
        # GQA: each kept KV head brings its nq query heads.
        q_keep = (kv_keep.unsqueeze(1) * nq + torch.arange(nq, device=dev)).reshape(-1)

        def chans(heads):  # head indices -> flat per-head-dim channel indices
            return (heads.unsqueeze(1) * hd + torch.arange(hd, device=dev)).reshape(-1)

        q_ch = chans(q_keep)
        qkv_rows = torch.cat(
            [q_ch, QH * hd + chans(kv_keep), QH * hd + H * hd + chans(kv_keep)]
        )

        swaps = []  # (store_dict, name, original_tensor)

        def swap(module, name, new):
            store = (
                module._parameters if name in module._parameters else module._buffers
            )
            swaps.append((store, name, store[name]))
            store[name] = new

        swap(self.qkv, "weight", self.qkv.weight[qkv_rows])
        if self.qkv.bias is not None:
            swap(self.qkv, "bias", self.qkv.bias[qkv_rows])
        swap(self.output, "weight", self.output.weight[:, q_ch])
        if getattr(self, "attention_gating", False) and hasattr(self, "gate"):
            swap(self.gate, "weight", self.gate.weight[q_ch])
            if self.gate.bias is not None:
                swap(self.gate, "bias", self.gate.bias[q_ch])
        if hasattr(self, "depth_bias") and hasattr(self, "qkv_dim"):
            # depth_bias packs [qkv_dim qkv cols | hidden_size output cols];
            # prune only the qkv columns, keep the output tail intact.
            out_cols = torch.arange(
                self.qkv_dim, self.depth_bias.weight.shape[1], device=dev
            )
            keep_cols = torch.cat([qkv_rows, out_cols])
            swap(self.depth_bias, "weight", self.depth_bias.weight[:, keep_cols])
        for name in ("betas", "init_mem", "init_z"):  # Infini per-head state
            if hasattr(self, name):
                swap(self, name, getattr(self, name)[:, q_keep])

        # Scalars derived from the head counts have to narrow along with the
        # tensors they index, and be restored with them. ``qkv_dim`` is the
        # split point between depth_bias's qkv columns and its output tail;
        # subclasses re-read it on every forward (arc.py:131, :146), so leaving
        # it at full width makes the sliced table index its own output columns
        # as if they were qkv channels.
        saved_scalars = {
            name: getattr(self, name)
            for name in ("num_query_heads", "num_heads", "qkv_dim")
            if hasattr(self, name)
        }
        self.num_query_heads, self.num_heads = int(q_keep.numel()), int(kv_keep.numel())
        if "qkv_dim" in saved_scalars:
            self.qkv_dim = int(qkv_rows.numel())
        try:
            yield
        finally:
            for store, name, old in reversed(swaps):
                store[name] = old
            for name, old in saved_scalars.items():
                setattr(self, name, old)

    def _import_flex_attention(self) -> None:
        """Import FlexAttention components (GPU only - falls back to SDPA on CPU)."""
        try:
            from torch.nn.attention.flex_attention import (
                and_masks,
                create_block_mask,
                flex_attention,
            )

            self.flex_attention = flex_attention
            self.create_block_mask = create_block_mask
            self.and_masks = and_masks
        except ImportError:
            print("[CausalAttention] FlexAttention not available, using SDPA fallback")
            self.flex_attention = None
            self.create_block_mask = None
            self.and_masks = None

    def _build_mask_mod(self):
        """Build a mask_mod closure following the FlexAttention pattern.

        Captures primitive values (not self) so the resulting function is
        compatible with torch.compile tracing inside create_block_mask.
        """
        window_size = self.window_size  # capture as plain int or None

        def ghost_causal_mask(b, h, q_idx, kv_idx):
            # Ghost token (kv_idx=0) is always accessible.
            # Actual tokens: kv_idx=1 is position 0, so q_idx + 1 >= kv_idx.
            return (kv_idx == 0) | (q_idx + 1 >= kv_idx)

        if window_size is None:
            return ghost_causal_mask

        def ghost_sliding_window_mask(b, h, q_idx, kv_idx):
            # Ghost token must also pass here, otherwise and_masks kills it.
            # Actual tokens: limit to window_size positions (ghost-shifted).
            return (kv_idx == 0) | (q_idx - (kv_idx - 1) <= window_size)

        return self.and_masks(ghost_causal_mask, ghost_sliding_window_mask)

    def _create_causal_mask(
        self, q_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a block mask with ghostmax support, optional sliding window.

        Uses composable mask_mod functions per the FlexAttention API:
        causal and sliding window constraints are composed via and_masks.

        Args:
            q_len: Query sequence length (number of actual positions)
            kv_len: Key/Value sequence length (should be q_len + 1 for ghost)
            device: Device to create mask on

        Returns:
            Block mask for attention with ghostmax
        """
        cache_key = (q_len, kv_len, str(device))

        if cache_key in self.block_mask_cache:
            return self.block_mask_cache[cache_key]

        block_mask = self.create_block_mask(
            self._build_mask_mod(),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        self.block_mask_cache[cache_key] = block_mask
        return block_mask

    def _use_cpu_fallback(self, device: torch.device) -> bool:
        """
        Determine if we should use CPU fallback instead of flex_attention.

        FlexAttention uses torch.compile internally, which doesn't support CPU.

        Args:
            device: The device tensors are on

        Returns:
            True if we should use SDPA fallback, False if we can use flex_attention
        """
        # Use fallback if flex_attention isn't available
        if self.flex_attention is None:
            return True

        # Use fallback on CPU (flex_attention compilation fails on CPU)
        if device.type == "cpu":
            return True

        return False

    def _maybe_dropoff(self, k: Tensor, v: Tensor, current_depth: int):
        """Dropoff ablation (next/dropoff.md): withhold the causal tip so the
        model leans on delayed context. Applied to real K/V before the zero
        ghost is prepended. Fires at ``dropoff_step`` only, or at every pass
        under ``dropoff_every``.

        TRAINING ONLY, and that is a NECESSITY of this envelope rather than a
        design position. The warp is anchored to the current forward's ``T``,
        and a cached decode passes ``T = 1``: ``dist = [0]``, so
        ``warp = 1 - exp(0) = 0`` and the token's value is multiplied by ZERO.
        ``_adjust_kv`` runs before the cache write (see
        ``InfiniAttention.forward``), so every V entering the cache would be
        exactly zero and the attention branch would output zero at every decode
        step. Not a distribution shift - a dead branch.

        The train/inference asymmetry is real and uncorrected either way. Dropout
        gets away with it by applying its expectation at test time; a
        multiplicative envelope in [0, 1] has no analogous rescaling, so the two
        coherent designs are (a) train-only, as here, or (b) on at both, which
        needs the envelope re-anchored to ABSOLUTE sequence position so a
        one-token forward knows where the tip is. Neither is measured. (b) also
        means the model never reads the most recent token's value at generation,
        which is a different model, not a consistency fix.

        Before this gate, whether dropoff fired at inference depended on whether
        KL halting stopped the loop before the last pass
        (``praxis/halting/kl.py`` returns False while training, so halting is an
        inference-only mechanism). Gated here rather than at each call site so
        arc, single-head arc and kaleidoscope all inherit it.

        Two modes (``dropoff_mode``):

        - ``shift``: a causal (pad, not wrap) shift of K/V one position toward the
          future, so every query's newest reachable key is its predecessor. Crude
          and uniform.
        - ``warp``: the dual of ghostmax's positionless start-sink - a
          feature-dependent envelope on V that sinks the tip (last position -> 0)
          and recovers backward at a per-feature rate. Attending to the tip then
          injects ~0 per feature (the sink is at the tip, not the start), and the
          envelope modulates the value stream backward from it. Feature-dependence
          rides the value, not the attention weight (weights are per-head, not
          per-feature).
        """
        if self.dropoff_step is None or not self.training:
            return k, v
        if not self.dropoff_every and current_depth != self.dropoff_step:
            return k, v
        if self.dropoff_mode == "warp":
            return k, self._dropoff_warp_value(v)
        # default: shift
        k = torch.cat([torch.zeros_like(k[:, :, :1]), k[:, :, :-1]], dim=2)
        v = torch.cat([torch.zeros_like(v[:, :, :1]), v[:, :, :-1]], dim=2)
        return k, v

    @staticmethod
    def _dropoff_warp_value(v: Tensor) -> Tensor:
        """Per-feature envelope, 0 at the tip and recovering backward at a
        per-feature rate: ``1 - exp(-(T-1-t) * rate_d)``. Sinks the most-recent
        position's value per feature; the start is left intact."""
        T, D = v.shape[2], v.shape[3]
        dist = torch.arange(T - 1, -1, -1, device=v.device, dtype=v.dtype)  # 0 at tip
        rate = torch.linspace(
            0.5, 2.0, D, device=v.device, dtype=v.dtype
        )  # per-feature
        warp = 1.0 - torch.exp(-dist[:, None] * rate[None, :])  # [T, D], 0 at the tip
        return v * warp

    def _ghost_aware_attention(
        self,
        q: Tensor,
        k_ghost: Tensor,
        v_ghost: Tensor,
        seq_len: int,
        is_gqa: bool,
    ) -> Tensor:
        """Manual masked attention preserving the ghost column.

        Inputs already include the zero ghost at kv index 0
        (``k_ghost`` / ``v_ghost`` have shape ``[B, H, seq_len+1, D]``).
        Implements softmax1 by leaving the ghost in the softmax denominator
        - the ghost's zero V contributes nothing to the output but adds an
        extra ``exp(0)=1`` term to the sum, exactly matching
        :func:`~praxis.functional.ghostmax`.

        Args:
            q: Query tensor ``[B, num_query_heads, seq_len, head_dim]``.
            k_ghost: Key tensor including ghost at index 0
                ``[B, num_heads, seq_len+1, head_dim]``.
            v_ghost: Value tensor including ghost at index 0.
            seq_len: Number of real (non-ghost) positions.
            is_gqa: Whether to expand K/V across query-head groups.
        """
        device = q.device
        batch_size, _, _, head_dim = q.shape

        if is_gqa:
            k_exp = k_ghost.repeat_interleave(self.num_queries, dim=1)
            v_exp = v_ghost.repeat_interleave(self.num_queries, dim=1)
        else:
            k_exp, v_exp = k_ghost, v_ghost

        scale = 1.0 / (head_dim**0.5)
        scores = (q @ k_exp.transpose(-2, -1)) * scale  # [B, Hq, S, S+1]

        # Apply the encoding's post-score bias (no-op for RoPE/HoPE/NoPE).
        # Strip the ghost column (kv_idx=0) so it never receives bias, run
        # after_scores on the real keys, then put the ghost column back.
        if seq_len > 0:
            ghost_col = scores[..., :1]
            real_scores = self.encoding.after_scores(scores[..., 1:])
            scores = torch.cat([ghost_col, real_scores], dim=-1)

        # Build mask: ghost (kv_idx=0) always reachable; real keys
        # gated by causal + optional sliding window.
        q_pos = torch.arange(seq_len, device=device)
        kv_pos_real = torch.arange(seq_len + 1, device=device) - 1
        if self.causal:
            allowed = q_pos.unsqueeze(-1) >= kv_pos_real.unsqueeze(0)
        else:
            allowed = torch.ones(seq_len, seq_len + 1, dtype=torch.bool, device=device)
        if self.window_size is not None:
            within_window = (
                q_pos.unsqueeze(-1) - kv_pos_real.unsqueeze(0)
            ) <= self.window_size
            allowed = allowed & within_window
        # Ghost column always passes.
        allowed[:, 0] = True
        scores = scores.masked_fill(~allowed.view(1, 1, seq_len, seq_len + 1), -1e9)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p)
        return weights @ v_exp

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        """
        Forward pass of the FlexAttention module.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor (currently ignored - use causal masking)
            past_key_values: Optional cache for key/value pairs (not currently supported)
            block_ids: Optional tensor indicating block structure
            current_depth: Current depth in the network (for caching)

        Returns:
            Tuple containing:
            - Output tensor after attention and projection
            - Updated cache (if using caching)
            - Auxiliary loss value (always 0 for this implementation)
        """
        batch_size, seq_len, _ = inputs.shape

        # Note: attention_mask is currently not used with FlexAttention.
        # Causal masking is handled by block_mask.
        # Padding masks could be implemented via score_mod if needed.

        # Calculate QKV
        qkv = self.qkv(inputs)

        # Split into Q, K, V with proper dimensions
        q_dim = self.num_query_heads * self.head_dim
        kv_dim = self.num_heads * self.head_dim

        q = qkv[..., :q_dim]
        k = qkv[..., q_dim : q_dim + kv_dim]
        v = qkv[..., q_dim + kv_dim :]

        # Reshape for multi-head attention
        # (B, T, num_heads * head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(
            1, 2
        )

        # For K and V, we always use num_heads (not num_query_heads) for GQA support
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply positional encoding to Q/K (no-op for ALiBi/NoPE, rotates
        # for RoPE/HoPE). Must happen before ghostmax so the ghost token
        # remains a pristine zero.
        q, k, v = self.encoding.before_scores(
            q, k, v, current_depth=current_depth, positions=positions
        )

        # Dropoff ablation: optionally withhold the causal tip at one depth step.
        k, v = self._maybe_dropoff(k, v, current_depth)

        # Ghostmax: Prepend zero token to K and V (after positional encoding)
        # This implements softmax1 by adding an implicit exp(0)=1 to the denominator
        # and a zero vector that contributes nothing to the output.
        # See: https://www.evanmiller.org/attention-is-off-by-one.html
        zero_k = torch.zeros(
            batch_size, self.num_heads, 1, self.head_dim, device=k.device, dtype=k.dtype
        )
        zero_v = torch.zeros(
            batch_size, self.num_heads, 1, self.head_dim, device=v.device, dtype=v.dtype
        )
        k = torch.cat([zero_k, k], dim=2)  # (B, H, T+1, D)
        v = torch.cat([zero_v, v], dim=2)  # (B, H, T+1, D)
        kv_len = seq_len + 1

        # FlexAttention score modification (no-op for RoPE/HoPE/NoPE, ALiBi
        # supplies its slope-based closure). ghost_offset=1 lets ALiBi
        # leave the kv_idx=0 ghost column unbiased.
        score_mod = self.encoding.build_score_mod(
            self.num_query_heads, inputs.device, ghost_offset=1
        )

        # Determine if we're using GQA
        is_gqa = self.num_queries > 1

        # Check if we should use CPU fallback
        use_fallback = self._use_cpu_fallback(inputs.device)

        if use_fallback:
            if not hasattr(self, "_cpu_fallback_warned"):
                print(
                    "[CausalAttention] Using manual ghost-aware SDPA fallback "
                    "(CPU device - flex_attention not supported)"
                )
                self._cpu_fallback_warned = True
            # Manual masked attention that preserves the ghost column so
            # softmax1/ghostmax behavior matches the flex_attention path.
            attn_output = self._ghost_aware_attention(
                q, k, v, seq_len=seq_len, is_gqa=is_gqa
            )
        else:
            # Use FlexAttention (GPU only)
            # Handle masking: use causal block_mask with ghost token support
            if self.causal:
                # Create causal block_mask that allows ghost access
                block_mask = self._create_causal_mask(seq_len, kv_len, inputs.device)
            else:
                block_mask = None

            # Apply FlexAttention with causal mask and GQA support
            attn_output = self.flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                score_mod=score_mod,
                enable_gqa=is_gqa,
                scale=None,  # Use default: 1.0 / sqrt(head_dim)
            )

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.output(attn_output)
        output = self.dropout(output)

        # Return output, cache (None for now), and aux_loss (0)
        return output, past_key_values, 0.0
