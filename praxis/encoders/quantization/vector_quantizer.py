"""
Vector Quantization modules for discrete information bottlenecks.

Ported from abstractinator with adaptations for praxis:
- Default special token IDs set to -1 (praxis VQ operates on continuous patch embeddings)
- forbidden_ids/protected_ids optional (default None)
- Compile-friendly design (no .item() in forward path)
"""

from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Compile-friendly VQ with EMA and vectorized, guard-stable code resets.

    Key properties:
      - No Python mirrors used in forward() -- only device tensors.
      - No Python branching on GPU scalars (.item()).
      - Resets run with constant shapes (R_MAX), gated by a 0-dim tensor.
      - API: forward(z) -> (z_q_ste, vq_loss, indices, perplexity)
    """

    def __init__(
        self,
        K: int,
        D: int,
        beta: float = 0.25,
        ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        reset_codes: bool = True,
        reset_interval: int = 500,
        max_codes_to_reset_pct: float = 0.5,
        replacement_buffer_size: int = 8192,
        vectors_per_step_to_buffer: int = 1024,
        bos_token_id: int = -1,
        eos_token_id: int = -1,
        padding_token_id: int = -1,
        stale_after: int = 1000,
        forbidden_ids: Optional[List[int]] = None,
        protected_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.K = int(K)
        self.D = int(D)
        self.beta = float(beta)
        self.ema = bool(ema)
        self.decay = float(decay)
        self.eps = float(eps)

        self.reset_codes = bool(reset_codes)
        self.reset_interval = int(reset_interval)
        self.max_codes_to_reset_pct = float(max_codes_to_reset_pct)
        self.replacement_buffer_size = int(replacement_buffer_size)
        self.vectors_per_step_to_buffer = int(vectors_per_step_to_buffer)

        # specials
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.padding_token_id = int(padding_token_id)

        self.register_buffer("forbidden_mask", torch.zeros(self.K, dtype=torch.bool))
        if forbidden_ids:
            self.forbidden_mask[torch.tensor(forbidden_ids, dtype=torch.long)] = True

        self.register_buffer("protected_mask", torch.zeros(self.K, dtype=torch.bool))
        if protected_ids:
            self.protected_mask[torch.tensor(protected_ids, dtype=torch.long)] = True

        # Codebook and EMA stats
        self.codebook = nn.Parameter(torch.randn(self.K, self.D))
        if self.ema:
            self.register_buffer(
                "ema_cluster_size", torch.zeros(self.K, dtype=torch.float32)
            )
            self.register_buffer(
                "ema_weight_sum", self.codebook.data.clone().to(torch.float32)
            )

        # Replacement/reset machinery as DEVICE STATE (no Python mirrors)
        if self.reset_codes:
            # circular buffer of recent vectors
            self.register_buffer(
                "replacement_buffer",
                torch.empty(self.replacement_buffer_size, self.D),
            )
            self.replacement_buffer.zero_()
            # 0-dim device counters/flags
            self.register_buffer("buffer_idx", torch.zeros((), dtype=torch.long))
            self.register_buffer("buffer_is_full", torch.zeros((), dtype=torch.bool))
            self.register_buffer(
                "steps_since_last_reset", torch.zeros((), dtype=torch.long)
            )
            # dead threshold
            self.min_usage_threshold = 1.0
            # constant reset budget (shape-stable)
            self.R_MAX = int(max(0, round(self.K * self.max_codes_to_reset_pct)))
            # global step and per-code last-hit tracker for staleness gating
            self.register_buffer("step", torch.zeros((), dtype=torch.long))
            self.register_buffer("last_hit_step", torch.zeros(self.K, dtype=torch.long))
            self.stale_after = int(stale_after)

    # ---------------------------
    # Internal helpers (tensor-only)
    # ---------------------------

    @torch.no_grad()
    def _update_replacement_buffer_tensor(self, flat_input: torch.Tensor) -> None:
        """Device-only circular buffer update. No Python guards, no graph breaks."""
        N = flat_input.size(0)
        if N == 0:
            return

        # optional subsample to cap copy cost
        if N > self.vectors_per_step_to_buffer:
            idx = torch.randperm(N, device=flat_input.device)[
                : self.vectors_per_step_to_buffer
            ]
            flat_input = flat_input.index_select(0, idx)
            N = flat_input.size(0)

        B = self.replacement_buffer_size  # Python constant
        idx0 = self.buffer_idx  # 0-dim LongTensor on device

        write_idx = (
            idx0 + torch.arange(N, device=flat_input.device, dtype=torch.long)
        ) % B
        self.replacement_buffer.index_copy_(0, write_idx, flat_input)

        # advance head and set 'full' flag if wrapped
        new_idx = (idx0 + N) % B
        self.buffer_idx.copy_(new_idx)

        wrapped = (idx0 + N) >= B  # 0-dim bool
        self.buffer_is_full.logical_or_(wrapped)  # in-place OR

    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, flat_input: torch.Tensor) -> None:
        """EMA updates (tensor-only)."""
        enc = encodings
        # mask specials in EMA stats
        try:
            pad_id = int(self.padding_token_id)
        except Exception:
            pad_id = -1
        if (pad_id is not None) and (pad_id >= 0) and (pad_id < enc.size(1)):
            if enc is encodings:
                enc = enc.clone()
            enc[:, pad_id] = 0

        # cast inputs to float32 for stable EMA math
        enc_f = enc.to(torch.float32)  # (N,K) float32
        flat_f = flat_input.to(torch.float32)  # (N,D) float32

        dw = enc_f.transpose(0, 1) @ flat_f  # (K,D) float32
        cluster = enc_f.sum(0)  # (K,) float32

        self.ema_cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
        self.ema_weight_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        stabilized = (
            (self.ema_cluster_size + self.eps) / (n + self.K * self.eps)
        ) * n
        new_cb_f32 = self.ema_weight_sum / stabilized.unsqueeze(1)  # (K,D) float32
        self.codebook.data.copy_(new_cb_f32.to(self.codebook.dtype))

        # Track recency of usage for reset gating
        hits = cluster > 0
        if hits.any():
            idx = hits.nonzero(as_tuple=False).squeeze(1)
            self.last_hit_step.index_copy_(0, idx, self.step.expand(idx.size(0)))

    @torch.no_grad()
    def _maybe_vectorized_reset(self) -> None:
        """
        Cheap, guard-stable resets:
          - runs every step (no Python guards), but O(R_MAX) work only
          - no Python .item() / .any() on device tensors
          - constant shapes: candidates = R_MAX
        """
        if not self.reset_codes or self.R_MAX <= 0:
            return

        # step counter & gate
        self.steps_since_last_reset.add_(1)
        do_reset = (
            self.steps_since_last_reset >= self.reset_interval
        )  # 0-dim bool on device

        # Replacement buffer "valid count"
        B = self.replacement_buffer_size
        valid_count = torch.where(
            self.buffer_is_full,
            torch.tensor(B, device=self.buffer_idx.device),
            self.buffer_idx,
        )  # 0-dim long
        has_source = valid_count > 0  # 0-dim bool

        # Sample R_MAX candidate code rows (constant shape)
        rand_idx = torch.randint(
            0, self.K, (self.R_MAX,), device=self.codebook.device
        )

        # Which of those candidates are dead? (exclude specials) and stale
        if self.ema:
            usage = self.ema_cluster_size
        else:
            usage = torch.ones(self.K, device=self.codebook.device)

        dead_mask = usage < self.min_usage_threshold
        dead_mask = dead_mask.clone()
        if self.protected_mask.any():
            dead_mask[self.protected_mask] = False

        try:
            pad_id = int(self.padding_token_id)
        except Exception:
            pad_id = -1
        if (pad_id is not None) and (pad_id >= 0) and (pad_id < self.K):
            dead_mask[pad_id] = False

        # Staleness gating: only reset codes not used for a while
        stale_mask = torch.ones_like(dead_mask, dtype=torch.bool)
        if hasattr(self, "last_hit_step"):
            stale_mask = (self.step - self.last_hit_step) >= int(
                getattr(self, "stale_after", 0)
            )
        dead_mask &= stale_mask

        # Apply mask for sampled rows only; also gate by do_reset & has_source
        apply_rows = dead_mask[rand_idx]
        apply_rows &= do_reset
        apply_rows &= has_source

        # Sample replacement vectors (constant shape)
        if self.R_MAX > 0:
            rbuf_idx = torch.randint(
                0, B, (self.R_MAX,), device=self.replacement_buffer.device
            )
            valid_count_safe = torch.clamp(valid_count, min=1)
            rbuf_idx = rbuf_idx % valid_count_safe
            repl = self.replacement_buffer.index_select(0, rbuf_idx)  # (R_MAX, D)
        else:
            repl = torch.empty(0, self.D, device=self.codebook.device)

        # Read current candidate rows and build their new values
        cand_old = self.codebook.index_select(0, rand_idx)  # (R_MAX, D)
        cand_new = torch.where(apply_rows.unsqueeze(1), repl, cand_old)

        # Write back candidate rows only (constant-length index_copy_)
        self.codebook.index_copy_(0, rand_idx, cand_new)

        if self.ema:
            old_cnt = self.ema_cluster_size.index_select(0, rand_idx)
            new_cnt = torch.where(apply_rows, torch.ones_like(old_cnt), old_cnt)
            self.ema_cluster_size.index_copy_(0, rand_idx, new_cnt)

            old_sum = self.ema_weight_sum.index_select(0, rand_idx)
            new_sum = torch.where(apply_rows.unsqueeze(1), cand_new, old_sum)
            self.ema_weight_sum.index_copy_(0, rand_idx, new_sum)

        # Console log: how many codes were reset vs eligible (among sampled)
        resets = int(apply_rows.to(torch.int64).sum().item())
        if resets > 0:
            eligible = int(dead_mask[rand_idx].to(torch.int64).sum().item())
            logging.getLogger(__name__).info(
                "VQ reset: %d codes reset out of %d eligible (R_MAX=%d, K=%d)",
                resets,
                eligible,
                int(self.R_MAX),
                int(self.K),
            )

        # Zero the step counter when reset actually fired
        keep = (~do_reset).to(self.steps_since_last_reset.dtype)
        self.steps_since_last_reset.mul_(keep)

    # ---------------------------
    # Forward
    # ---------------------------

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, Q, D)
        returns:
            z_q_ste: (B, Q, D) straight-through quantized vectors
            vq_loss: scalar tensor
            indices: (B, Q) argmin ids
            perplexity: scalar tensor
        """
        B, Q, Din = z.shape
        assert Din == self.D, f"Expected D={self.D}, got {Din}"

        flat = z.reshape(-1, self.D)

        # Update replacement buffer (tensor-only)
        if self.training and self.reset_codes:
            self._update_replacement_buffer_tensor(flat.detach())

        # Nearest neighbors (squared L2)
        x2 = (flat * flat).sum(dim=1, keepdim=True)  # (N,1)
        e2 = (self.codebook * self.codebook).sum(dim=1)  # (K,)
        xe = F.linear(flat, self.codebook)  # (N,K)
        distances = x2 - 2 * xe + e2.unsqueeze(0)  # (N,K)
        if self.forbidden_mask.any():
            big = torch.finfo(distances.dtype).max
            distances = distances.masked_fill(self.forbidden_mask.unsqueeze(0), big)
        indices = distances.argmin(dim=1)

        z_q = F.embedding(indices, self.codebook).view(B, Q, self.D)

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + (self.beta * commitment_loss)

        # Straight-through
        z_q_ste = z + (z_q - z).detach()

        # EMA + vectorized reset (no Python mirrors)
        if self.training:
            if self.ema:
                onehot = F.one_hot(indices, self.K).type_as(flat)  # (N,K)
                self._ema_update(onehot, flat)
            if self.reset_codes:
                self.step.add_(1)
                self._maybe_vectorized_reset()

        # Perplexity (usage)
        if self.ema:
            counts = self.ema_cluster_size.clone()
        else:
            counts = torch.bincount(indices, minlength=self.K).to(flat.dtype)

        try:
            pad_id = int(self.padding_token_id)
        except Exception:
            pad_id = -1
        if (pad_id is not None) and (pad_id >= 0) and (pad_id < counts.size(0)):
            counts[pad_id] = 0
        if self.forbidden_mask.any():
            counts = counts.masked_fill(self.forbidden_mask, 0)

        counts = counts.clamp(min=self.eps)
        probs = counts / (counts.sum() + self.eps)
        entropy = -(probs * probs.log()).sum()
        perplexity = entropy.exp().to(z.dtype).detach()

        return z_q_ste, vq_loss, indices.view(B, Q), perplexity


class MultiStageResidualVQ(nn.Module):
    """
    Residual VQ with N stages that operate directly in model space D.

    - Each stage quantizes the running residual; outputs are summed in D.
    - Exposes a composed integer index per position so downstream components can
      treat it as a single discrete token in a space of size K**depth.
    """

    def __init__(
        self,
        K: int,
        D: int,
        depth: int = 2,
        beta: float = 0.25,
        ema: bool = True,
        decay: float = 0.999,
        eps: float = 1e-5,
        reset_codes: bool = True,
        reset_interval: int = 250,
        max_codes_to_reset_pct: float = 0.5,
        replacement_buffer_size: int = 8192,
        vectors_per_step_to_buffer: int = 1024,
        bos_token_id: int = -1,
        eos_token_id: int = -1,
        padding_token_id: int = -1,
    ) -> None:
        super().__init__()

        self.D = int(D)
        self.depth = int(depth)
        self.K = int(K)
        self.K_eff = int(self.K**self.depth)

        # Special token ids
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.padding_token_id = int(padding_token_id)

        # expose a codec so downstream code can (de)compose indices
        self.codec = ComposedIndexCodec(
            K=self.K,
            depth=self.depth,
            bos=self.bos_token_id,
            eos=self.eos_token_id,
            pad=self.padding_token_id,
        )

        vq_kwargs = dict(
            beta=beta,
            ema=ema,
            decay=decay,
            eps=eps,
            reset_codes=reset_codes,
            reset_interval=reset_interval,
            max_codes_to_reset_pct=max_codes_to_reset_pct,
            replacement_buffer_size=replacement_buffer_size,
            vectors_per_step_to_buffer=vectors_per_step_to_buffer,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            padding_token_id=padding_token_id,
        )
        self.stages = nn.ModuleList(
            [VectorQuantizer(K=self.K, D=self.D, **vq_kwargs) for _ in range(self.depth)]
        )

        # Sentinel pad vector in the D-space
        self.register_buffer("pad_vector", torch.zeros(D))

    def _compose_indices(self, idx_list: list[torch.Tensor]) -> torch.Tensor:
        base = 1
        composed = torch.zeros_like(idx_list[0])
        for i, idx in enumerate(idx_list):
            if i > 0:
                base = base * self.K
            composed = composed + idx * base
        return composed

    def forward(self, z: torch.Tensor):
        B, Q, _ = z.shape

        # Detect pad sentinels
        pad_vec = self.pad_vector.view(1, 1, -1)
        is_pad = (z == pad_vec).all(dim=-1)  # (B, Q)

        total_loss = torch.tensor(0.0, device=z.device)
        idx_list: list[torch.Tensor] = []
        ppl_list: list[torch.Tensor] = []
        q_sum = torch.zeros_like(z)
        r = z
        for stage in self.stages:
            q_i, loss_i, idx_i, ppl_i = stage(r)
            q_sum = q_sum + q_i
            total_loss = total_loss + loss_i
            idx_list.append(idx_i.view(-1))
            ppl_list.append(ppl_i)
            r = r - q_i.detach()

        z_hat = q_sum
        idx_comp = self._compose_indices(idx_list).view(B, Q)

        if is_pad.any():
            idx_comp = idx_comp.masked_fill(is_pad, self.padding_token_id)
            z_hat = torch.where(
                is_pad.unsqueeze(-1), self.pad_vector.view(1, 1, -1), z_hat
            )

        perplexity = torch.ones((), device=z.device)
        for p in ppl_list:
            perplexity = perplexity * p

        return z_hat, total_loss, idx_comp, perplexity.detach()

    # convenience accessors
    def stage_codebook(self, s: int) -> torch.Tensor:
        return self.stages[s].codebook  # (K, D)

    @property
    def stage_codebooks(self) -> list[torch.Tensor]:
        return [st.codebook for st in self.stages]


class ComposedIndexCodec:
    def __init__(self, K: int, depth: int, bos: int, eos: int, pad: int):
        self.K = int(K)
        self.depth = int(depth)
        self.special = {int(bos), int(eos), int(pad)}
        self.special_list = sorted(list(self.special))
        self.special_to_local = {sid: i for i, sid in enumerate(self.special_list)}

    def is_special(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(x, dtype=torch.bool)
        for sid in self.special:
            mask |= x == sid
        return mask

    @dynamo.disable()
    def decompose(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: (B, L) nonnegative composed ids
        returns:
          digits: list length=depth of (B, L) in [0..K-1]
          special_mask: (B, L) bool
        """
        x = x.long()
        device = x.device
        base = self.K ** torch.arange(
            0, self.depth, device=device, dtype=torch.long
        )  # (D,)
        q = torch.div(
            x.unsqueeze(-1), base, rounding_mode="trunc"
        )  # (B,L,D) long
        digits = torch.remainder(q, self.K)  # (B,L,D) long
        out = [digits[..., d] for d in range(self.depth)]  # list of (B,L) long
        special = self.is_special(x)
        return out, special

    @dynamo.disable()
    def compose(self, digits: List[torch.Tensor]) -> torch.Tensor:
        """
        digits: list length=depth of (B, L) in [0..K-1], long/int
        returns composed: (B, L) long
        """
        assert len(digits) == self.depth
        device = digits[0].device
        D = self.depth
        stacked = torch.stack([d.long() for d in digits], dim=-1)  # (B,L,D) long
        base = self.K ** torch.arange(
            0, D, device=device, dtype=torch.long
        )  # (D,) long
        composed = (stacked * base).sum(dim=-1)  # (B,L) long
        return composed.long()

    @dynamo.disable()
    def special_local_index(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        idx = torch.empty_like(x, dtype=torch.long)
        for sid, j in self.special_to_local.items():
            idx[x == sid] = j
        return idx


def embed_rvq_indices(
    vq: "MultiStageResidualVQ", idx: torch.Tensor
) -> torch.Tensor:
    """Embed composed RVQ indices by summing stage code vectors in model space D."""

    idx = idx.long()
    B, L = idx.shape
    device = idx.device
    dtype = vq.stage_codebook(0).dtype

    digits, _ = vq.codec.decompose(idx)
    emb = torch.zeros(B, L, vq.D, device=device, dtype=dtype)
    for s, digit in enumerate(digits):
        emb = emb + F.embedding(digit, vq.stage_codebook(s))

    pad_id = getattr(vq, "padding_token_id", None)
    if pad_id is not None and pad_id >= 0:
        pad_mask = idx == pad_id
        if pad_mask.any():
            pad_vec = vq.pad_vector.to(device=device, dtype=emb.dtype)
            emb = torch.where(pad_mask.unsqueeze(-1), pad_vec.view(1, 1, -1), emb)

    return emb


def rvq_stage_logits(
    vq: "MultiStageResidualVQ",
    h: torch.Tensor,
    *,
    residual_conditioning: bool = True,
    use_sqdist_logits: bool = False,
    teacher_digits: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    logits, _ = _rvq_stage_logits_internal(
        vq,
        h,
        residual_conditioning=residual_conditioning,
        use_sqdist_logits=use_sqdist_logits,
        teacher_digits=teacher_digits,
        return_greedy=False,
    )
    return logits


def rvq_stage_logits_and_greedy(
    vq: "MultiStageResidualVQ",
    h: torch.Tensor,
    *,
    residual_conditioning: bool = True,
    use_sqdist_logits: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    logits, greedy = _rvq_stage_logits_internal(
        vq,
        h,
        residual_conditioning=residual_conditioning,
        use_sqdist_logits=use_sqdist_logits,
        teacher_digits=None,
        return_greedy=True,
    )
    return logits, greedy


def _rvq_stage_logits_internal(
    vq: "MultiStageResidualVQ",
    h: torch.Tensor,
    *,
    residual_conditioning: bool,
    use_sqdist_logits: bool,
    teacher_digits: Optional[List[torch.Tensor]],
    return_greedy: bool,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Internal helper optionally returning greedy digits when teacher targets absent."""

    r = h
    logits: List[torch.Tensor] = []
    greedy_digits: List[torch.Tensor] = []
    for s in range(vq.depth):
        codebook = vq.stage_codebook(s)
        if use_sqdist_logits:
            r2 = (r * r).sum(-1, keepdim=True)
            e2 = (codebook * codebook).sum(-1).view(1, 1, -1)
            dot = F.linear(r, codebook)
            logits_s = -(r2 - 2 * dot + e2)
        else:
            logits_s = F.linear(r, codebook)
        logits.append(logits_s)

        if residual_conditioning:
            if teacher_digits is not None:
                digits_s = teacher_digits[s]
            else:
                digits_s = logits_s.argmax(dim=-1)
                if return_greedy:
                    greedy_digits.append(digits_s)
            e = F.embedding(digits_s, codebook)
            r = r - e.detach()

    if return_greedy:
        return logits, greedy_digits
    return logits, None
