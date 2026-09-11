"""SMEAR at the granularity the paper actually uses.

Soft Merging of Experts with Adaptive Routing (arXiv:2306.03745), applied to
per-module targets rather than to whole decoder blocks:

                     paper                        here
  merged unit        adapters after attention,    per-module targets found by
                     FFN and cross-attention      walking the module tree
  coefficients       one router per adapter       one row per target
  routing            per example                  per example, causally (Linear
                                                  targets; see CAUSALITY)
  balancing          expert dropout               same, 0.1

Two things here are not in the paper, and neither is novel:

  * BASE PLUS DEVIATIONS instead of N independent expert copies. With
    ``P_e = base + delta_e`` and coefficients summing to one,
    ``base + sum_e w_e delta_e == sum_e w_e P_e``, so this is the paper's merge
    in a different basis. It buys exact identity at initialization and a shared
    trunk that receives full gradient whatever the routing does
    (``d(merged)/d(base) = sum_e w_e = 1``), so a starved deviation costs its
    rank rather than a whole block. Large deviations are additionally
    rank-constrained, which is LoRA.
  * PEFT-style target discovery (praxis/routers/targeting.py), so merge sites are
    found rather than hand-placed.

CAUSALITY. The paper pools the example's hidden states, and in its decoder it
pools the ENCODER's final states "to prevent information leakage from later
target tokens". A decoder-only trunk has no such fully visible source, so a
sequence mean here would hand every position's merged weights the future.
Position ``t`` therefore routes on the running mean of positions ``0..t``: the
paper's pooled input at the last position, and nothing later anywhere. Linear
targets take those per-position coefficients through ``MergedLinear`` at no
extra cost. Elementwise and indexed targets (norms, residual gates, the
per-depth table) can hold only one geometry per forward, and a geometry shared
by every position cannot read any of them, so they merge on the input-free
depth prior - the paper never routes layernorm parameters either. Lory
(arXiv:2405.03133) meets the same constraint a segment at a time: segment k
routes on the mean of segment k-1, and the first segment on itself behind a
stop-gradient, which still reads that segment's later tokens in the forward.
The running mean here is the one-token limit of that rule with no exception.

Sharpening is OFF by default: ``p**4`` drives losing deviations to zero
gradient, which is the dead-expert mechanism. ``VEAR`` re-enables it for the
comparison.

Companion: praxis/routers/targeting.py, praxis/routers/vear.py.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis import registry
from praxis.transforms.targeting import (
    DENSE_DELTA_MAX_NUMEL,
    TargetGroup,
    describe,
    discover_targets,
)

# Merges between refreshes of the routing diagnostics. Matches SMEAR's cadence
# and DynamicsLoggerCallback's log_freq; the diagnostics cost host-device syncs
# and the merge runs once per recurrent pass.
METRICS_INTERVAL: int = 10

# How far the routing decision is shared before the merge.
#
#   "token"   - every position routes on its own state. Beyond the paper, and
#               possible only on the Linear targets, where associativity means
#               the merged weight is never materialized (see MergedLinear).
#   "example" - every position routes on the running mean of its prefix: the
#               paper's per-example pooling, made causal. Default.
#   "batch"   - one input-free geometry (the depth prior) for the whole batch:
#               the control arm for input-dependent routing.
#
# Elementwise and indexed targets (norms, residual gates, the per-depth table)
# take the depth prior under every reduction: one geometry per forward is all
# their forwards can hold, and it cannot read any position without reading the
# future of the positions before it.
REDUCTIONS = ("token", "example", "batch")

# Rank of a factored deviation, as a divisor of the smaller weight dimension.
# Config-derived rather than a flag, the same way PEER sizes its bank against
# the dense FFN it replaces (praxis/dense/peer.py).
RANK_DIVISOR: int = 8
MIN_RANK: int = 4

# Probability of dropping an entire deviation from a routing decision. This is
# SMEAR's OWN load-balancing mechanism (the paper uses expert dropout, not an
# auxiliary balance loss or a DeepSeek-style bias), at the paper's rate. Without
# it one deviation per target monopolizes its coefficient.
#
# Safe here in a way it is not under the paper's merge. When every expert is
# dropped, the renormalized coefficients are all zero, and a base-plus-deviation
# merge then falls back to ``base`` EXACTLY. SMEAR's ``sum_e w_e P_e`` under the
# same draw yields an all-zero parameter block.
MODULAR_EXPERT_DROPOUT: float = 0.1


def _set_submodule(root: nn.Module, dotted: str, replacement: nn.Module) -> None:
    """Rebind ``root.<dotted>`` to ``replacement``, tolerating list indices."""
    parts = dotted.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = replacement
    else:
        setattr(parent, last, replacement)


class MergedLinear(nn.Module):
    """A routed ``nn.Linear``: base weight plus N low-rank deviations, merged
    per example or per position.

    Associativity is what makes that affordable. Merging first and applying second needs one
    weight tensor per distinct coefficient vector, so per-example routing would
    mean a whole ``[B, out, in]`` stack and per-position routing a ``[B, T, out,
    in]`` one. Applying first and merging second does not::

        y_b = (W + sum_e c_be B_e A_e) x_b
            = W x_b + sum_e c_be * B_e (A_e x_b)

    The right-hand form never materializes a merged weight. Cost per token is
    ``N * r * (in + out)`` against the base's ``in * out``; at ``r = min/8`` and
    ``N = 4`` that is roughly one extra base projection, and the only new
    activation is ``[B, T, N, r]``.

    Per-example coefficients are what let the routing learn: under one shared
    merge every example receives the same routing gradient, and a constant
    router is the fixed point.

    The base ``weight`` and ``bias`` are the ORIGINAL Parameter objects, held
    directly rather than behind a nested Linear, so qualified names are
    unchanged (``attn.qkv.weight`` stays ``attn.qkv.weight``) and anything
    introspecting ``.weight`` / ``.in_features`` keeps working.
    """

    def __init__(self, base: nn.Linear, num_experts: int, rank: int) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # From the config, like every other sizing decision in this repo. The
        # registry carries one entry per router, not one per expert count.
        self.num_experts = int(
            num_experts
            if num_experts is not None
            else getattr(config, "num_experts", 4)
        )
        self.rank = int(rank)

        self.weight = base.weight
        self.bias = base.bias

        # LoRA's initialization: A random, B zero, so every deviation is exactly
        # zero at step 0 and the wrapper is bit-identical to the Linear it
        # replaced. A config swap stays a clean A/B rather than a reroll.
        self.lora_a = nn.Parameter(
            torch.empty(self.num_experts, rank, self.in_features)
        )
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
        self.lora_b = nn.Parameter(
            torch.zeros(self.num_experts, self.out_features, rank)
        )

        # Set for the duration of one block forward by the router's coefficient
        # scope; None means "run as a plain Linear", which is what inference
        # paths and any caller that bypasses the router get.
        self._coeff: Optional[Tensor] = None
        self._warned = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, rank={self.rank}"
        )

    @property
    def delta_numel(self) -> int:
        return self.lora_a.numel() + self.lora_b.numel()

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        coeff = self._coeff
        # Fall back to the base whenever the coefficients cannot apply: no scope
        # is open, or the batch was reshaped between routing and here (cached
        # decode). Silently misaligning a coefficient with an example would be
        # far worse than routing nothing.
        # Leading axes must line up: [B, N] against [B, ..., in] for
        # per-example routing, [B, S, N] against [B, S, in] for per-token.
        if coeff is None:
            return y
        if tuple(coeff.shape[:-1]) != tuple(x.shape[: coeff.dim() - 1]):
            # Training never reshapes between routing and here, so a mismatch
            # there means this target's input is not laid out [B, T, ...] and
            # its deviations would train on nothing, silently.
            if self.training and not self._warned:
                self._warned = True
                print(
                    f"[SMEAR] a routed Linear received input {tuple(x.shape)} "
                    f"that does not line up with its {tuple(coeff.shape)} "
                    "coefficients; it runs unrouted."
                )
            return y
        # [B, ..., N, r] - each deviation's low-rank projection of the input.
        u = torch.einsum("...i,eri->...er", x, self.lora_a.to(x.dtype))
        # Insert whatever sequence-ish axes the coefficients do not already
        # carry, so per-example coefficients broadcast across positions and
        # per-token ones align with them.
        missing = (x.dim() - 1) - (coeff.dim() - 1)
        shape = coeff.shape[:1] + (1,) * missing + coeff.shape[1:] + (1,)
        u = u * coeff.to(u.dtype).view(shape)
        return y + torch.einsum("...er,eor->...o", u, self.lora_b.to(x.dtype))


def _get_param(module: nn.Module, dotted: str) -> Optional[Tensor]:
    """Fetch a parameter by fully-qualified name (``attn.qkv.weight``)."""
    parts = dotted.split(".")
    obj: Any = module
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return getattr(obj, parts[-1], None)


class SMEAR(nn.Module):
    """Soft-merging of experts, at the granularity the paper uses."""

    # How the decoder should build the layer stack for this router. "shared"
    # means ONE block, reused at every layer position - the same topology the
    # SMEAR bank produced, minus the replication. Read by
    # praxis/decoders/base.py::_router_layout.
    LAYER_LAYOUT: str = "shared"

    # Exponent applied to the coefficients before merging. 1.0 is a no-op; see
    # the module docstring for why this is not VEAR's 4.0.
    SHARPEN: float = 1.0

    # Whole-deviation dropout, SMEAR's own balancing mechanism. See
    # MODULAR_EXPERT_DROPOUT.
    EXPERT_DROPOUT: float = MODULAR_EXPERT_DROPOUT

    def __init__(
        self,
        config: Any,
        block: Optional[nn.Module] = None,
        num_experts: Optional[int] = None,
        target_profile: str = "all",
        reduction: str = "example",
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if reduction not in REDUCTIONS:
            raise ValueError(
                f"Unknown reduction {reduction!r}; known: {sorted(REDUCTIONS)}"
            )
        self.reduction = reduction
        if block is None:
            raise ValueError(
                "SMEAR routers merge deviations against a concrete block; pass "
                "block=... at construction (see praxis/decoders/base.py)."
            )
        if target_profile not in registry.namespace("target_profiles"):
            raise ValueError(
                f"Unknown target profile {target_profile!r}; "
                f"known: {sorted(registry.namespace("target_profiles"))}"
            )

        # From the config, like every other sizing decision in this repo. The
        # registry carries one entry per router, not one per expert count.
        self.num_experts = int(
            num_experts
            if num_experts is not None
            else getattr(config, "num_experts", 4)
        )
        self.hidden_size = config.hidden_size
        self.profile_name = target_profile

        spec = registry.lookup("target_profiles", target_profile)
        self.targets, skipped = discover_targets(block, spec)
        if not self.targets:
            raise ValueError(
                f"Target profile {target_profile!r} matched nothing in "
                f"{type(block).__name__}. Skips: {skipped}"
            )
        self._skipped = skipped

        # Targets split by HOW they are merged, not by what they are:
        #
        #   routed - the target is an nn.Linear, so associativity lets each
        #     position carry its own coefficients at low-rank cost (MergedLinear).
        #     These are the adapter-shaped modules, which is exactly what the
        #     SMEAR paper routes: adapters inserted after self-attention, the
        #     feed-forward and cross-attention.
        #   depth prior - norms, residual gates, kappa/mu, the per-depth table.
        #     Elementwise or indexed rather than matmuls, so one geometry per
        #     forward is all they hold, and the paper does not treat them as
        #     experts either (it trains layernorm parameters but never routes
        #     them).
        self.wrappers = nn.ModuleDict()  # metric label -> MergedLinear
        self._wrapper_row: Dict[str, int] = {}  # metric label -> router row
        self.deltas = nn.ParameterDict()
        self._factored: Dict[str, bool] = {}
        self._param_row: Dict[str, int] = {}
        merged_numel = 0

        for row, group in enumerate(self.targets):
            module = block.get_submodule(group.name) if group.name else block
            merged_numel += sum(_get_param(block, p).numel() for p in group.params)
            if isinstance(module, nn.Linear):
                # Rank is forced here regardless of size: a DENSE per-example
                # deviation would cost N * in * out per token (four base
                # projections at N=4) where the factored form costs about one,
                # and a small matrix's rank-r bank is cheap anyway.
                rank = max(
                    MIN_RANK,
                    min(module.in_features, module.out_features) // RANK_DIVISOR,
                )
                wrapper = MergedLinear(module, self.num_experts, rank)
                _set_submodule(block, group.name, wrapper)
                self.wrappers[group.label] = wrapper
                self._wrapper_row[group.label] = row
                continue
            for pname in group.params:
                self._param_row[pname] = row
                self._build_delta(pname, _get_param(block, pname))

        self.merged_numel = merged_numel
        self.delta_numel = sum(p.numel() for p in self.deltas.values()) + sum(
            w.delta_numel for w in self.wrappers.values()
        )

        # One router head for every (target, expert) pair. LayerNorm on the
        # pooled input, weight-normalized projection: SMEAR's arrangement, only
        # wider in the output.
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, len(self.targets) * self.num_experts)

        # Per-recurrent-pass additive bias on the per-target logits - the
        # ArcAttention idiom (praxis/attention/arc.py), widened to
        # ``targets * experts`` so each pass can move each module
        # independently. Zero-init, so it is exactly absent until it learns
        # otherwise.
        self.depth = getattr(config, "depth", 1) or 1
        self.depth_bias = nn.Embedding(self.depth, len(self.targets) * self.num_experts)
        nn.init.zeros_(self.depth_bias.weight)

        self._metrics: Dict[str, float] = {}
        # On-device running sums over the passes since the last flush, and
        # how many passes went into them. See _log_metrics.
        self._accum: Dict[str, Tensor] = {}
        self._passes: int = 0
        self._tick: int = 0
        # Passes per diagnostic window: one full recurrent loop, so a window
        # spans every depth even though halting varies how many actually run.
        self._window: int = max(1, self.depth)

        if verbose:
            print(describe(self.targets, skipped))
            print(
                f"[SMEAR] {len(self.targets)} targets x {self.num_experts} experts; "
                f"deviations hold {self.delta_numel:,} parameters "
                f"({self.delta_numel / max(1, merged_numel):.2f}x the merged base)"
            )
            print(
                f"[SMEAR] {len(self.wrappers)} target(s) routed causally, "
                f"reduction={self.reduction} "
                f"({', '.join(self.wrappers) or 'none'}); "
                f"{len(self._param_row)} parameter(s) on the depth prior; "
                f"expert dropout {self.EXPERT_DROPOUT}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(targets={len(self.targets)}, "
            f"num_experts={self.num_experts}, profile={self.profile_name})"
        )

    # --- construction ---------------------------------------------------------

    @staticmethod
    def _key(pname: str) -> str:
        return pname.replace(".", "__")

    def _build_delta(self, pname: str, base: Tensor) -> int:
        """Allocate this parameter's deviation bank. Returns the base's numel.

        Large 2-D weights factor to rank r; everything else - biases, norm
        scales, residual gates, embeddings small enough not to care - takes a
        dense deviation. The choice is made from the shape alone, so there is
        nothing to tune and nothing to configure per experiment.
        """
        key = self._key(pname)
        n = self.num_experts
        if base.dim() == 2 and base.numel() > DENSE_DELTA_MAX_NUMEL:
            out_dim, in_dim = base.shape
            rank = max(MIN_RANK, min(out_dim, in_dim) // RANK_DIVISOR)
            # LoRA's initialization: A random, B zero, so the product - and
            # therefore the whole merge - is EXACTLY zero at step 0.
            a = nn.Parameter(torch.empty(n, rank, in_dim))
            nn.init.normal_(a, mean=0.0, std=0.02)
            self.deltas[key + "__a"] = a
            self.deltas[key + "__b"] = nn.Parameter(torch.zeros(n, out_dim, rank))
            self._factored[pname] = True
        else:
            self.deltas[key] = nn.Parameter(torch.zeros(n, *base.shape))
            self._factored[pname] = False
        return base.numel()

    # --- routing --------------------------------------------------------------

    def _route_logits(self, router_input: Tensor, current_depth: int) -> Tensor:
        """Per-target routing logits, shape [batch, targets, experts].

        Weight-normalized like SMEAR's, and the single place routing is decided
        so the depth-aware subclass overrides here and nowhere else.
        """
        normalized = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized, self.router.bias)
        logits = logits.view(
            *router_input.shape[:-1], len(self.targets), self.num_experts
        )
        # The decoder can loop past `depth` when halting samples deeper than the
        # table; wrap rather than raise, so a pass reuses an existing row.
        bias = self.depth_bias(
            torch.tensor(int(current_depth) % self.depth, device=logits.device)
        )
        return logits + bias.view(len(self.targets), self.num_experts)

    def _prior_logits(self, current_depth: int) -> Tensor:
        """Input-free logits ``[targets, experts]``: the router's bias plus the
        per-depth row. The one routing every position may share, because it
        reads none of them."""
        logits = self.router.bias.view(len(self.targets), self.num_experts)
        bias = self.depth_bias(
            torch.tensor(int(current_depth) % self.depth, device=logits.device)
        )
        return logits + bias.view_as(logits)

    def _prefix_mean(self, inputs: Tensor, cache: Any, current_depth: int) -> Tensor:
        """Mean over positions ``0..t`` at every position ``t``, ``[B, T, D]``.

        Accumulated in float32, since a low-precision cumsum drifts over a long
        row. Under cached decode the sum continues from the state carried for
        this depth when it lines up with the depth's cached length, and falls
        back to the suffix alone when it does not (a rollback, a batch change),
        which is the crystal head's rule (``_route_causal``).
        """
        x = inputs.float()
        batch, length = x.shape[0], x.shape[1]
        total = x.cumsum(dim=1)
        count = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
        count = count.view(1, -1, 1).expand(batch, -1, 1)
        if hasattr(cache, "get_head_state") and hasattr(cache, "get_seq_length"):
            depth = int(current_depth)
            key = f"{type(self).__name__}-prefix:{id(self)}:{depth}"
            cached = int(cache.get_seq_length(depth))
            state = cache.get_head_state(key)
            if (
                state is not None
                and state["pos"] == cached
                and state["sum"].shape[0] == batch
            ):
                total = total + state["sum"].to(x.device).unsqueeze(1)
                count = count + state["count"].to(x.device).view(batch, 1, 1)
            cache.set_head_state(
                key,
                {
                    "sum": total[:, -1].detach(),
                    "count": count[:, -1, 0].detach(),
                    "pos": cached + length,
                },
            )
        return (total / count).to(inputs.dtype)

    def _regularize(self, probs: Tensor, draw_shape: Optional[tuple] = None) -> Tensor:
        """Expert dropout, then sharpening: ``probs`` -> merge coefficients.

        Dropout is SMEAR's own load-balancing mechanism; ``draw_shape`` lets one
        draw cover several positions (one per example under "example"). Under
        gradient checkpointing the recomputed pass draws the SAME mask, because
        torch.utils.checkpoint runs with preserve_rng_state=True
        (praxis/decoders/checkpoint.py). An all-dropped row renormalizes to
        zeros rather than NaN, and zero coefficients mean "use the base
        unchanged" - see MODULAR_EXPERT_DROPOUT.
        """
        merge = probs
        if self.training and self.EXPERT_DROPOUT > 0:
            shape = probs.shape if draw_shape is None else draw_shape
            keep = torch.bernoulli(
                torch.full(
                    shape,
                    1.0 - self.EXPERT_DROPOUT,
                    device=probs.device,
                    dtype=probs.dtype,
                )
            )
            merge = merge * keep
            merge = merge / merge.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.SHARPEN != 1.0:
            merge = merge.pow(self.SHARPEN)
            merge = merge / merge.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return merge

    def _coefficients(
        self, inputs: Tensor, current_depth: int, cache: Any = None
    ) -> Tuple[Tensor, Tensor]:
        """Merge coefficients for the Linear targets, and the router's own probs.

        ``[B, T, targets, experts]`` under "example" and "token", one routing
        per position that reads no later position; ``[B, targets, experts]``
        under "batch", the depth prior repeated for every example.
        """
        if self.reduction == "batch":
            probs = F.softmax(self._prior_logits(current_depth), dim=-1)
            merge = self._regularize(probs)
            batch = inputs.shape[0]
            return merge.expand(batch, -1, -1), probs.expand(batch, -1, -1)
        if self.reduction == "token":
            routed, draw_shape = inputs, None
        else:
            routed = self._prefix_mean(inputs, cache, current_depth)
            draw_shape = (inputs.shape[0], 1, len(self.targets), self.num_experts)
        logits = self._route_logits(self.router_norm(routed), current_depth)
        probs = F.softmax(logits, dim=-1)
        return self._regularize(probs, draw_shape), probs

    def _depth_prior(self, current_depth: int) -> Tensor:
        """``[targets, experts]`` coefficients for the elementwise targets."""
        return self._regularize(F.softmax(self._prior_logits(current_depth), dim=-1))

    @staticmethod
    def _flatten(merge: Tensor) -> Tensor:
        """``[..., targets, experts]`` -> ``[targets, experts]``, averaging every
        leading axis. Diagnostics only."""
        return merge.reshape(-1, merge.shape[-2], merge.shape[-1]).mean(dim=0)

    def _applied(self, merge: Tensor, prior: Tensor) -> Tensor:
        """Mean coefficients each target actually ran with, ``[targets, experts]``:
        the routed rows for the Linear targets, the depth prior for the rest."""
        applied = prior.clone()
        routed = self._flatten(merge)
        for row in self._wrapper_row.values():
            applied[row] = routed[row]
        return applied

    @contextmanager
    def _coefficient_scope(self, merge: Tensor) -> Iterator[None]:
        """Hand each Linear target its ``[B, T, N]`` (or ``[B, N]``) coefficients
        for one block forward, then take them back.

        Scoped rather than passed as an argument because the coefficients have
        to reach modules several levels inside the block whose forward
        signatures know nothing about routing. This mirrors the width policy's
        ``scope`` (praxis/width), which live-patches the same block from the
        same place in the decoder loop. Restoring on exit matters: a wrapper
        left holding a stale coefficient would silently route the next forward
        with the previous one's routing.
        """
        try:
            for label, row in self._wrapper_row.items():
                self.wrappers[label]._coeff = merge[..., row, :]
            yield
        finally:
            for label in self._wrapper_row:
                self.wrappers[label]._coeff = None

    # --- merging --------------------------------------------------------------

    def _delta_for(self, pname: str, w: Tensor, dtype: torch.dtype) -> Tensor:
        """``sum_e w_e * delta_e`` for one parameter."""
        key = self._key(pname)
        if self._factored[pname]:
            a = self.deltas[key + "__a"]  # [N, r, in]
            b = self.deltas[key + "__b"]  # [N, out, r]
            scaled = b * w.to(b.dtype).view(-1, 1, 1)
            return torch.einsum("nor,nri->oi", scaled, a).to(dtype)
        bank = self.deltas[key]  # [N, *shape]
        return torch.tensordot(w.to(bank.dtype), bank, dims=([0], [0])).to(dtype)

    def _merged_state_dict(self, layer: nn.Module, w: Tensor) -> Dict[str, Tensor]:
        """Base parameters plus their routed deviations.

        Only targeted names appear. ``functional_call`` leaves every absent name
        bound to the module's own parameter, so untargeted modules - PEER, the
        shared long-term memory, anything lazy - run exactly as they would
        without a router, at no cost.
        """
        merged: Dict[str, Tensor] = {}
        for pname, row in self._param_row.items():
            base = _get_param(layer, pname)
            if base is None:
                raise ValueError(f"Target parameter {pname!r} vanished from the block.")
            merged[pname] = base + self._delta_for(pname, w[row], base.dtype)
        return merged

    # --- forward --------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any):
        """Router-mode forward: seven positional arguments, or eight when an
        encoder supplies a byte timeline (praxis/layers/local.py)."""
        if len(args) not in (7, 8):
            raise NotImplementedError(
                "SMEAR routers support router mode only (7 or 8 positional args "
                f"from LocalLayer); got {len(args)}."
            )
        layer, inputs, attention_mask, past_key_values = args[:4]
        current_state, current_depth, block_ids = args[4:7]
        positions = args[7] if len(args) == 8 else None

        merge, probs = self._coefficients(inputs, current_depth, past_key_values)

        # Elementwise and indexed targets merge on the input-free depth prior;
        # the Linear targets take their per-position rows through the scope.
        prior = self._depth_prior(current_depth)
        merged = self._merged_state_dict(layer, prior)

        # Diagnostics run over a contiguous WINDOW of `depth` passes and then
        # sleep for METRICS_INTERVAL - 1 windows: long enough to span every
        # depth, while touching only one pass in METRICS_INTERVAL (every pass
        # would cost ~10% per pass). Ordered after the merge because
        # _delta_scale reads the merged tensors rather than recomputing them.
        if (self._tick // self._window) % METRICS_INTERVAL == 0:
            with torch.no_grad():
                applied = self._applied(merge, prior)
                self._log_metrics(applied, probs)
                self._delta_scale(layer, merged, applied)
        self._tick += 1
        # Flush on a COMPLETE window, so the reported average always covers a
        # whole recurrent loop rather than whichever passes happened to land
        # either side of a tick boundary.
        if self._passes >= self._window:
            with torch.no_grad():
                self._flush_metrics()

        forward_args = (
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        )
        # By KEYWORD: the block's 7th positional slot is router_weights, so
        # passing positions there would feed it to the FFN gate.
        forward_kwargs = {} if positions is None else {"positions": positions}

        # tie_weights=False: the same module is reparametrized once per recurrent pass, and functional_call's
        # parameter-aliasing machinery corrupts the merged graph across those
        # reuses, surfacing as a double-backward on a freed graph.
        with self._coefficient_scope(merge):
            result = torch.func.functional_call(
                layer, merged, forward_args, forward_kwargs, tie_weights=False
            )

        if isinstance(result, tuple) and len(result) == 4:
            return result
        if isinstance(result, tuple) and len(result) == 3:
            return result[0], result[1], result[2], 0.0
        return result, past_key_values, current_state, 0.0

    # --- auxiliary loss -------------------------------------------------------

    # --- diagnostics ----------------------------------------------------------

    @staticmethod
    def _entropy(probs: Tensor, dim: int = -1) -> Tensor:
        """Shannon entropy in nats. ``clamp_min``, never ``+ eps``: an epsilon
        pushes a weight of exactly 1.0 above 1 and makes the entropy negative."""
        p = probs.clamp_min(1e-12)
        return (-(p * p.log()).sum(dim=dim)).clamp_min(0.0)

    def _log_metrics(self, merge: Tensor, probs: Tensor) -> None:
        """Four scalars and one heatmap, with no depth prefix: one router serves
        every depth, and ``router_depth_specialization`` reports how the passes
        differ.

        Averaged over a contiguous WINDOW of passes rather than sampled from one:
        ``_tick`` advances once per recurrent pass, so sampling every Nth tick
        would report a fixed subset of depths. Accumulation stays on-device and
        only the flush calls ``.item()``, one host-device sync per window.
        """
        try:
            self._reset_accum_if_moved(merge)
            n = self.num_experts
            w = self._flatten(merge)  # [T, N] - what the heatmap reports either way
            # The heatmap: every target's merge weights, one card. Kept as a
            # single [T, N] tensor so accumulating it costs no host traffic;
            # the per-cell names are attached at flush.
            self._add("coeff", w)
            self._passes += 1

            # Are the deviations being USED, or has each target picked one and
            # abandoned the rest? Fraction of (target, expert) pairs carrying
            # more than half their fair share, averaged over targets: 1.0 at
            # perfect balance, 1/N at total collapse. This is the number expert
            # dropout is there to hold up, so it is the direct read on whether
            # the paper's balancing mechanism is working.
            self._add(
                "smear_expert_utilization",
                (w > (0.5 / n)).float().sum(dim=-1).mean() / n,
            )

            # Only the Linear targets route on the input; the rest run on the
            # depth prior, so their rows of `probs` describe nothing applied.
            routed_rows = sorted(self._wrapper_row.values())
            if n > 1 and routed_rows:
                probs = probs[..., routed_rows, :]
                # Does the router read its input? I(input; expert) per target,
                # normalized to [0, 1], then averaged. Zero means every sequence
                # in the batch got the same coefficients, i.e. the router is a
                # constant and the whole mechanism is a reparametrized base.
                flat = probs.reshape(-1, probs.shape[-2], probs.shape[-1])
                mean_p = flat.mean(dim=0)  # [T, N]
                h_mean = self._entropy(mean_p, dim=-1)  # [T]
                h_seq = self._entropy(flat, dim=-1).mean(dim=0)  # [T]
                mi = ((h_mean - h_seq) / math.log(n)).clamp(0.0, 1.0)
                self._add("smear_input_dependence", mi.mean())
                self._add("smear_input_dependence_max", mi.max())

                # Utilization above is computed on the batch-MEAN coefficients,
                # so it cannot tell "every example picked deviation 2" from
                # "each example picked a different one" - the first is collapse,
                # the second is specialization, and VEAR is built to produce the
                # second. These two separate them:
                #
                #   sharpness high + diversity high  specialization (VEAR's aim)
                #   sharpness high + diversity low   collapse to one deviation
                #   sharpness low  + diversity low   a soft blend, near-constant
                #
                # Measured on `probs`, i.e. the router's own opinion re-sharpened
                # but WITHOUT expert dropout: dropout zeroes a tenth of the
                # coefficients and renormalizes, which reads as peakedness the
                # router did not choose and would bias smear against vear.
                sel = probs
                if self.SHARPEN != 1.0:
                    sel = sel.pow(self.SHARPEN)
                    sel = sel / sel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                sel = sel.reshape(-1, sel.shape[-2], sel.shape[-1])  # [D, T, N]

                # How peaked ONE routing decision is. 0 = uniform blend over the
                # deviations, 1 = a single deviation chosen outright.
                self._add(
                    "smear_selection_sharpness",
                    (1.0 - self._entropy(sel, dim=-1).mean() / math.log(n)).clamp(
                        0.0, 1.0
                    ),
                )

                # Do different decisions land on different deviations? Entropy of
                # the argmax distribution, normalized. 0 = every example picks the
                # SAME deviation (so a sharp router is just a collapsed one);
                # 1 = the selections spread evenly across the bank. Exactly 0 by
                # construction under reduction="batch".
                picks = F.one_hot(sel.argmax(dim=-1), n).to(sel.dtype)  # [D, T, N]
                self._add(
                    "smear_selection_diversity",
                    (self._entropy(picks.mean(dim=0), dim=-1) / math.log(n)).mean(),
                )

            # THE number this design exists to produce. Mean pairwise L1
            # distance between different targets' coefficient rows, over its
            # maximum of 2. Zero means every module chose the same mixture, so
            # per-module granularity bought nothing and the block is back to one
            # scalar; high means the modules genuinely disagree, which is the
            # only thing that justifies the extra rows.
            if n > 1 and len(self.targets) > 1:
                d = (w.unsqueeze(0) - w.unsqueeze(1)).abs().sum(-1)  # [T, T]
                t_count = len(self.targets)
                self._add(
                    "smear_target_dispersion",
                    d.sum() / (t_count * (t_count - 1) * 2.0),
                )
        except Exception:
            # Diagnostics never break a step; they write to a plain dict and
            # never reach the loss.
            pass

    def _delta_scale(
        self, layer: nn.Module, merged: Dict[str, Tensor], w: Tensor
    ) -> None:
        """``||sum_e w_e delta_e|| / ||base||`` per target, Frobenius.

        The coefficients say how the deviations are MIXED; this says whether the
        mixture moves the geometry at all. Both are needed to read the design:
        a rich mixture over deviations that are numerically tiny is a router
        arguing about nothing, and is indistinguishable from a real one on the
        coefficient heatmap alone.

        Reported for the mean applied coefficients, matching the heatmap, even
        though the Linear targets apply their own row per position. Delta over BASE rather
        than over merged, so the number is read directly as "the routing moved
        this module's weights by N% of their own norm".
        """
        try:
            self._reset_accum_if_moved(w)
            self._delta_scale_inner(layer, merged, w)
        except Exception:
            # Optional telemetry must never reach the step.
            pass

    def _delta_scale_inner(
        self, layer: nn.Module, merged: Dict[str, Tensor], w: Tensor
    ) -> None:
        rows: List[Optional[Tensor]] = [None] * len(self.targets)
        parts: Dict[int, List[Tensor]] = {}

        # Elementwise / indexed targets: functional_call already materialized
        # base + delta, so the deviation is one subtraction away.
        for pname, row in self._param_row.items():
            base = _get_param(layer, pname)
            if base is None:
                return
            parts.setdefault(row, []).append(
                (merged[pname] - base).norm() / base.norm().clamp_min(1e-12)
            )

        # Linear targets never materialize a merged weight (that is the whole
        # point of MergedLinear), so the low-rank product is formed here and
        # here only. These are small: [out, in] at hidden_size scale.
        for label, row in self._wrapper_row.items():
            wrapper = self.wrappers[label]
            delta = torch.einsum(
                "e,eor,eri->oi",
                w[row].to(wrapper.lora_b.dtype),
                wrapper.lora_b,
                wrapper.lora_a,
            )
            parts.setdefault(row, []).append(
                delta.norm() / wrapper.weight.norm().clamp_min(1e-12)
            )

        for row, values in parts.items():
            rows[row] = torch.stack(values).mean()
        if any(r is None for r in rows):
            return
        self._add("delta_scale", torch.stack(rows))

    def _reset_accum_if_moved(self, ref: Tensor) -> None:
        """Drop a partial window left on another device.

        ``initialize_lazy_modules`` runs a dummy forward on CPU before the model
        is moved to its device (praxis/utils/system.py), so the accumulator can
        be holding CPU tensors that the next pass cannot be added to. That is a
        hard RuntimeError in the middle of forward(), not a wrong number. A
        partial window is worth nothing; not crashing is worth everything.
        """
        for value in self._accum.values():
            if value.device != ref.device:
                self._accum, self._passes = {}, 0
            break

    def _add(self, key: str, value: Tensor) -> None:
        """Sum one pass's diagnostic into the on-device accumulator.

        Detached, so nothing here can hold a graph alive across passes.
        """
        prev = self._accum.get(key)
        value = value.detach()
        self._accum[key] = value if prev is None else prev + value

    def _flush_metrics(self) -> None:
        """Average the accumulated passes into the float dict the logger drains.
        The only place this class touches the host."""
        if not self._accum or self._passes == 0:
            return
        try:
            passes = float(self._passes)
            coeff = self._accum.pop("coeff", None)
            if coeff is not None:
                rows = (coeff / passes).tolist()
                for t, group in enumerate(self.targets):
                    for e, value in enumerate(rows[t]):
                        self._metrics[f"smear_coeff_{group.label}_{e}"] = value
            scale = self._accum.pop("delta_scale", None)
            if scale is not None:
                scale = scale / passes
                for t, group in enumerate(self.targets):
                    self._metrics[f"smear_delta_scale_{group.label}"] = scale[t].item()
                self._metrics["smear_delta_scale_mean"] = scale.mean().item()
            for key, total in self._accum.items():
                self._metrics[key] = (total / passes).item()
        except Exception:
            pass
        finally:
            self._accum = {}
            self._passes = 0

    def get_metrics(self) -> dict:
        from praxis.metrics.specialization import depth_dispersion

        # A window normally closes long before the first poll. If it has not -
        # a run shorter than one recurrent loop, or a test doing a single
        # forward - report the partial average rather than nothing at all.
        if self._passes and not self._metrics:
            self._flush_metrics()
        out = self._metrics.copy()
        stats = depth_dispersion(self.depth_bias.weight)
        if stats is not None:
            out["router_depth_specialization"] = stats["specialization"]
            out["router_depth_similarity"] = stats["similarity"]
        return out
