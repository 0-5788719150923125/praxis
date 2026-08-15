"""SMEAR at the granularity the paper actually uses.

This is NOT a new method. It is Soft Merging of Experts with Adaptive Routing
(https://arxiv.org/abs/2306.03745) applied the way the paper applies it, because
``praxis/routers/smear.py`` does not. That file is the same algorithm pointed at
the wrong object, and every difference here is a step back TOWARD the paper
rather than past it:

                     paper                    smear.py / vear.py      here
  merged unit        adapters, inserted       an entire decoder       per-module
                     after self-attention,    block                   targets
                     FFN and cross-attention
  coefficients       one router per adapter   one scalar for the      one row per
                                              whole block             target
  routing            per example              batch mean              per example
                                                                      (Linears)
  balancing          expert dropout           expert dropout 0.1      same, 0.1

The consequences of that drift were measured, not theorized. Merging a whole
block under one scalar leaves the router injecting ``num_experts - 1`` free
numbers per forward - three, at N=4 - while paying for N copies of everything,
including the ``ffn_type: peer`` feedforward, which is the largest module in the
model and ALREADY a per-token mixture over its own product-key bank. And the
batch mean is worse than merely coarse: the loss reaches the routing only
through ``probs.mean(0)``, so every example receives the identical routing
gradient ``dL/dw / B``. A constant router is that design's fixed point, which is
what ``routing_input_dependence`` decaying to ~1e-5 on abstractinator-g, and to
exactly 0 on the first cut of this file, both were.

Two things here are not in the paper, and neither is novel:

  * BASE PLUS DEVIATIONS instead of N independent expert copies. With
    ``P_e = base + delta_e`` and coefficients summing to one,
    ``base + sum_e w_e delta_e == sum_e w_e P_e``, so this is the paper's merge
    written in a different basis. It buys exact identity at initialization and a
    shared trunk that receives full gradient whatever the routing does
    (``d(merged)/d(base) = sum_e w_e = 1``), so a starved deviation now costs its
    rank rather than a whole block. Large deviations are additionally
    rank-constrained, which is LoRA.
  * PEFT-STYLE TARGET DISCOVERY (praxis/routers/targeting.py), so the merge sites
    are found by walking the module tree instead of being hand-placed. Plumbing.

Honest limit. Routing is per EXAMPLE, never per TOKEN. Associativity buys the
former for Linear targets - see ``MergedLinear`` - but a per-token merge would
need a distinct geometry per position, which remains the province of the
fast-weights work in praxis/routers/prismatic.py. Elementwise and indexed
targets (norms, residual gates, the per-depth table) stay on the batch mean,
which is also what the paper does: it trains layernorm parameters but never
treats them as experts.

Sharpening is OFF by default, unlike VEAR - ``p**4`` drives the losing deviations
to zero gradient, which is the mechanism behind abstractinator-g's dead experts.
``ModularVEAR`` re-enables it for the comparison.

Companion: praxis/routers/targeting.py, praxis/routers/smear.py.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.routers.targeting import (
    DENSE_DELTA_MAX_NUMEL,
    TARGET_PROFILES,
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
#   "token"   - every position routes itself. Beyond the paper, and possible
#               only on the Linear targets, where associativity means the merged
#               weight is never materialized (see MergedLinear). Costs nothing
#               extra: the coefficient tensor gains a sequence axis the einsum
#               was already broadcasting over.
#   "example" - one routing per sequence, the paper's granularity. Default.
#   "batch"   - one routing for the whole batch, which is what this repo's old
#               whole-block SMEAR did. Kept for the controlled comparison ONLY.
#               It is very likely never the right choice: under a batch mean the
#               loss reaches the routing only through ``probs.mean(0)``, so every
#               example receives the identical routing gradient and a CONSTANT
#               router is the fixed point. Measured on abstractinator-m, where
#               input dependence decayed to exactly 0.
#
# Elementwise and indexed targets (norms, residual gates, the per-depth table)
# always reduce to one geometry per forward regardless: their merged parameter
# would have to vary per position, which needs a rewritten forward, not a
# reassociated one.
REDUCTIONS = ("token", "example", "batch")

# Rank of a factored deviation, as a divisor of the smaller weight dimension.
# Config-derived rather than a flag, the same way PEER sizes its bank against
# the dense FFN it replaces (praxis/dense/peer.py).
RANK_DIVISOR: int = 8
MIN_RANK: int = 4

# Probability of dropping an entire deviation from a routing decision. This is
# SMEAR's OWN load-balancing mechanism (the paper uses expert dropout, not an
# auxiliary balance loss or a DeepSeek-style bias), at the paper's rate, and it
# is what the first cut of this router was missing: nothing stopped one deviation per
# target from monopolizing its coefficient, and on abstractinator-m every one of
# the twelve targets duly saturated to near one-hot.
#
# Safe here in a way it is not under the original SMEAR. When every expert is
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
    PER EXAMPLE.

    This is the module that lets the modular router leave the batch mean behind, and the
    reason it can is associativity. Merging first and applying second needs one
    weight tensor per distinct coefficient vector, so per-example routing would
    mean a whole ``[B, out, in]`` stack. Applying first and merging second does
    not::

        y_b = (W + sum_e c_be B_e A_e) x_b
            = W x_b + sum_e c_be * B_e (A_e x_b)

    The right-hand form never materializes a merged weight. Cost per token is
    ``N * r * (in + out)`` against the base's ``in * out``; at ``r = min/8`` and
    ``N = 4`` that is roughly one extra base projection, and the only new
    activation is ``[B, T, N, r]``.

    WHY PER EXAMPLE IS THE POINT. Under a batch mean the loss reaches the
    routing only through ``probs.mean(0)``, so every example in the batch
    receives the SAME routing gradient, ``dL/dw / B``. No gradient path can
    express "example 1 wanted deviation 2 while example 5 wanted deviation 3",
    which makes a constant router the design's fixed point rather than a
    training failure - and ``smear_input_dependence`` decaying to exactly 0 is
    that fixed point being reached. Per-example merging restores the signal.

    The base ``weight`` and ``bias`` are the ORIGINAL Parameter objects, held
    directly rather than behind a nested Linear, so qualified names and any
    checkpoint written before this wrapper existed both still resolve
    (``attn.qkv.weight`` stays ``attn.qkv.weight``) and anything introspecting
    ``.weight`` / ``.in_features`` keeps working.
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
        if coeff is None or tuple(coeff.shape[:-1]) != tuple(
            x.shape[: coeff.dim() - 1]
        ):
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
        if target_profile not in TARGET_PROFILES:
            raise ValueError(
                f"Unknown target profile {target_profile!r}; "
                f"known: {sorted(TARGET_PROFILES)}"
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

        spec = TARGET_PROFILES[target_profile]
        self.targets, skipped = discover_targets(block, spec)
        if not self.targets:
            raise ValueError(
                f"Target profile {target_profile!r} matched nothing in "
                f"{type(block).__name__}. Skips: {skipped}"
            )
        self._skipped = skipped

        # Targets split by HOW they are merged, not by what they are:
        #
        #   per-example - the target is an nn.Linear, so associativity lets each
        #     example carry its own coefficients at low-rank cost (MergedLinear).
        #     These are the adapter-shaped modules, which is exactly what the
        #     SMEAR paper routes: adapters inserted after self-attention, the
        #     feed-forward and cross-attention.
        #   batch-mean - norms, residual gates, kappa/mu, the per-depth table.
        #     Elementwise or indexed rather than matmuls, so the associativity
        #     trick does not apply, and the paper does not treat them as experts
        #     either (it trains layernorm parameters but never routes them).
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
        # otherwise; that is why it is unconditional rather than a separate
        # class, which is what `arc_smear` / `arc_vear` used to be.
        self.depth = getattr(config, "depth", 1) or 1
        self.depth_bias = nn.Embedding(self.depth, len(self.targets) * self.num_experts)
        nn.init.zeros_(self.depth_bias.weight)

        self._metrics: Dict[str, float] = {}
        # On-device running sums over the passes since the last flush, and
        # how many passes went into them. See _log_metrics for why sampling
        # one pass instead was an aliasing bug.
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
                f"[SMEAR] {len(self.wrappers)} target(s) routed PER EXAMPLE "
                f"({', '.join(self.wrappers) or 'none'}); "
                f"{len(self._param_row)} parameter(s) on the batch-mean path; "
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

    def _coefficients(
        self, inputs: Tensor, current_depth: int
    ) -> Tuple[Tensor, Tensor]:
        """Merge coefficients ``[B, targets, experts]`` and the router's own probs.

        Returns the FULL per-example tensor, not a batch mean. Callers that
        cannot use per-example coefficients (the elementwise targets) take the
        mean themselves, which keeps the mean an explicit, local decision rather
        than something baked into every path.

        Pooling is ``inputs.mean(dim=1)`` - averaged over the sequence, which is
        what the SMEAR paper's router reads too. Per-TOKEN routing remains out
        of reach: it would need a distinct merged geometry per position, and the
        associativity trick buys per-example, not per-token.
        """
        # Per-token routing reads every position; the coarser reductions pool
        # over the sequence first, which is what the paper's router does.
        routed_input = inputs if self.reduction == "token" else inputs.mean(dim=1)
        logits = self._route_logits(self.router_norm(routed_input), current_depth)
        probs = F.softmax(logits, dim=-1)  # [..., T, N] - the router's own opinion

        merge = probs
        # Expert dropout: SMEAR's own load-balancing mechanism, drawn
        # independently per (example, target) so a deviation that would
        # otherwise monopolize a target keeps being made to share. Under
        # gradient checkpointing the forward re-runs during backward, and
        # torch.utils.checkpoint is called with use_reentrant=False and the
        # default preserve_rng_state=True (praxis/decoders/checkpoint.py), so
        # the recomputed pass draws the SAME mask the original did.
        if self.training and self.EXPERT_DROPOUT > 0:
            keep = torch.bernoulli(torch.full_like(merge, 1.0 - self.EXPERT_DROPOUT))
            merge = merge * keep
            # An all-dropped row renormalizes to zeros rather than NaN, and zero
            # coefficients mean "use the base unchanged" - see MODULAR_EXPERT_DROPOUT.
            merge = merge / merge.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.SHARPEN != 1.0:
            merge = merge.pow(self.SHARPEN)
            merge = merge / merge.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.reduction == "batch":
            # Every example shares one geometry. This is what `smear` and `vear`
            # used to do unconditionally, kept ONLY so the per-example claim can
            # be A/B'd against it - see REDUCTIONS for why it is not a default.
            merge = merge.mean(dim=0, keepdim=True).expand_as(merge)
        return merge, probs

    @staticmethod
    def _flatten(merge: Tensor) -> Tensor:
        """``[..., targets, experts]`` -> ``[targets, experts]``.

        The elementwise targets always merge to one geometry per forward, so
        they need every leading axis collapsed - one under "example", two under
        "token".
        """
        return merge.reshape(-1, merge.shape[-2], merge.shape[-1]).mean(dim=0)

    @contextmanager
    def _coefficient_scope(self, merge: Tensor) -> Iterator[None]:
        """Hand each per-example target its ``[B, N]`` coefficients for one
        block forward, then take them back.

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
        """Router-mode forward.

        Arity handling mirrors SMEAR's deliberately rather than sharing it: the
        legacy router is frozen for backward compatibility, so the two are kept
        independent. Seven positional arguments, or eight when an encoder
        supplies a byte timeline (praxis/layers/local.py).
        """
        if len(args) not in (7, 8):
            raise NotImplementedError(
                "SMEAR routers support router mode only (7 or 8 positional args "
                f"from LocalLayer); got {len(args)}."
            )
        layer, inputs, attention_mask, past_key_values = args[:4]
        current_state, current_depth, block_ids = args[4:7]
        positions = args[7] if len(args) == 8 else None

        merge, probs = self._coefficients(inputs, current_depth)

        # Elementwise and indexed targets still merge on the batch mean; the
        # Linear targets take their per-example rows through the scope below.
        mean_w = self._flatten(merge)
        merged = self._merged_state_dict(layer, mean_w)

        # Diagnostics run over a contiguous WINDOW of `depth` passes and then
        # sleep for METRICS_INTERVAL - 1 windows. See _log_metrics: sampling one
        # pass per interval aliased across recurrent depth, and sampling every
        # pass fixed that but cost +9.8% per pass, which at ~96 passes per
        # optimizer step is a third of the step. A window is long enough to span
        # every depth in a step and still touches only one pass in
        # METRICS_INTERVAL. Ordered after the merge because _delta_scale reads
        # the merged tensors rather than recomputing them.
        if (self._tick // self._window) % METRICS_INTERVAL == 0:
            with torch.no_grad():
                self._log_metrics(merge, probs)
                self._delta_scale(layer, merged, mean_w)
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

        # tie_weights=False for the same reason SMEAR needs it: the same module
        # is reparametrized once per recurrent pass, and functional_call's
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
        """Four scalars and one heatmap, deliberately.

        SMEAR emitted nine chart families keyed by ``layer_{depth}_``, which at
        depth 6 rendered 54 lines - every one of them the SAME router sampled at
        a different recurrent pass, since one router serves every depth. Nothing
        here carries a depth prefix, and the per-pass story is told by
        ``router_depth_specialization``, which is where the depths genuinely
        differ.

        Averaged over a contiguous WINDOW of passes rather than sampled from one
        of them. Sampling was an aliasing bug, not a cheaper approximation:
        the old code logged whenever ``_tick % METRICS_INTERVAL == 0`` while
        ``_tick`` advances once per recurrent pass, so at depth 6 and interval 10
        it reported depths 0, 2 and 4 and NEVER 1, 3 or 5. With
        ``router_depth_specialization`` at 0.78 on abstractinator-n those passes
        are not interchangeable, and what surfaced on the charts as a router
        oscillating between collapsed and diverse was in part the sampler
        rotating between depths that differ. Accumulation stays on-device and
        only the flush calls ``.item()``, so this still costs one host-device
        sync per window and not one per pass.
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

            if n > 1:
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
                # construction under reduction="batch", which is the honest
                # reading of that mode.
                picks = F.one_hot(sel.argmax(dim=-1), n).to(sel.dtype)  # [D, T, N]
                self._add(
                    "smear_selection_diversity",
                    (self._entropy(picks.mean(dim=0), dim=-1) / math.log(n)).mean(),
                )

                # THE number this design exists to produce. Mean pairwise L1
                # distance between different targets' coefficient rows, over its
                # maximum of 2. Zero means every module chose the same mixture,
                # so per-module granularity bought nothing and the block is back
                # to one scalar; high means the modules genuinely disagree, which
                # is the only thing that justifies the extra rows.
                if len(self.targets) > 1:
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

    def _delta_scale(self, layer: nn.Module, merged: Dict[str, Tensor], w: Tensor) -> None:
        """``||sum_e w_e delta_e|| / ||base||`` per target, Frobenius.

        The coefficients say how the deviations are MIXED; this says whether the
        mixture moves the geometry at all. Both are needed to read the design:
        a rich mixture over deviations that are numerically tiny is a router
        arguing about nothing, and is indistinguishable from a real one on the
        coefficient heatmap alone.

        Reported for the mean coefficients, matching the heatmap, even though the
        Linear targets apply their own row per example. Delta over BASE rather
        than over merged, so the number is read directly as "the routing moved
        this module's weights by N% of their own norm".
        """
        try:
            self._reset_accum_if_moved(w)
            self._delta_scale_inner(layer, merged, w)
        except Exception:
            # Optional telemetry must never reach the step. A diagnostic that
            # can raise into forward() is how a profiler took abstractinator-m
            # down; this one had the same shape.
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

        The only place this class touches the host, which is what keeps
        per-pass accumulation as cheap as the old per-interval sampling.
        """
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
