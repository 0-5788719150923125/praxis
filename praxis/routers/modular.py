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

# Rank of a factored deviation, as a divisor of the smaller weight dimension.
# Config-derived rather than a flag, the same way PEER sizes its bank against
# the dense FFN it replaces (praxis/dense/peer.py).
RANK_DIVISOR: int = 8
MIN_RANK: int = 4

# Weight on the inter-expert repulsion, applied WITHIN each target's own
# coefficient rows so a target's deviations occupy distinct directions. Same
# scale as VEAR_REPULSION; carried over unchanged so the two are comparable.
MODULAR_REPULSION: float = 0.01

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
    training failure - and ``smear_modular_input_dependence`` decaying to exactly 0 is
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
        self.num_experts = int(num_experts)
        self.rank = int(rank)

        self.weight = base.weight
        self.bias = base.bias

        # LoRA's initialization: A random, B zero, so every deviation is exactly
        # zero at step 0 and the wrapper is bit-identical to the Linear it
        # replaced. A config swap stays a clean A/B rather than a reroll.
        self.lora_a = nn.Parameter(torch.empty(self.num_experts, rank, self.in_features))
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
        if coeff is None or coeff.shape[0] != x.shape[0]:
            return y
        # [B, ..., N, r] - each deviation's low-rank projection of the input.
        u = torch.einsum("...i,eri->...er", x, self.lora_a.to(x.dtype))
        # Broadcast [B, N] over however many sequence-ish axes sit between.
        shape = (coeff.shape[0],) + (1,) * (x.dim() - 2) + (coeff.shape[-1], 1)
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


class ModularSMEAR(nn.Module):
    """Per-module soft merging over a shared block plus N deviations."""

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
        num_experts: int = 4,
        target_profile: str = "unrouted",
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if block is None:
            raise ValueError(
                "Modular SMEAR routers merge deviations against a concrete block; pass "
                "block=... at construction (see praxis/decoders/base.py)."
            )
        if target_profile not in TARGET_PROFILES:
            raise ValueError(
                f"Unknown target profile {target_profile!r}; "
                f"known: {sorted(TARGET_PROFILES)}"
            )

        self.num_experts = int(num_experts)
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
            merged_numel += sum(
                _get_param(block, p).numel() for p in group.params
            )
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

        self._metrics: Dict[str, float] = {}
        self._tick: int = 0

        if verbose:
            print(describe(self.targets, skipped))
            print(
                f"[SMEAR/modular] {len(self.targets)} targets x {self.num_experts} experts; "
                f"deviations hold {self.delta_numel:,} parameters "
                f"({self.delta_numel / max(1, merged_numel):.2f}x the merged base)"
            )
            print(
                f"[SMEAR/modular] {len(self.wrappers)} target(s) routed PER EXAMPLE "
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
        return logits.view(-1, len(self.targets), self.num_experts)

    def _coefficients(self, inputs: Tensor, current_depth: int) -> Tuple[Tensor, Tensor]:
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
        pooled = self.router_norm(inputs.mean(dim=1))
        logits = self._route_logits(pooled, current_depth)
        probs = F.softmax(logits, dim=-1)  # [B, T, N] - the router's own opinion

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
        return merge, probs

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
                self.wrappers[label]._coeff = merge[:, row, :]
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
                "Modular SMEAR routers support router mode only (7 or 8 positional args "
                f"from LocalLayer); got {len(args)}."
            )
        layer, inputs, attention_mask, past_key_values = args[:4]
        current_state, current_depth, block_ids = args[4:7]
        positions = args[7] if len(args) == 8 else None

        merge, probs = self._coefficients(inputs, current_depth)

        if self._tick % METRICS_INTERVAL == 0:
            with torch.no_grad():
                self._log_metrics(merge, probs)
        self._tick += 1

        # Elementwise and indexed targets still merge on the batch mean; the
        # Linear targets take their per-example rows through the scope below.
        merged = self._merged_state_dict(layer, merge.mean(dim=0))

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

    def router_aux_loss(self) -> Dict[str, Tensor]:
        """Inter-expert repulsion, collected ONCE per step outside the forward.

        Parameter-only, so computing it inside the gradient-checkpointed
        recurrent forward would both add it once per pass and let it escape the
        checkpointed region - the two hazards VEAR documents. Repulsion is
        computed WITHIN each target's own expert rows: the goal is that a given
        module's deviations point in distinct directions, not that unrelated
        modules disagree with each other.
        """
        if not self.training or self.num_experts < 2:
            return {}
        w = self.router.weight.view(len(self.targets), self.num_experts, -1)
        w = F.normalize(w, dim=-1)
        sim = torch.bmm(w, w.transpose(1, 2))  # [T, N, N]
        eye = torch.eye(self.num_experts, device=w.device, dtype=w.dtype)
        off = (sim - eye).abs().sum() / (
            len(self.targets) * self.num_experts * (self.num_experts - 1)
        )
        return {"smear_modular_repulsion": MODULAR_REPULSION * off}

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
        here carries a depth prefix: each pass overwrites the last, so what
        surfaces is the most recent pass, the same convention
        ``ArcSingleAttention._record_gate`` uses. The per-pass story is told by
        ``router_depth_specialization`` on the arc subclass, which is the only
        place it is actually a different object.
        """
        try:
            n = self.num_experts
            w = merge.mean(dim=0)  # [T, N] - what the heatmap reports either way
            # The heatmap: every target's merge weights, one card.
            for t, group in enumerate(self.targets):
                for e in range(n):
                    self._metrics[f"smear_modular_coeff_{group.label}_{e}"] = w[t, e].item()

            # Are the deviations being USED, or has each target picked one and
            # abandoned the rest? Fraction of (target, expert) pairs carrying
            # more than half their fair share, averaged over targets: 1.0 at
            # perfect balance, 1/N at total collapse. This is the number expert
            # dropout is there to hold up, so it is the direct read on whether
            # the paper's balancing mechanism is working.
            self._metrics["smear_modular_expert_utilization"] = (
                (w > (0.5 / n)).float().sum(dim=-1).mean() / n
            ).item()

            if n > 1:
                # Does the router read its input? I(input; expert) per target,
                # normalized to [0, 1], then averaged. Zero means every sequence
                # in the batch got the same coefficients, i.e. the router is a
                # constant and the whole mechanism is a reparametrized base.
                mean_p = probs.mean(dim=0)  # [T, N]
                h_mean = self._entropy(mean_p, dim=-1)  # [T]
                h_seq = self._entropy(probs, dim=-1).mean(dim=0)  # [T]
                mi = ((h_mean - h_seq) / math.log(n)).clamp(0.0, 1.0)
                self._metrics["smear_modular_input_dependence"] = mi.mean().item()
                self._metrics["smear_modular_input_dependence_max"] = mi.max().item()

                # THE number this design exists to produce. Mean pairwise L1
                # distance between different targets' coefficient rows, over its
                # maximum of 2. Zero means every module chose the same mixture,
                # so per-module granularity bought nothing and the block is back
                # to one scalar; high means the modules genuinely disagree, which
                # is the only thing that justifies the extra rows.
                if len(self.targets) > 1:
                    d = (w.unsqueeze(0) - w.unsqueeze(1)).abs().sum(-1)  # [T, T]
                    t_count = len(self.targets)
                    self._metrics["smear_modular_target_dispersion"] = (
                        d.sum() / (t_count * (t_count - 1) * 2.0)
                    ).item()
        except Exception:
            # Diagnostics never break a step; they write to a plain dict and
            # never reach the loss.
            pass

    def get_metrics(self) -> dict:
        return self._metrics.copy()


class ModularVEAR(ModularSMEAR):
    """Modular SMEAR with VEAR's sharpening, for the controlled comparison.

    Kept because the sharpening question is worth answering per-module rather
    than assumed: under the base-plus-deviation reparametrization a starved
    deviation no longer costs a whole block, so the failure mode that made
    ``p**4`` destructive in VEAR is materially cheaper here. It is still off by
    default - see the module docstring.
    """

    SHARPEN: float = 4.0


class DepthBiasedModular:
    """Per-recurrent-pass additive bias on the per-target routing logits.

    The ArcAttention idiom (praxis/attention/arc.py) and the exact analogue of
    ``DepthBiasedRouting`` in praxis/routers/arc_smear.py, widened from
    ``num_experts`` to ``num_targets * num_experts`` so each pass can move each
    module independently. Zero-init, so it is identity against its parent.
    """

    def _init_depth_bias(self, config: Any) -> None:
        self.depth = getattr(config, "depth", 1) or 1
        self.depth_bias = nn.Embedding(
            self.depth, len(self.targets) * self.num_experts
        )
        nn.init.zeros_(self.depth_bias.weight)

    def _route_logits(self, router_input: Tensor, current_depth: int) -> Tensor:
        logits = super()._route_logits(router_input, current_depth)
        # The decoder can loop past `depth` when halting samples deeper than the
        # table; wrap rather than raise, so a pass reuses an existing row.
        idx = int(current_depth) % self.depth
        bias = self.depth_bias(torch.tensor(idx, device=logits.device))
        return logits + bias.view(len(self.targets), self.num_experts)

    def get_metrics(self) -> dict:
        from praxis.metrics.specialization import depth_dispersion

        out = super().get_metrics()
        stats = depth_dispersion(self.depth_bias.weight)
        if stats is not None:
            out["router_depth_specialization"] = stats["specialization"]
            out["router_depth_similarity"] = stats["similarity"]
        return out


class ArcModularSMEAR(DepthBiasedModular, ModularSMEAR):
    """Modular SMEAR with a per-recurrent-pass routing bias."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self._init_depth_bias(config)


class ArcModularVEAR(DepthBiasedModular, ModularVEAR):
    """Modular SMEAR with sharpening and a per-recurrent-pass routing bias."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self._init_depth_bias(config)
