"""ParallelHead: run standardized Praxis heads side by side, gate-combined.

Where :class:`~praxis.heads.stacked.SequentialHead` chains heads (each stage's
``transform`` composes, the terminal classifies), ParallelHead runs its branch
heads on the *same* input and blends their ``transform`` outputs with a learned
per-token softmax gate::

    w = softmax(gate(h))                 # [..., n_branches]
    out = sum_i w[..., i] * branch_i.transform(h)

The gate forces the branches to balance their contributions per token - an
ablation / XOR-style decision rather than a fixed pipeline.

It works at two levels. As a non-terminal SequentialHead stage it blends branch
``transform`` outputs (feature-level). As a terminal/top head it blends the
branches' ``forward`` outputs (logit-level) and is itself the model's head. The
``prismatic`` profile uses the latter as a top-level split that balances
bias against variance per token::

    Parallel(Sequential(HarmonicField), Sequential(HarmonicField, CrystalClassifier))

- branch 0 = a harmonic field read out by a plain linear head (a strong
  structural prior - the bias arm),
- branch 1 = a harmonic field refracted through the crystal distance
  classifier (the more expressive variance arm).

The gate exposes no single linear projection (the two arms read out
differently), so there is no classifier for cut-CE - fine because crystal
forbids it, so prismatic trains on full logits. A centroid loss (HALO) instead
borrows the crystal arm's centers via ``classifier`` (see that property).

Branches are passed as *builders* (a head class or ``functools.partial`` over
one), exactly like SequentialHead. Because two branches can share a class (two
``HarmonicField``s emit identical metric keys), every branch's metrics,
snapshots, aux losses and chart descriptions are namespaced under a ``p{i}_``
prefix; per-branch cards get a ``#i`` title suffix and keep the producing leaf
class as their caller, so they render independently on the dashboard.
"""

import copy
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead

HeadSpec = Union[BaseHead, Callable[..., BaseHead]]


class ParallelHead(BaseHead):
    """Gate-combined parallel branches; a SequentialHead stage or top head."""

    # A composed head ties via a self-tying branch (e.g. crystal), so the model
    # keeps it under tie_word_embeddings rather than swapping in TiedWeights.
    self_ties = True

    # Floors the log-gap so an exact tie is a bounded (not infinite) penalty and
    # the gradient stays finite. Fixed, model-agnostic.
    _REPULSION_EPS = 1e-2

    @property
    def causal_readout(self) -> bool:
        """A gated combination is causal only if every branch is: one
        sequence-pooling branch contaminates the combined logits. A stem feeds
        every branch that does not read the trunk, so it counts too."""
        parts = list(self.branches)
        if self.stem is not None:
            parts.append(self.stem)
        return all(getattr(b, "causal_readout", False) for b in parts)

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        *,
        branches: List[HeadSpec],
        gate_repulsion: float = 0.0,
        stem: Optional[HeadSpec] = None,
    ) -> None:
        """``stem`` is an optional transform applied ONCE and shared by every
        branch, instead of each branch carrying its own copy.

        prismatic2 through prismatic5 give each arm its own ``HarmonicField``,
        which is why three arms cost three field evaluations - measured at ~30%
        of total compute in ``abstractinator-j``, against 53-69% dormant
        capacity per field. A stem computes the field once and lets the arms
        differ only in how they READ it, which is the distinction the gate
        actually rewarded there (arm 1 field->crystal at 0.736, arm 2
        field->linear at 0.234, against the separate-bias arm at 0.029).

        A branch may opt out with ``reads_trunk = True`` and receive the raw
        hidden states instead. HaloHead sets it: HALOLoss scores the trunk
        embeddings, so putting a transform in front would train one feature
        space and score another. The GATE always reads the trunk, so its
        decision stays a judgement about the arms rather than about the stem.

        Default None, so every existing prismatic profile is unchanged.
        """
        super().__init__(config, encoder)
        if not branches:
            raise ValueError("ParallelHead needs at least one branch.")
        built = [
            b if isinstance(b, BaseHead) else b(config, encoder=encoder)
            for b in branches
        ]
        self.branches = nn.ModuleList(built)
        self.stem: Optional[BaseHead] = (
            None
            if stem is None
            else (stem if isinstance(stem, BaseHead) else stem(config, encoder=encoder))
        )
        self._gate_mean: Optional[Tensor] = None
        self._gate_entropy: Optional[float] = None
        self._gate_min_gap: Optional[float] = None
        self._gate_repulsion: Optional[Tensor] = None
        # Level-repulsion strength on the gate weights (0 = off), bound by the
        # head-registry profile (e.g. prismatic3_repel), not a config flag.
        # Drives the mean per-branch weights to DISTINCT tiers (e.g. 70/20/10),
        # penalizing near-ties (70/15/15) like repelling energy levels. NB: with
        # 2 branches the only tie is 50/50, so repulsion there reduces to
        # winner-take-all; it's meant for 3+ branches.
        self._repulsion_lambda = float(gate_repulsion or 0.0)

        # Size the gate to the feature dim the branches transform (encoder
        # layout in encoder mode, else config hidden size). When the encoder
        # owns the whole output pipeline, output_dims() is None and there's
        # nothing to gate - the head passes through (mirrors HarmonicHead).
        dims = self.output_dims()
        if dims is None:
            self.gate = None
        else:
            feature_dim, _ = dims
            self.gate = nn.Linear(feature_dim, len(self.branches), bias=False)

    def compose_repr(self) -> str:
        arms = ", ".join(b.compose_repr() for b in self.branches)
        if self.stem is None:
            return f"Parallel({arms})"
        return f"Parallel({self.stem.compose_repr()} -> [{arms}])"

    def __repr__(self) -> str:
        return self.compose_repr()

    def _gate_weights(self, gate_logits: Tensor) -> Tensor:
        """Per-token softmax gate weights, plus the cached diagnostics and the
        training-only level-repulsion shared by both combine paths."""
        w = torch.softmax(gate_logits, dim=-1)  # [..., n]
        self._update_gate_stats(w)
        if self.training and self._repulsion_lambda > 0.0 and len(self.branches) > 1:
            self._gate_repulsion = self._level_repulsion(w)
        return w

    def _gate_combine(self, outputs: List[Tensor], gate_logits: Tensor) -> Tensor:
        """FEATURE blend (non-terminal ``transform``): weighted sum of the
        branches' feature outputs. There is no distribution to mix here, so the
        raw-output blend is correct; only the terminal classify path swaps to a
        softmax mixture (see ``_gate_combine_logits``)."""
        w = self._gate_weights(gate_logits)
        stacked = torch.stack(outputs, dim=-1)  # [..., d, n]
        return (stacked * w.unsqueeze(-2)).sum(dim=-1)  # [..., d]

    def _gate_combine_logits(
        self, outputs: List[Tensor], gate_logits: Tensor
    ) -> Tensor:
        """DISTRIBUTION blend (terminal classify): a mixture of softmaxes rather
        than a weighted sum of raw logits::

            log p = logsumexp_i( log w_i + log_softmax(logits_i) )

        This is scale-invariant - the gate gradient couples to bounded log-probs
        instead of the branches' raw logit magnitudes, which span |64| on the
        linear arms vs ~8 on the crystal arm and were hammering the tiny gate
        weight (~17x grad/weight, the persistent clip source). The result is a
        normalized log-prob (``sum_v exp = 1``, ``max <= 0``), so cross-entropy
        and argmax are unchanged and crystal's ``max~0`` logit contract holds for
        free."""
        w = self._gate_weights(gate_logits)
        logw = w.clamp_min(1e-9).log().unsqueeze(-2)  # [..., 1, n]
        logp = torch.stack(
            [torch.log_softmax(o, dim=-1) for o in outputs], dim=-1
        )  # [..., V, n]
        return torch.logsumexp(logp + logw, dim=-1)  # [..., V]

    def _level_repulsion(self, w: Tensor) -> Tensor:
        """Pairwise log-gap repulsion on the mean branch weights (grad-carrying).

        ``-mean_{i<j} log(|m_i - m_j| + eps)`` over the batch-mean weight vector
        ``m``. Small as the tiers separate, large (bounded by eps) as any two
        approach equality - so the optimizer is pushed to keep them distinct.
        """
        m = w.reshape(-1, w.shape[-1]).mean(dim=0)  # [n], sums to 1
        diff = (m.unsqueeze(0) - m.unsqueeze(1)).abs()
        iu = torch.triu_indices(m.numel(), m.numel(), offset=1, device=m.device)
        gaps = diff[iu[0], iu[1]]
        return -(gaps + self._REPULSION_EPS).log().mean()

    def _update_gate_stats(self, w: Tensor) -> None:
        """Cache cheap gate diagnostics from the latest forward, for logging."""
        with torch.no_grad():
            flat = w.reshape(-1, w.shape[-1])
            self._gate_mean = flat.mean(dim=0)
            p = flat.clamp_min(1e-9)
            self._gate_entropy = float((-(p * p.log()).sum(dim=-1)).mean().item())
            # Smallest gap between mean branch weights: -> 0 when two branches
            # become equally important (the degeneracy the repulsion fights).
            if self._gate_mean.numel() > 1:
                d = (self._gate_mean.unsqueeze(0) - self._gate_mean.unsqueeze(1)).abs()
                iu = torch.triu_indices(
                    self._gate_mean.numel(), self._gate_mean.numel(), offset=1
                )
                self._gate_min_gap = float(d[iu[0], iu[1]].min().item())

    def _stem_out(self, hidden_states: Tensor) -> Tensor:
        """The shared stem transform, computed ONCE per call (the whole point:
        one field evaluation instead of one per arm). Identity when no stem."""
        if self.stem is None:
            return hidden_states
        return self.stem.transform(hidden_states)

    def _branch_input(
        self, branch: BaseHead, stemmed: Tensor, trunk: Tensor
    ) -> Tensor:
        """Stem output, unless the branch declares it must score trunk features
        (``reads_trunk``, set by HaloHead - see __init__)."""
        return trunk if getattr(branch, "reads_trunk", False) else stemmed

    def transform(self, hidden_states: Tensor) -> Tensor:
        """The gated mixture of branch transforms - this head's contribution as
        a non-terminal SequentialHead stage."""
        if self.gate is None:
            return hidden_states
        stemmed = self._stem_out(hidden_states)
        outs = [
            b.transform(self._branch_input(b, stemmed, hidden_states))
            for b in self.branches
        ]
        return self._gate_combine(outs, self.gate(hidden_states))

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """Standalone (terminal): gate-combine the branches' classifier outputs
        as a mixture of softmaxes, so the gate gradient stays scale-invariant
        across heterogeneous branch logit magnitudes (see _gate_combine_logits).
        Terminal ParallelHeads classify; a non-terminal one blends features via
        ``transform``/``_gate_combine`` instead."""
        if self.gate is None:
            return hidden_states
        stemmed = self._stem_out(hidden_states)
        # A branch marked detach_in_blend (the HALO arm) contributes its
        # distribution to the mixture but no gradient flows into it from the
        # blended CE: the gate still learns how much to trust the arm (its
        # gradient rides the mixture weights), while the arm's parameters
        # train solely under their own objective (HALOLoss's geometric
        # terms). Keeps the gate share an uncontaminated verdict on the arm.
        outs = [
            (
                b(self._branch_input(b, stemmed, hidden_states), **kwargs).detach()
                if getattr(b, "detach_in_blend", False) and self.training
                else b(self._branch_input(b, stemmed, hidden_states), **kwargs)
            )
            for b in self.branches
        ]
        return self._gate_combine_logits(outs, self.gate(hidden_states))

    @property
    def classifier(self) -> Optional[nn.Module]:
        # The gated arms read out differently, so there is no shared linear
        # projection for cut-CE (which is why crystal forbids it). A centroid
        # loss (HALO) wants a dedicated HALO arm above all (``is_halo``, the
        # prismatic5 branch): HALOLoss then runs its honest composite mode -
        # CE on the blended logits for the gate/other arms, the geometric
        # objective for the HALO arm - so every branch keeps a training
        # signal. Lacking one, fall back to lending a crystal arm's centers,
        # then any weight-bearing branch (the legacy side-loss mode; note the
        # harmonic/gate machinery sees little gradient under it).
        centers_fallback = None
        weight_fallback = None
        for b in self.branches:
            c = getattr(b, "classifier", None)
            if c is None:
                continue
            if getattr(c, "is_halo", False):
                return c
            if centers_fallback is None and hasattr(c, "centers"):
                centers_fallback = c
            if weight_fallback is None and hasattr(c, "weight"):
                weight_fallback = c
        return centers_fallback or weight_fallback

    def set_downstream(self, classifier: Optional[nn.Module]) -> None:
        """Point every branch's grad-ratio at the real downstream classifier."""
        for b in self.branches:
            if hasattr(b, "set_downstream"):
                b.set_downstream(classifier)
        # The stem feeds a MIXTURE of readouts, so no single classifier is "the"
        # downstream one. Lend it the same target the branches got; the
        # grad-ratio it reports is then a ratio against that readout, not
        # against the blend. Read it as a trend, not an absolute.
        if self.stem is not None and hasattr(self.stem, "set_downstream"):
            self.stem.set_downstream(classifier)

    # ── Namespaced diagnostics ──────────────────────────────────────────────

    def aux_losses(self) -> dict:
        out: dict = {}
        for i, b in enumerate(self.branches):
            for k, v in b.aux_losses().items():
                out[f"p{i}_{k}"] = v
        # The stem is not an arm, so it gets its own namespace rather than a
        # p{i}_ slot - otherwise its series would collide with an arm's the
        # moment the arm count changes.
        if self.stem is not None:
            for k, v in self.stem.aux_losses().items():
                out[f"stem_{k}"] = v
        # Pre-scaled, mirroring crystal's convention; omitted when off.
        if self._repulsion_lambda > 0.0 and self._gate_repulsion is not None:
            out["gate_repulsion"] = self._repulsion_lambda * self._gate_repulsion
        return out

    def training_metrics(self) -> dict:
        out: dict = {}
        for i, b in enumerate(self.branches):
            for k, v in b.training_metrics().items():
                out[f"p{i}_{k}"] = v
        if self.stem is not None:
            for k, v in self.stem.training_metrics().items():
                out[f"stem_{k}"] = v
        if self._gate_mean is not None:
            for i in range(len(self.branches)):
                out[f"gate_weight_{i}"] = float(self._gate_mean[i].item())
            out["gate_entropy"] = self._gate_entropy
            if self._gate_min_gap is not None:
                out["gate_min_gap"] = self._gate_min_gap
        return out

    def dashboard_snapshots(self) -> dict:
        out: dict = {}
        for i, b in enumerate(self.branches):
            for k, v in b.dashboard_snapshots().items():
                out[f"p{i}_{k}"] = v
        if self.stem is not None:
            for k, v in self.stem.dashboard_snapshots().items():
                out[f"stem_{k}"] = v
        return out

    def all_metric_descriptions(self) -> dict:
        from praxis.metrics.descriptions import resolve_callers

        out: dict = {}
        for i, b in enumerate(self.branches):
            callers = resolve_callers(b)
            for k, v in b.all_metric_descriptions().items():
                out[f"p{i}_{k}"] = self._namespace_entry(v, f"p{i}", f"#{i}", callers.get(k))
        if self.stem is not None:
            callers = resolve_callers(self.stem)
            for k, v in self.stem.all_metric_descriptions().items():
                out[f"stem_{k}"] = self._namespace_entry(
                    v, "stem", "(shared)", callers.get(k)
                )
        out.update(self._gate_descriptions())
        return out

    def _namespace_entry(
        self, value: Any, prefix: str, label: str, caller: Optional[str]
    ) -> Any:
        """Tag a part's description with its slot (title suffix ``label``,
        ``prefix``-namespaced series group) and pin the producing leaf class as
        its caller. ``prefix`` is ``p{i}`` for an arm and ``stem`` for the
        shared stem, matching the keys the metrics dicts emit."""
        if isinstance(value, str):
            entry: dict = {"description": value}
            if caller:
                entry["caller"] = caller
            return entry
        if not isinstance(value, dict):
            return value
        entry = copy.deepcopy(value)
        for hint_key in ("chart", "snapshot"):
            hint = entry.get(hint_key)
            if isinstance(hint, dict) and isinstance(hint.get("title"), str):
                hint["title"] = f"{hint['title']} {label}"
        chart = entry.get("chart")
        if isinstance(chart, dict) and isinstance(chart.get("series_group"), str):
            chart["series_group"] = f"{prefix}_{chart['series_group']}"
        if caller:
            entry["caller"] = caller
        return entry

    def _gate_descriptions(self) -> dict:
        out: dict = {}
        for i in range(len(self.branches)):
            out[f"gate_weight_{i}"] = {
                "description": (
                    "Mean per-token softmax weight the gate assigns to this "
                    "parallel branch. The branches compete to explain each "
                    "token; a weight pinned near 0 or 1 means the gate has "
                    "specialized."
                ),
                "chart": {
                    "title": "Parallel Gate Weights",
                    "y_label": "Mean Gate Weight",
                    "group": "parallel_head",
                    "group_order": 60,
                    "order": 10,
                    "series_group": "parallel_gate",
                    "series_label": f"branch {i}",
                },
                "caller": "ParallelHead",
            }
        out["gate_entropy"] = {
            "description": (
                "Entropy of the per-token branch gate (nats). High = branches "
                "share the work evenly; low = the gate commits to one branch "
                "(XOR-like specialization)."
            ),
            "chart": {
                "title": "Parallel Gate Entropy",
                "y_label": "Entropy (nats)",
                "group": "parallel_head",
                "order": 20,
            },
            "caller": "ParallelHead",
        }
        out["gate_min_gap"] = {
            "description": (
                "Smallest gap between any two mean branch weights. Near 0 means "
                "two branches have become equally important (degenerate tiers) - "
                "what the gate repulsion (prismatic3_repel) pushes apart. Larger = "
                "cleanly ranked tiers (e.g. 70/20/10)."
            ),
            "chart": {
                "title": "Parallel Gate Min Gap",
                "y_label": "Min Weight Gap",
                "group": "parallel_head",
                "order": 30,
            },
            "caller": "ParallelHead",
        }
        return out
