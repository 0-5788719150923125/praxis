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

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        *,
        branches: List[HeadSpec],
        gate_repulsion: float = 0.0,
    ) -> None:
        super().__init__(config, encoder)
        if not branches:
            raise ValueError("ParallelHead needs at least one branch.")
        built = [
            b if isinstance(b, BaseHead) else b(config, encoder=encoder)
            for b in branches
        ]
        self.branches = nn.ModuleList(built)
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
        return "Parallel(" + ", ".join(b.compose_repr() for b in self.branches) + ")"

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

    def transform(self, hidden_states: Tensor) -> Tensor:
        """The gated mixture of branch transforms - this head's contribution as
        a non-terminal SequentialHead stage."""
        if self.gate is None:
            return hidden_states
        outs = [b.transform(hidden_states) for b in self.branches]
        return self._gate_combine(outs, self.gate(hidden_states))

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """Standalone (terminal): gate-combine the branches' classifier outputs
        as a mixture of softmaxes, so the gate gradient stays scale-invariant
        across heterogeneous branch logit magnitudes (see _gate_combine_logits).
        Terminal ParallelHeads classify; a non-terminal one blends features via
        ``transform``/``_gate_combine`` instead."""
        if self.gate is None:
            return hidden_states
        outs = [b(hidden_states, **kwargs) for b in self.branches]
        return self._gate_combine_logits(outs, self.gate(hidden_states))

    @property
    def classifier(self) -> Optional[nn.Module]:
        # The gated arms read out differently, so there is no shared linear
        # projection for cut-CE (which is why crystal forbids it). But a
        # centroid loss (HALO) only needs *a* centroid matrix: lend it the
        # crystal arm's centers, preferring a branch that exposes ``centers``,
        # then any weight-bearing branch. NB: HALO scores the pre-head
        # embeddings against these centroids and ignores the gated logits, so
        # the harmonic/gate machinery sees little gradient under it.
        fallback = None
        for b in self.branches:
            c = getattr(b, "classifier", None)
            if c is None:
                continue
            if hasattr(c, "centers"):
                return c
            if fallback is None and hasattr(c, "weight"):
                fallback = c
        return fallback

    def set_downstream(self, classifier: Optional[nn.Module]) -> None:
        """Point every branch's grad-ratio at the real downstream classifier."""
        for b in self.branches:
            if hasattr(b, "set_downstream"):
                b.set_downstream(classifier)

    # ── Namespaced diagnostics ──────────────────────────────────────────────

    def aux_losses(self) -> dict:
        out: dict = {}
        for i, b in enumerate(self.branches):
            for k, v in b.aux_losses().items():
                out[f"p{i}_{k}"] = v
        # Pre-scaled, mirroring crystal's convention; omitted when off.
        if self._repulsion_lambda > 0.0 and self._gate_repulsion is not None:
            out["gate_repulsion"] = self._repulsion_lambda * self._gate_repulsion
        return out

    def training_metrics(self) -> dict:
        out: dict = {}
        for i, b in enumerate(self.branches):
            for k, v in b.training_metrics().items():
                out[f"p{i}_{k}"] = v
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
        return out

    def all_metric_descriptions(self) -> dict:
        from praxis.metrics.descriptions import resolve_callers

        out: dict = {}
        for i, b in enumerate(self.branches):
            callers = resolve_callers(b)
            for k, v in b.all_metric_descriptions().items():
                out[f"p{i}_{k}"] = self._namespace_entry(v, i, callers.get(k))
        out.update(self._gate_descriptions())
        return out

    def _namespace_entry(self, value: Any, i: int, caller: Optional[str]) -> Any:
        """Tag a branch's description with its index (title ``#i``, namespaced
        series group) and pin the producing leaf class as its caller."""
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
                hint["title"] = f"{hint['title']} #{i}"
        chart = entry.get("chart")
        if isinstance(chart, dict) and isinstance(chart.get("series_group"), str):
            chart["series_group"] = f"p{i}_{chart['series_group']}"
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
