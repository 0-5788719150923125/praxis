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

    Parallel(arms=[Sequential(HarmonicField),
                   Sequential(HarmonicField, CrystalClassifier)])

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

import contextlib
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.heads.base import BaseHead

HeadSpec = Union[BaseHead, Callable[..., BaseHead]]


# ── Per-arm objectives and gradient surgery ─────────────────────────────────
#
# WHAT THIS IS FOR. A gated mixture combines PREDICTIONS well - it is a mixture
# of softmaxes, the standard construction - but it also decides, as a side
# effect, how much each arm gets TRAINED. Cross-entropy on the mixture reaches
# arm i scaled by its posterior responsibility
# ``w_i p_i(y) / sum_j w_j p_j(y)``, so an arm the gate has stopped trusting
# stops receiving gradient and cannot recover. Measured on abstractinator-n:
# gate shares 0.972 / 6.3e-08 / 0.028, entropy collapsed 1.058 -> 0.0015. The
# second arm is not merely losing, it is numerically unreachable.
#
# The fix is not a different blend. A mixture of softmaxes is the standard way
# to combine finished predictions and there is nothing wrong with it. The fix
# is to stop letting the blend own the TRAINING signal: give every arm its own
# cross-entropy against the labels, so each trains as a standalone classifier.
#
# THAT TURNS THE HEAD INTO A GENUINE MULTI-TASK PROBLEM, which is where the
# Jacobian comes in: N objectives over one shared trunk, stacked one row per
# arm and combined by something other than a plain sum.
#
# WHY IT IS CHEAP, contrary to the obvious cost model. It needs no extra trunk
# forward and no extra trunk backward. The arms branch at ONE point, and
# surgery is only needed where parameters are SHARED, so the per-arm gradients
# are taken with respect to that ACTIVATION - N backwards through N small
# classifiers, never through the trunk. The corrected gradient reaches the
# trunk once, through the surrogate in ``arm_objectives``.

# Steps between arm-conflict measurements. Sampled for the reason
# ObjectiveConflict samples: each costs a few small backwards. Baked.
ARM_CONFLICT_INTERVAL: int = 100


def _pcgrad(grads: List[Tensor]) -> Tensor:
    """PCGrad (Yu et al. 2020): drop each task's conflicting component, then sum.

    For every ordered pair, if ``g_i . g_j < 0`` the two objectives disagree
    about the shared representation, and ``g_i`` loses its projection onto
    ``g_j``::

        g_i <- g_i - (g_i . g_j / ||g_j||^2) g_j

    Non-conflicting pairs are left exactly alone, so with no conflict anywhere
    the result is bit-identical to the plain sum. That is what makes it safe as
    a default rather than a bet: it can only act where there is something to
    act on. It is also the only rule in its family with no hyperparameter,
    which is why it is the one here.

    ONE DELIBERATE DEVIATION FROM THE PAPER. Yu et al. project sequentially -
    each removal feeds the next test - which makes the result depend on the
    order tasks are visited, and they randomize that order every step to
    unbias it. Here every projection is measured against the ORIGINAL ``g_j``,
    so the result is order-independent and no RNG enters the training path.
    That matters in this codebase: a diagnostic drawing from the global RNG has
    already once perturbed the stream of the run it was watching.

    The cost of the deviation is honest and small: with three or more mutually
    conflicting rows the result is not exactly the projection onto the
    intersection of the half-spaces, only onto each one measured independently.
    With two rows the two agree exactly.
    """
    flat = [g.flatten() for g in grads]
    out = []
    for i, gi in enumerate(flat):
        proj = gi.clone()
        for j, gj in enumerate(flat):
            if i == j:
                continue
            dot = torch.dot(gi, gj)
            if dot < 0:
                proj = proj - (dot / gj.dot(gj).clamp_min(1e-12)) * gj
        out.append(proj)
    return torch.stack(out).sum(0).view_as(grads[0])


class ParallelHead(BaseHead):
    """Gate-combined parallel branches; a SequentialHead stage or top head."""

    # A composed head ties via a self-tying branch (e.g. crystal), so the model
    # keeps it under tie_word_embeddings rather than swapping in TiedWeights.
    self_ties = True

    # Floors the log-gap so an exact tie is a bounded (not infinite) penalty and
    # the gradient stays finite. Fixed, model-agnostic.
    _REPULSION_EPS = 1e-2

    # Per-arm solo objectives + PCGrad on the shared trunk. Measurement
    # (``arm_conflict``) is universal so every prismatic profile reports its own
    # arm Jacobian; the INTERVENTION is opt-in and only SurgicalParallelHead
    # turns it on, so prismatic2-8 train exactly as they did.
    arm_surgery: bool = False

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
        self._arm_metrics: dict = {}
        self._arm_step = 0
        self._arm_surgery_norm: Optional[float] = None
        # Set when an arm declined to supply its own objective, which makes the
        # Jacobian incomplete and disables the surgery outright.
        self._arm_gap = False
        # Level-repulsion strength on the gate weights (0 = off), bound by the
        # head-registry profile (e.g. prismatic3_repel), not a config flag.
        # Drives the mean per-branch weights to DISTINCT tiers (e.g. 70/20/10),
        # penalizing near-ties (70/15/15) like repelling energy levels. NB: with
        # 2 branches the only tie is 50/50, so repulsion there reduces to
        # winner-take-all; it's meant for 3+ branches.
        self._repulsion_lambda = float(gate_repulsion or 0.0)
        # See SurgicalParallelHead. False everywhere else, so prismatic2-8 are
        # bit-for-bit unchanged.
        self.detach_gate_input = getattr(type(self), "detach_gate_input", False)

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
        """Blueprint label, in the keyword style every other module uses.

        This used to read ``Parallel(HarmonicField -> [A, B, C])``. The arrow
        was invented notation - no torch module reprs like that - and it was
        also WRONG about the wiring, because it implied the stem feeds every
        arm. An arm with ``reads_trunk`` (HALO) branches ABOVE the stem and
        scores the raw trunk hidden states, since HALOLoss scores those same
        features and a transform in front would train one feature space and
        score another. So arms that bypass the stem say so.
        """
        arms = ", ".join(
            (
                f"{b.compose_repr()}(reads_trunk=True)"
                if getattr(b, "reads_trunk", False)
                else b.compose_repr()
            )
            for b in self.branches
        )
        if self.stem is None:
            return f"Parallel(arms=[{arms}])"
        return f"Parallel(stem={self.stem.compose_repr()}, arms=[{arms}])"

    def __repr__(self) -> str:
        return self.compose_repr()

    @contextlib.contextmanager
    def undetached(self):
        """Restore the ordinary gradient path through this head, temporarily.

        A surgical head detaches every arm in the blend and detaches the gate's
        input, so the mixture cross-entropy trains ONLY the gate - the arms have
        their own objectives instead. That is right for the main loss and wrong
        for every other consumer of the same head, because detaching an arm's
        OUTPUT severs the path back to the head's INPUT.

        MTP is that other consumer. It transforms the trunk's hidden states into
        draft states and classifies them with this head, so under the detached
        blend its loss reaches nothing at all - not the arms, not the trunk, and
        not even MTP's own bank. Measured: prismatic9 left 0 of 9 MTP parameters
        with a gradient where prismatic8 had 9 of 9, which is why every
        ``mtp_field_*`` series sat frozen at its initialization.

        Callers that own their own objective and want the whole head trained by
        it wrap their use of the head in this.
        """
        saved = [getattr(b, "detach_in_blend", False) for b in self.branches]
        saved_gate = self.detach_gate_input
        for b in self.branches:
            b.detach_in_blend = False
        self.detach_gate_input = False
        try:
            yield
        finally:
            for b, was in zip(self.branches, saved):
                b.detach_in_blend = was
            self.detach_gate_input = saved_gate

    def _gate_in(self, hidden_states: Tensor) -> Tensor:
        """What the gate projects. Detached under a surgical head so the gate's
        loss cannot bypass the arm arbitration on its way to the trunk."""
        if self.detach_gate_input and self.training:
            return hidden_states.detach()
        return hidden_states

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

    @torch.compiler.disable
    def _update_gate_stats(self, w: Tensor) -> None:
        """Cache cheap gate diagnostics from the latest forward, for logging.

        ``torch.compiler.disable`` because this is telemetry, and the two
        ``.item()`` calls below make Dynamo guard on a FLOAT VALUE - so a new
        graph is compiled every time the gate entropy changes, which is every
        step. Measured on abstractinator-q: this frame hit
        ``config.recompile_limit (8)`` on its own with
        ``last reason: ___stack2 == 1.047181248664856``, after which Dynamo
        gives up and runs it eager anyway. Same call made for the same reason as
        the patcher and the residual VQ (see project notes, 2026-08-03);
        inert when nothing is compiling, so the eager path is unchanged.
        """
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

    def _branch_input(self, branch: BaseHead, stemmed: Tensor, trunk: Tensor) -> Tensor:
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
        return self._gate_combine(outs, self.gate(self._gate_in(hidden_states)))

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
        return self._gate_combine_logits(outs, self.gate(self._gate_in(hidden_states)))

    # ── Arm-level Jacobian: measurement, then optionally the intervention ──

    def _arm_grads(self, hidden_states: Tensor, labels: Tensor, criterion=None):
        """``(z, losses, grads)``: each arm's solo CE and its gradient at the
        branch point where the arms diverge.

        The arms run on a DETACHED copy of their input, so ``autograd.grad``
        stops at that leaf and never enters the trunk - which is the whole
        reason this is affordable. Each backward crosses one small classifier.

        A ``reads_trunk`` arm (HALO) branches above the stem, so its gradient
        lands on the trunk tensor rather than the stem output. Both are the
        same shape (the stem is dim-preserving) and the trunk is the surface
        actually shared, so the rows stay comparable and belong to one Jacobian.
        """
        z = self._stem_out(hidden_states)
        z_d = z.detach().requires_grad_(True)
        trunk_d = hidden_states.detach().requires_grad_(True)
        losses, grads, index = [], [], []
        for i, b in enumerate(self.branches):
            inp = trunk_d if getattr(b, "reads_trunk", False) else z_d
            # Each arm's OWN objective, not an invented cross-entropy. For most
            # arms those coincide; for the HALO arm they do not, and using CE
            # there would add a second objective fighting its real one.
            loss = b.arm_loss(inp, labels, criterion)
            if loss is None:
                self._arm_gap = True
                continue
            (g,) = torch.autograd.grad(loss, inp, retain_graph=True, allow_unused=True)
            if g is None:
                continue
            losses.append(loss)
            grads.append(g)
            index.append(i)
        return z, losses, grads, index

    def arm_conflict(
        self, hidden_states: Tensor, labels: Tensor, criterion=None
    ) -> Dict[str, float]:
        """Pairwise cosines and relative magnitudes of the arms' solo gradients.

        The measurement ObjectiveConflict structurally cannot make. That one
        compares LOSS TERMS, and the arms are not loss terms - one
        cross-entropy reaches all of them through a mixture, weighted by
        posterior responsibility. This is the head's own Jacobian, one row per
        arm, and it is the gate on whether surgery is worth turning on.

        ``arm_grad_share_i`` rides alongside because a cosine is
        scale-invariant: two arms can read perfectly orthogonal while one of
        them is contributing nothing, and those are different diagnoses.
        """
        if self.gate is None or labels is None or not self.training:
            return {}
        # A forward under `torch.no_grad()` with `training=True` is a real
        # configuration, not a contradiction: lazy-module initialization does
        # exactly that (praxis/utils/system.py - `model.train()` then a
        # `no_grad` dummy pass). Nothing downstream will call backward, so
        # measuring is pointless, and a dummy all-ones batch would poison the
        # first reading anyway.
        if not torch.is_grad_enabled():
            return dict(self._arm_metrics)
        if torch.compiler.is_compiling():
            return dict(self._arm_metrics)
        step = getattr(self, "_arm_step", 0)
        self._arm_step = step + 1
        if step % ARM_CONFLICT_INTERVAL != 0:
            return dict(self._arm_metrics)
        self._arm_gap = False
        with torch.enable_grad():
            _, losses, grads, index = self._arm_grads(hidden_states, labels, criterion)
        if len(grads) < 2:
            return dict(self._arm_metrics)
        flat = [g.detach().float().flatten() for g in grads]
        norms = [float(g.norm()) for g in flat]
        ref = max(norms) or 1.0
        out: Dict[str, float] = {}
        worst = 1.0
        for a in range(len(flat)):
            i = index[a]
            out[f"arm_grad_share_{i}"] = norms[a] / ref
            out[f"arm_solo_loss_{i}"] = float(losses[a].detach())
            for b_ in range(a + 1, len(flat)):
                if norms[a] < 1e-12 or norms[b_] < 1e-12:
                    continue
                c = float(
                    (flat[a] @ flat[b_] / (norms[a] * norms[b_])).clamp(-1.0, 1.0)
                )
                out[f"arm_cos_{i}{index[b_]}"] = c
                worst = min(worst, c)
        out["arm_cos_min"] = worst
        out.update(self._override_shares(flat, norms, index))
        self._arm_metrics = out
        return out

    @staticmethod
    def _override_shares(flat, norms, index) -> Dict[str, float]:
        """How much of each arm's pull the combined update REVERSES.

        This is the question a magnitude ratio only gestures at. Under a
        sign-based optimizer (LionGeo's sign and spectral arms both discard
        magnitude; only its Frobenius arm does not) a larger row does not take
        larger steps. What it does is win the SIGN wherever rows disagree, so a
        row 20x its neighbour casts twenty votes to their one on contested
        coordinates and dictates the direction there.

        So: take the combined update the trunk will actually receive, and for
        each arm report the fraction of ITS OWN gradient mass sitting on
        coordinates where the combined sign is opposite to what that arm wanted.
        Mass-weighted rather than counted, because a flipped coordinate the arm
        barely cared about is not an override.

        0 = this arm is never contradicted. Toward 1 = the arm is being
        systematically overruled, and its own objective cannot act.

        Deliberately computed on the PLAIN SUM, not on the PCGrad result, so it
        measures the problem rather than the fix. Under prismatic9 compare it
        against ``arm_override_pcg_*`` for the same arms: the gap between them
        IS what the surgery bought, in the units that matter.
        """
        out: Dict[str, float] = {}
        if not flat:
            return out
        plain = torch.stack(flat).sum(0)
        unit = [g / g.norm().clamp_min(1e-12) for g in flat]
        for tag, combined in (
            ("", plain),
            ("pcg_", _pcgrad(flat)),
            ("eq_", _pcgrad(unit)),
        ):
            csign = torch.sign(combined)
            for a, g in enumerate(flat):
                if norms[a] < 1e-12:
                    continue
                mass = g.abs()
                flipped = (torch.sign(g) * csign) < 0
                out[f"arm_override_{tag}{index[a]}"] = float(
                    (mass[flipped].sum() / mass.sum()).item()
                )
        return out

    def arm_objectives(
        self, hidden_states: Tensor, labels: Tensor, criterion=None
    ) -> Dict[str, Tensor]:
        """Losses that replace the mixture as the arms' training signal.

        Empty unless the profile sets ``arm_surgery`` (prismatic9), so every
        other profile is untouched.

        Two kinds, and the split IS the design:

        * ``arm{i}_ce`` - arm i's own cross-entropy on the detached branch
          input. Trains arm i's parameters as a standalone classifier and
          reaches the trunk not at all. This is what lets the crystal arm train
          the way a bare crystal head does, which is the condition its PCA
          geometry was ever observed under.
        * ``arm_surgery`` - the surrogate ``(z * g_hat).sum()``, whose gradient
          with respect to ``z`` is exactly ``g_hat``, the PCGrad-combined
          per-arm gradient. This is the ONLY gradient the trunk receives from
          the arms, and it is the entire Jacobian intervention.

        The mixture cross-entropy remains the main loss and keeps training the
        gate, which is why SurgicalParallelHead detaches every arm in the
        blend: the blend is a judgement about finished predictions, not a
        training path.
        """
        if not self.arm_surgery or self.gate is None or labels is None:
            return {}
        if not self.training or torch.compiler.is_compiling():
            return {}
        # See arm_conflict: `training=True` under `no_grad` is the lazy-init
        # pass. Building an arm loss there yields a tensor with no grad_fn, and
        # handing that to autograd.grad raises "element 0 of tensors does not
        # require grad". No backward is coming, so there is nothing to arbitrate.
        if not torch.is_grad_enabled():
            return {}
        self._arm_gap = False
        z, losses, grads, index = self._arm_grads(hidden_states, labels, criterion)
        if not grads or self._arm_gap:
            # A row that reaches the shared representation but sits OUTSIDE the
            # arbitration is worse than no arbitration: it routes around the
            # very thing the surgery exists to do. Refuse rather than ship a
            # partial Jacobian. _arm_gap is set when an arm declined to give
            # its objective (e.g. HaloHead without HALOLoss as the criterion).
            return {}
        out: Dict[str, Tensor] = {
            f"arm{index[a]}_loss": l for a, l in enumerate(losses)
        }
        rows = [g.detach() for g in grads]
        if self.equalize_rows:
            scale = torch.stack(rows).sum(0).norm()
            rows = [g / g.norm().clamp_min(1e-12) for g in rows]
            combined = _pcgrad(rows)
            # Restore the step magnitude the trunk would have received, so this
            # changes the update's DIRECTION and not the effective learning rate.
            combined = combined * (scale / combined.norm().clamp_min(1e-12))
        else:
            combined = _pcgrad(rows)
        surrogate = (z * combined.to(z.dtype)).sum()
        # Value-neutral: subtracting its own detached value makes the term
        # exactly 0.0 in the reported loss while leaving its gradient
        # untouched. Without this the surrogate's arbitrary magnitude lands in
        # the loss curve, which is a number people read.
        out["arm_surgery"] = surrogate - surrogate.detach()
        self._arm_surgery_norm = float(combined.norm())
        return out

    def _arm_descriptions(self) -> dict:
        """Cards for the arm Jacobian. Built from the live arm count, since a
        profile's arm count is a property of the profile."""
        live = list(range(len(self.branches)))
        _G = "arm_jacobian"
        out: dict = {
            "arm_cos_min": {
                "description": (
                    "Minimum pairwise cosine between the arms' own-objective "
                    "gradients where they branch - the head's Jacobian. "
                    "Persistently negative is the case for PCGrad."
                ),
                "chart": {
                    "title": "Arm Gradient Conflict",
                    "y_label": "cosine between arms",
                    "y_scale": "linear",
                    "group": _G,
                    "group_order": 460,
                    "order": 10,
                    "series_group": "arm_cos",
                    "series_label": "worst pair",
                },
                "caller": type(self).__name__,
            },
            "arm_surgery_norm": {
                "description": (
                    "Norm of the combined gradient handed to the trunk "
                    "(prismatic9 only). Equal to the plain sum it replaces "
                    "means PCGrad found nothing to project."
                ),
                "chart": {
                    "title": "Surgical Trunk Gradient",
                    "y_label": "||g_hat||",
                    "y_scale": "logarithmic",
                    "group": _G,
                    "order": 40,
                },
                "caller": type(self).__name__,
            },
        }
        for pos, i in enumerate(live):
            out[f"arm_override_{i}"] = {
                "description": (
                    f"Fraction of arm {i}'s gradient MASS the plain sum "
                    "points the wrong way on. 0.5 is the null (no vote, not "
                    "opposition); 0 = never contradicted."
                ),
                "chart": {
                    "group": _G,
                    "order": 50 + pos,
                    "series_group": "arm_override",
                    "series_label": f"arm {i} (plain sum)",
                    "title": "Arm Override" if pos == 0 else None,
                    "y_label": "overruled gradient mass" if pos == 0 else None,
                },
                "caller": type(self).__name__,
            }
            out[f"arm_override_eq_{i}"] = {
                "description": (
                    f"Arm {i}'s overruled mass after PCGrad on ROW-EQUALIZED "
                    "gradients - what prismatic9 applies. Not GradNorm: no "
                    "alpha, and step size is preserved."
                ),
                "chart": {
                    "group": _G,
                    "order": 70 + pos,
                    "series_group": "arm_override",
                    "series_label": f"arm {i} (equalized)",
                },
                "caller": type(self).__name__,
            }
            out[f"arm_override_pcg_{i}"] = {
                "description": (
                    f"Arm {i}'s overruled mass after PCGrad. Equal to the "
                    "plain figure means nothing conflicted (its no-op); lower "
                    "means it stopped a louder row reversing this one."
                ),
                "chart": {
                    "group": _G,
                    "order": 60 + pos,
                    "series_group": "arm_override",
                    "series_label": f"arm {i} (PCGrad)",
                },
                "caller": type(self).__name__,
            }
            out[f"arm_grad_share_{i}"] = {
                "description": (
                    f"Arm {i}'s solo gradient norm relative to the largest "
                    "arm's - the half a cosine cannot supply. From the arm's "
                    "OWN objective, so it survives gate collapse."
                ),
                "chart": {
                    "group": _G,
                    "order": 20 + pos,
                    "series_group": "arm_share",
                    "series_label": f"arm {i}",
                    "title": "Arm Gradient Share" if pos == 0 else None,
                    "y_label": "||g_i|| / max ||g||" if pos == 0 else None,
                },
                "caller": type(self).__name__,
            }
            out[f"arm_solo_loss_{i}"] = {
                "description": (
                    f"Arm {i}'s OWN objective scored alone - CE for most "
                    "arms, HALO's geometry for that one, so read per-series. "
                    "Separates a bad arm from a starved one."
                ),
                "chart": {
                    "group": _G,
                    "order": 30 + pos,
                    "series_group": "arm_solo",
                    "series_label": f"arm {i}",
                    "title": "Arm Solo Objective" if pos == 0 else None,
                    "y_label": "loss (per-arm scale)" if pos == 0 else None,
                },
                "caller": type(self).__name__,
            }
            for j in live[pos + 1 :]:
                out[f"arm_cos_{i}{j}"] = {
                    "description": (
                        f"Cosine between arm {i}'s and arm {j}'s own-objective "
                        "gradients at the branch point. Negative means one is "
                        "cancelling the other."
                    ),
                    "chart": {
                        "group": _G,
                        "order": 11,
                        "series_group": "arm_cos",
                        "series_label": f"{i}-{j}",
                    },
                    "caller": type(self).__name__,
                }
        # Only the first series in a series_group carries title/axis; the rest
        # ride it. Drop the None placeholders rather than shipping them.
        for entry in out.values():
            entry["chart"] = {k: v for k, v in entry["chart"].items() if v is not None}
        return out

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
        if self._arm_metrics:
            out.update(self._arm_metrics)
        if self._arm_surgery_norm is not None:
            out["arm_surgery_norm"] = self._arm_surgery_norm
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
                out[f"p{i}_{k}"] = self._namespace_entry(
                    v, f"p{i}", f"#{i}", callers.get(k)
                )
        if self.stem is not None:
            callers = resolve_callers(self.stem)
            for k, v in self.stem.all_metric_descriptions().items():
                out[f"stem_{k}"] = self._namespace_entry(
                    v, "stem", "(shared)", callers.get(k)
                )
        out.update(self._gate_descriptions())
        out.update(self._arm_descriptions())
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
                    "Mean per-token softmax weight the gate gives this branch. Pinned "
                    "near 0 or 1 = the gate has specialized."
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
                "Entropy of the per-token branch gate (nats). High = the branches "
                "share the work; low = the gate commits to one."
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
                "Smallest gap between any two mean branch weights. Near 0 = two "
                "branches are equally important, which the gate repulsion pushes "
                "apart."
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


class SurgicalParallelHead(ParallelHead):
    """ParallelHead whose arms train on their own objectives, combined by PCGrad.

    Identical to ParallelHead in the forward pass and at inference: the same
    mixture of softmaxes over the same arms, so nothing about how predictions
    are made changes. What changes is training, and only training.

    The honest cost, stated up front: solo cross-entropy on every arm removes
    the DIVISION OF LABOUR. Under the mixture, arms specialize - each covers
    what it explains best and the gate routes accordingly. Trained alone, all
    arms learn the whole task and the head becomes an ensemble of near-
    redundant predictors rather than a set of complementary ones. That is the
    trade, it is deliberate, and val NLL is where it would show up.
    """

    arm_surgery = True

    # Give every objective an EQUAL VOTE in the DIRECTION of the trunk update.
    # Normalize each Jacobian row to unit norm before combining, then rescale
    # the result to the norm the plain sum would have had: nothing to tune, and
    # the effective step size is unchanged.
    #
    # Not GradNorm, which equalizes training RATES via learned loss weights and
    # an `alpha` restoring exponent - a hyperparameter, and an assumption (that
    # all objectives converge together) that is wrong for a geometric constraint
    # already at its target.
    #
    # PCGrad alone is not enough: it acts only where rows CONFLICT, and when a
    # row is 20-65x its neighbours and merely ORTHOGONAL to them there is
    # nothing to project - the small rows are drowned, not opposed.
    # `arm_override_*` on a smoke model read 0.465 / 0.506 / 0.000 before and
    # 0.467 / 0.493 / 0.001 after PCGrad. And 0.5 IS the null for an orthogonal
    # row against a dominant one, so those numbers say the crystal arm has no
    # vote, not that it is being fought.
    #
    # Consistent with the rest of the system: LionGeo's sign and spectral arms
    # discard gradient magnitude anyway, and research/body.tex argues magnitude
    # is the wrong readout for significance (a boundary flip is silent in norm).
    equalize_rows = True
    equalize_rows = True

    # The gate reads a DETACHED trunk. Its cross-entropy would otherwise be a
    # fourth gradient into the trunk that never passed through the surgery -
    # the same defect as an excluded arm, just quieter. The gate's job is to
    # judge finished predictions, which needs the trunk as INPUT and not as
    # something it gets to reshape.
    detach_gate_input = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Detach every arm in the blend so the mixture CE trains the GATE and
        # nothing else. Without this each arm would receive BOTH its solo
        # gradient and the mixture's responsibility-weighted one, and the
        # starvation this head exists to remove would come straight back in
        # through the second path.
        for b in self.branches:
            b.detach_in_blend = True

    def compose_repr(self) -> str:
        return "Surgical" + super().compose_repr()
