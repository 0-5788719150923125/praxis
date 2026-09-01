import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.regularizer_base import BaseRegularizer

# SimCTG margin: penalize cosine similarity above (1 - RHO) between distinct
# tokens. Fixed, model-agnostic constant from arxiv 2202.06417 - not a knob to
# tune per run.
RHO = 0.5


class ContrastiveIsotropyLoss(BaseRegularizer):
    """SimCTG isotropy regularizer. Pushes apart the representations of distinct
    tokens within a sequence so the space stays discriminative - the geometry
    contrastive-search decoding relies on. Additive to the main objective; it
    does not replace the LM loss.

    From "A Contrastive Framework for Neural Text Generation" (arxiv 2202.06417).
    """

    name = "contrastive"

    # Chart hints for the values training_metrics() produces, kept beside
    # them so both edit in one place. Surfaced to the Dynamics tab manifest.
    metric_descriptions = {
        "contrastive_loss": {
            "description": (
                "SimCTG isotropy regularizer on token representations. "
                "Additive auxiliary loss; the main objective is untouched."
            ),
            "chart": {
                "title": "Contrastive Isotropy Loss",
                "y_label": "Loss",
                "y_scale": "linear",
                "group": "contrastive_isotropy",
                "group_order": 90,
                "order": 10,
            },
        },
        "repr_anisotropy": {
            "description": (
                "Mean off-diagonal cosine of the reps the loss acts on - the "
                "quantity SimCTG's margin is written against, so it is charted "
                "as the loss's own gauge. READ IT WITH repr_dimensions: it is a "
                "rank-1 statistic, equal (up to an affine map) to the squared "
                "norm of the mean direction, so it is dominated by a shared "
                "offset and is NOT a measure of collapse. A cloud that is only "
                "TRANSLATED off the origin reads ~0.98 here while occupying its "
                "space fully, and a genuine rank-3 collapse in 111 dimensions "
                "reads ~0.00. High alone means 'the tokens share a direction', "
                "which may be benign."
            ),
            "chart": {
                "title": "Representation Anisotropy",
                "y_label": "Mean Cosine",
                "y_scale": "linear",
                "group": "contrastive_isotropy",
                "order": 20,
            },
        },
        "repr_dimensions": {
            "description": (
                "Participation ratio of the MEAN-CENTERED rep covariance, as a "
                "fraction of the isotropic null (the null is below D at finite "
                "T, so the ratio is normalized to sit at 1.0 for a spread cloud "
                "at any T and D). This is the collapse detector repr_anisotropy "
                "cannot be: 1.0 = variance spread over every direction, small = "
                "the reps live in a low-dimensional subspace. Falling here while "
                "repr_anisotropy is flat is real degeneration; the reverse is a "
                "shared offset and costs nothing."
            ),
            "chart": {
                "title": "Representation Dimensions Used",
                "y_label": "Participation / Isotropic Null",
                "y_scale": "linear",
                "group": "contrastive_isotropy",
                "order": 30,
            },
        },
        "repr_nematic": {
            "description": (
                "Axis alignment of the centered rep directions: 0 = no preferred "
                "axis, 1 = every rep on one line. Sign-blind and debiased, so it "
                "is exactly 0 in expectation for an isotropic cloud at any T and "
                "D. Sees the structure the mean cosine is blind to - two "
                "clusters pointing OPPOSITE ways read ~0.99 here and ~0.00 as "
                "mean cosine, because their mean directions cancel."
            ),
            "chart": {
                "title": "Representation Axis Order",
                "y_label": "Nematic Order",
                "y_scale": "linear",
                "group": "contrastive_isotropy",
                "order": 40,
            },
        },
        "contrastive_active_frac": {
            "description": (
                "Fraction of token pairs above the margin, i.e. the share of the "
                "Gram matrix the hinge is actually penalizing. At 1.0 relu() has "
                "stopped being a hinge: every pair is clipped in, the term is a "
                "plain linear penalty at constant maximum gradient, and the loss "
                "has no headroom left to push harder with. Sustained 1.0 while "
                "the geometry does not move means this regularizer is at full "
                "authority and losing."
            ),
            "chart": {
                "title": "Contrastive Hinge Active Fraction",
                "y_label": "Fraction of Pairs",
                "y_scale": "linear",
                "group": "contrastive_isotropy",
                "order": 50,
            },
        },
    }

    def __init__(
        self, pad_id: int = 0, margin: float = RHO, observe_only: bool = False
    ):
        super().__init__()
        self.pad_id = pad_id
        self.margin = margin
        # Diagnostic-only mode: compute every metric, contribute NO gradient.
        # The geometry readings this class owns - repr_anisotropy (the squared
        # magnetization), repr_nematic (the second moment), repr_dimensions -
        # are the instruments for representation collapse, and they lived only
        # on the path that also applied the loss. Removing the term to stop
        # demagnetizing therefore removed the only way to see whether that
        # helped: abstractinator-f dropped the loss and lost the evidence in the
        # same stroke. Measurement and force are now separable.
        self.observe_only = observe_only
        self._metrics: dict = {}

    def forward(self, hidden_states: Tensor, input_ids: Tensor, **_) -> Tensor:
        # hidden_states: [B, T, D] last-layer reps. Cost is O(T^2 * D) per
        # sequence; fine at experiment scale, revisit with chunking for long T.
        if self.observe_only:
            with torch.no_grad():
                self._forward_impl(hidden_states, input_ids)
            # An exact zero with no graph: added to the loss it is a no-op, and
            # nothing downstream has to know this regularizer is only watching.
            return hidden_states.new_zeros(())
        return self._forward_impl(hidden_states, input_ids)

    def _forward_impl(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        h = F.normalize(hidden_states, dim=-1)
        sims = torch.matmul(h, h.transpose(1, 2))  # [B, T, T] cosine, diagonal = 1

        B, T, _ = sims.shape
        # Expanded to the batch so `denom` counts every pair the numerator sums.
        # Left at [1, T, T] the mean divides B batches' worth of similarities by
        # one batch's worth of pairs, and both the loss and the metric come out
        # B times too large - but only on the path where the mask below does not
        # broadcast it, so the same run's scale would jump between the two.
        valid = (~torch.eye(T, device=sims.device, dtype=torch.bool)).expand(B, T, T)

        # Mask padded positions when input_ids align to the rep length (skips
        # the encoder/patch case, where token-level ids don't correspond 1:1).
        if input_ids is not None and input_ids.size(1) == T:
            keep = input_ids != self.pad_id  # [B, T]
            valid = valid & (keep.unsqueeze(1) & keep.unsqueeze(2))

        denom = valid.sum().clamp_min(1)
        loss = (F.relu(sims - (1.0 - self.margin)) * valid).sum() / denom

        # Own our diagnostics here, beside the math that produces them. The
        # margin gauge (mean cosine, hinge occupancy) comes off the Gram we
        # already built; the geometry readings need the second moment, which the
        # Gram cannot supply - see _spectral_metrics.
        with torch.no_grad():
            above = ((sims > (1.0 - self.margin)) & valid).sum() / denom
            self._metrics = {
                "contrastive_loss": float(loss.detach()),
                "repr_anisotropy": float((sims * valid).sum() / denom),
                "contrastive_active_frac": float(above),
                **self._spectral_metrics(hidden_states),
            }
        return loss

    @staticmethod
    def _spectral_metrics(hidden_states: Tensor) -> dict:
        """Second-moment geometry, which the pairwise cosine structurally cannot
        reach. Mean cosine is rank-1: it equals the squared norm of the mean
        direction up to an affine map, so it reports a shared OFFSET, and both of
        its failures are one-sided. A cloud that is merely translated off the
        origin reads ~0.98 while filling its space; a real rank-3 collapse in 111
        dimensions reads ~0.00; two clusters facing opposite ways read ~0.00
        because their means cancel - and that last case is the whole point,
        since locally-aligned-but-differently-oriented regions are exactly the
        structure worth telling apart from uniform collapse.

        Both readings are computed from traces rather than an eigendecomposition
        (O(T * D^2), no eigh) so they stay cheap and shape-stable under a varying
        sequence length, and both are normalized against their isotropic null so
        a run is comparable to itself as T and D change:

            repr_dimensions  (tr C)^2 / ||C||_F^2 for the centered covariance C,
                             over the null D / (1 + D/T). 1.0 = spread.
            repr_nematic     sqrt( (D/(D-1)) * (||M||_F^2 - 1/D) - 1/T ) for
                             M = mean(u u^T) over centered unit directions u.
                             The 1/T is exact, not fitted: under isotropy
                             E[||M||_F^2] = 1/T + (T-1)/(T*D), so the whole
                             expression has expectation zero at any T and D.
        """
        x = hidden_states.detach().float().flatten(0, -2)  # [B * T, D]
        n, d = x.shape
        if n < 2 or d < 2:
            return {"repr_dimensions": 0.0, "repr_nematic": 0.0}

        centered = x - x.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / n  # [D, D]
        participation = cov.trace().square() / cov.square().sum().clamp_min(1e-12)
        # The isotropic null sits below d at finite n (a Marchenko-Pastur edge,
        # not a defect), so divide it out and the healthy reading is 1.0.
        dimensions = participation * (1.0 + d / n) / d

        u = F.normalize(centered, dim=-1)
        second = u.T @ u / n  # [D, D]
        nematic_sq = (d / (d - 1)) * (second.square().sum() - 1.0 / d) - 1.0 / n
        return {
            "repr_dimensions": float(dimensions),
            "repr_nematic": float(nematic_sq.clamp_min(0).sqrt()),
        }

    def training_metrics(self) -> dict:
        """Scalars from the last forward, surfaced to the metrics logger."""
        return dict(self._metrics)
