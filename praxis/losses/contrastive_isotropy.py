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
                "Mean off-diagonal cosine of last-layer token reps. High = "
                "collapsed/anisotropic; should fall as the loss decorrelates "
                "reps - the signal that the representation geometry is improving."
            ),
            "chart": {
                "title": "Representation Anisotropy",
                "y_label": "Mean Cosine",
                "y_scale": "linear",
                "group": "contrastive_isotropy",
                "order": 20,
            },
        },
    }

    def __init__(self, pad_id: int = 0, margin: float = RHO):
        super().__init__()
        self.pad_id = pad_id
        self.margin = margin
        self._metrics: dict = {}

    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        # hidden_states: [B, T, D] last-layer reps. Cost is O(T^2 * D) per
        # sequence; fine at experiment scale, revisit with chunking for long T.
        h = F.normalize(hidden_states, dim=-1)
        sims = torch.matmul(h, h.transpose(1, 2))  # [B, T, T] cosine, diagonal = 1

        B, T, _ = sims.shape
        valid = ~torch.eye(T, device=sims.device, dtype=torch.bool).unsqueeze(0)

        # Mask padded positions when input_ids align to the rep length (skips
        # the encoder/patch case, where token-level ids don't correspond 1:1).
        if input_ids is not None and input_ids.size(1) == T:
            keep = input_ids != self.pad_id  # [B, T]
            valid = valid & (keep.unsqueeze(1) & keep.unsqueeze(2))

        denom = valid.sum().clamp_min(1)
        loss = (F.relu(sims - (1.0 - self.margin)) * valid).sum() / denom

        # Own our diagnostics here, beside the math that produces them.
        # repr_anisotropy = mean off-diagonal cosine; high = collapsed space.
        with torch.no_grad():
            self._metrics = {
                "contrastive_loss": float(loss.detach()),
                "repr_anisotropy": float((sims * valid).sum() / denom),
            }
        return loss

    def training_metrics(self) -> dict:
        """Scalars from the last forward, surfaced to the metrics logger."""
        return dict(self._metrics)
