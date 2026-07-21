"""Forward-path preference policy over chosen/rejected-tagged tokens.

The hh-rlhf card's contract: the pairs are preference-modeling data, not SFT
material. DPO's core insight makes the simplest compliant objective possible
with no reward model and no sampling - the policy IS the reward model, and a
reference-free (SimPO-style) margin needs only the model's own likelihoods:
push the mean per-token log-probability of chosen text above that of rejected
text through a logistic margin.

Byte-level block packing chunks documents arbitrarily, so true per-pair row
alignment does not survive the pipeline. The pairing is instead carried by
per-token task tags (``PREF_CHOSEN`` / ``PREF_REJECTED``, emitted by
``format_preference_pair``), and the margin contrasts the two tag POPULATIONS
within a batch - the chunk-level analogue of the pairwise loss, honestly an
unpaired approximation. The overall objective is ORPO-shaped: chosen text
keeps flowing through the main CE (the SFT anchor), rejected text is excluded
from the main CE entirely (``_build_loss_weights``) and appears only here,
being pushed down relative to chosen.

Recall-family policy (like engagement/joke): any number coexist, partitioned
by task tags, invoked with ``(logits, labels, assistant_mask, task_type_ids)``
on the ordinary training forward. No extra parameters, no rollouts, no
reference model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.tasks import TaskType

IGNORE_INDEX = -100


class PreferencePolicy(nn.Module):
    is_weight_controller = False
    needs_rl_datasets = False
    is_recall = True
    prefix = "preference"
    # Margin sharpness (SimPO's beta). Fixed, model-agnostic: 2.0 sits in the
    # paper's stable range and the loss is scale-bounded by logsigmoid anyway.
    BETA = 2.0

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rl_weight = getattr(config, "rl_weight", 0.1)
        self._metrics: dict = {}

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        assistant_mask: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        # Needs full per-token logits and task tags; degrade to a no-op rather
        # than guess if either is missing (cut-CE skips logits, eval mode, or
        # a batch with no preference-tagged rows).
        if (
            logits is None
            or logits.dim() != 3
            or task_type_ids is None
            or not self.training
        ):
            return None, {}

        # Alignment follows _compute_loss/_build_loss_weights exactly: labels
        # arrive PRE-SHIFTED (input_ids[..., 1:]) except for aligned encoders
        # (full-length), so logits pair with labels position-for-position after
        # truncating logits to the label length; the full-length task/assistant
        # masks align to labels via their trailing target_len positions.
        # Byte-latent repadding can leave off-by-one length gaps - align to
        # the common length like the other recall policies do.
        target_len = labels.size(-1)
        seq = min(logits.size(1), target_len)
        if seq <= 0:
            return None, {}
        shift_logits = logits[:, :seq]
        shift_labels = labels[:, :seq]
        shift_task = task_type_ids[..., -target_len:][:, :seq].to(shift_logits.device)

        valid = shift_labels != IGNORE_INDEX
        if assistant_mask is not None:
            mask = assistant_mask[..., -target_len:].to(shift_logits.device).bool()
            if mask.size(1) >= seq:
                valid = valid & mask[:, :seq]

        chosen = valid & (shift_task == int(TaskType.PREF_CHOSEN))
        rejected = valid & (shift_task == int(TaskType.PREF_REJECTED))
        n_chosen = int(chosen.sum())
        n_rejected = int(rejected.sum())
        # The margin needs both populations; a batch carrying only one side
        # contributes nothing (its chosen text still trains via the main CE).
        if n_chosen == 0 or n_rejected == 0:
            return None, {}

        safe_labels = shift_labels.clamp(min=0)
        logprob = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        # Length-normalized (mean per-token) side likelihoods - SimPO's
        # normalization, applied to the tag populations.
        lp_chosen = (logprob * chosen.float()).sum() / n_chosen
        lp_rejected = (logprob * rejected.float()).sum() / n_rejected
        margin = lp_chosen - lp_rejected

        loss = self.rl_weight * -F.logsigmoid(self.BETA * margin)

        p = self.prefix
        self._metrics = {
            f"{p}_margin": float(margin.detach()),
            f"{p}_chosen_logp": float(lp_chosen.detach()),
            f"{p}_rejected_logp": float(lp_rejected.detach()),
            f"{p}_chosen_tokens": float(n_chosen),
            f"{p}_rejected_tokens": float(n_rejected),
        }
        return loss, self._metrics

    def get_metrics(self) -> dict:
        return dict(self._metrics)
