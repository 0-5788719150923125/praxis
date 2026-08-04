"""Forward-path preference policy over chosen/rejected-tagged tokens.

The hh-rlhf card's contract: the pairs are preference-modeling data, not SFT
material. DPO's core insight makes the simplest compliant objective possible
with no reward model and no sampling - the policy IS the reward model, and a
reference-free (SimPO-style) margin needs only the model's own likelihoods:
push the mean per-token log-probability of chosen text above that of rejected
text through a logistic margin.

DISABLED BY DEFAULT, AND THE CONTRAST IS NOT PAIRED. No experiment lists this in
``rl_type``; see next/rl.md. The reason is not that packing loses the row
alignment - it is that ``format_preference_pair`` emits exactly ONE side per
call, chosen 50/50 at random (praxis/data/formatters/conversation.py). A pair's
two halves are therefore never co-resident by construction, so the margin below
contrasts one random hh-rlhf conversation's chosen text against a DIFFERENT
random conversation's rejected text. What it measures is largely the difficulty,
length and domain gap between two unrelated documents, not a preference. Live
metrics agree it is not working: preference_rejected_logp rose over the
abstractinator-g run (rejected text becoming MORE likely) while the margin
shrank.

Restoring a real pairwise objective needs a formatter change, not a policy
change: emit both sides with a shared pair id and thread that id alongside the
task tags, then contrast within a pair id. Specified in next/rl.md.

The margin contrasts the two tag POPULATIONS (``PREF_CHOSEN`` /
``PREF_REJECTED``) within a batch - the chunk-level analogue of the pairwise
loss, honestly an unpaired approximation. The overall objective is ORPO-shaped:
chosen text keeps flowing through the main CE (the SFT anchor), rejected text is
excluded from the main CE entirely (``_build_loss_weights``) and appears only
here, being pushed down relative to chosen.

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
    # PREF_CHOSEN / PREF_REJECTED come only from hh-rlhf
    # (DataFormat.PREFERENCE_PAIR), so the margin has nothing to score without
    # this collection. See ChatFormat-independent tagging in
    # praxis/data/formatters/conversation.py::format_preference_pair.
    dataset_collections = ("preference",)
    # Margin sharpness (SimPO's beta). Fixed, model-agnostic: 2.0 sits in the
    # paper's stable range and the loss is scale-bounded by logsigmoid anyway.
    BETA = 2.0
    # Minimum tokens on EACH side before a margin is computed. A mean per-token
    # log-prob over a handful of bytes is not a statistic - the live run reached
    # populations as small as ONE byte, and every such batch injected pure noise
    # into the objective at full weight. Skipping is the correct behaviour: the
    # chosen side still trains through the main CE either way.
    MIN_SIDE_TOKENS = 32

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
        # The margin needs both populations, each large enough to mean anything
        # (see MIN_SIDE_TOKENS); a batch that fails either test contributes
        # nothing, and its chosen text still trains via the main CE.
        if n_chosen < self.MIN_SIDE_TOKENS or n_rejected < self.MIN_SIDE_TOKENS:
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

        # Scale by this policy's share of the batch. The two side likelihoods
        # are each normalised by their OWN token count, while the main CE
        # normalises over every supervised position (praxis/losses/reduction.py),
        # so an unscaled term applied `rl_weight` of force to a population of a
        # few dozen bytes - measured at ~2.7x the main CE's per-token gradient on
        # the tokens it touched. Weighting by (n_chosen + n_rejected)/n_total
        # makes the term contribute in proportion to how much of the batch is
        # actually preference data, which is what rl_weight reads as.
        n_total = int((shift_labels != IGNORE_INDEX).sum())
        share = (n_chosen + n_rejected) / max(n_total, 1)

        loss = self.rl_weight * share * -F.logsigmoid(self.BETA * margin)

        p = self.prefix
        self._metrics = {
            f"{p}_margin": float(margin.detach()),
            f"{p}_share": share,
            f"{p}_chosen_logp": float(lp_chosen.detach()),
            f"{p}_rejected_logp": float(lp_rejected.detach()),
            f"{p}_chosen_tokens": float(n_chosen),
            f"{p}_rejected_tokens": float(n_rejected),
        }
        return loss, self._metrics

    def get_metrics(self) -> dict:
        return dict(self._metrics)
