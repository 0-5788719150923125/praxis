"""Forward-path preference policy over paired chosen/rejected responses.

The hh-rlhf pairs are preference-modeling data, not SFT material. DPO's core
insight makes the simplest compliant objective possible with no reward model and
no sampling - the policy IS the reward model, and a reference-free (SimPO-style)
margin needs only the model's own likelihoods: push the mean per-token
log-probability of a chosen response above that of the rejected response to the
SAME prompt, by a target margin.

What makes the comparison real is the pairing, and it is built in the data
pipeline rather than here. ``format_preference_pair`` emits both sides of a
pair, truncated at the turn where they diverge and sharing a character-identical
prompt; the manager enqueues them adjacently under one id; the packer carries
that id per token on the divergent response only. This policy buckets by that
id, so it compares two answers to one question rather than two unrelated
documents, and it never scores the prompt the two sides have in common.

The overall objective is ORPO-shaped: chosen text keeps flowing through the main
CE (the SFT anchor), rejected text is excluded from the main CE entirely
(``_build_loss_weights``) and appears only here, pushed down relative to chosen.

Recall-family policy (like engagement/joke): any number coexist, partitioned by
task tags, invoked on the ordinary training forward. No extra parameters, no
rollouts, no reference model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.tasks import PAIR_ID_SLOTS, TaskType

IGNORE_INDEX = -100


class PreferencePolicy(nn.Module):
    is_weight_controller = False
    needs_rl_datasets = False
    is_recall = True
    prefix = "preference"
    # PREF_CHOSEN / PREF_REJECTED come only from hh-rlhf
    # (DataFormat.PREFERENCE_PAIR), so the margin has nothing to score without
    # it - and the dataset card permits nothing else, so this policy is its
    # only legitimate consumer. Declared dataset-level rather than as a
    # collection so the pairing is owned here: `rl_type: preference` is both
    # necessary and sufficient to get the data. See ChatFormat-independent
    # tagging in praxis/data/formatters/conversation.py::format_preference_pair.
    dataset_weights = {"hh-rlhf": 1.0}
    # Margin sharpness and SimPO's target reward margin. Fixed and
    # model-agnostic: both sit in the paper's stable range (gamma/beta = 0.5),
    # and the loss is scale-bounded by logsigmoid anyway.
    BETA = 2.0
    GAMMA = 1.0
    # Minimum tokens on EACH side of a pair before it is scored. A mean
    # per-token log-prob over a couple of tokens is not a statistic. This is a
    # floor, not a filter: the divergent responses it scores have a median of
    # ~20 tokens on their shorter side, so a threshold set for whole
    # transcripts would silently discard most of the dataset (32 drops 71% of
    # pairs; 4 drops 2.4%).
    MIN_SIDE_TOKENS = 4

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
        pair_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        # Needs full per-token logits, task tags and the pairing; degrade to a
        # no-op rather than guess if any is missing (cut-CE skips logits, eval
        # mode, a batch with no preference rows, or a hand-built batch that
        # carries no pair ids).
        if (
            logits is None
            or logits.dim() != 3
            or task_type_ids is None
            or pair_ids is None
            or not self.training
        ):
            return None, {}

        # Alignment follows _compute_loss/_build_loss_weights exactly: labels
        # arrive PRE-SHIFTED (input_ids[..., 1:]) except for aligned encoders
        # (full-length), so logits pair with labels position-for-position after
        # truncating logits to the label length; the full-length task/assistant/
        # pair channels align to labels via their trailing target_len positions.
        # Byte-latent repadding can leave off-by-one length gaps - align to
        # the common length like the other recall policies do.
        target_len = labels.size(-1)
        seq = min(logits.size(1), target_len)
        if seq <= 0:
            return None, {}
        shift_logits = logits[:, :seq]
        shift_labels = labels[:, :seq]
        shift_task = task_type_ids[..., -target_len:][:, :seq].to(shift_logits.device)
        shift_pair = pair_ids[..., -target_len:][:, :seq].to(shift_logits.device).long()

        supervised = shift_labels != IGNORE_INDEX
        valid = supervised
        if assistant_mask is not None:
            mask = assistant_mask[..., -target_len:].to(shift_logits.device).bool()
            if mask.size(1) >= seq:
                valid = valid & mask[:, :seq]
        # An id outside the pool cannot be bucketed, so it is not scored. The
        # packer never emits one; a hand-built batch might.
        valid = valid & (shift_pair > 0) & (shift_pair < PAIR_ID_SLOTS)

        # float32 throughout the reduction below: under bf16 a count stops
        # being exact at 256, and these run to the hundreds.
        chosen = (valid & (shift_task == int(TaskType.PREF_CHOSEN))).float()
        rejected = (valid & (shift_task == int(TaskType.PREF_REJECTED))).float()

        safe_labels = shift_labels.clamp(min=0)
        logprob = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        # Per-pair sums and counts, gathered by a fixed-size scatter. Bucketing
        # by torch.unique would give a data-dependent shape; the pool is small
        # enough that a dense one costs nothing and stays static.
        index = shift_pair.reshape(-1)
        flat_logprob = logprob.reshape(-1).float()

        def gather(side: torch.Tensor):
            weight = side.reshape(-1)
            count = torch.zeros(PAIR_ID_SLOTS, device=index.device, dtype=torch.float32)
            total = torch.zeros_like(count)
            count.index_add_(0, index, weight)
            total.index_add_(0, index, flat_logprob * weight)
            return count, total

        count_chosen, sum_chosen = gather(chosen)
        count_rejected, sum_rejected = gather(rejected)

        # A pair is scored only with both sides present and each side above the
        # floor. The half that is missing is usually a pair the packer split
        # across two get_batch calls; its partner simply does not contribute.
        scored = (count_chosen >= self.MIN_SIDE_TOKENS) & (
            count_rejected >= self.MIN_SIDE_TOKENS
        )
        scored[0] = False  # bucket 0 is "takes part in no comparison"
        weight = scored.float()
        num_pairs = weight.sum()
        if int(num_pairs) == 0:
            return None, {}

        # Length-normalized (mean per-token) response likelihoods - SimPO's
        # normalization, per pair rather than pooled over the batch, so every
        # comparison counts once instead of in proportion to its length.
        logp_chosen = sum_chosen / count_chosen.clamp(min=1)
        logp_rejected = sum_rejected / count_rejected.clamp(min=1)
        margin = logp_chosen - logp_rejected
        per_pair = -F.logsigmoid(self.BETA * margin - self.GAMMA)

        def mean_over_pairs(values: torch.Tensor) -> torch.Tensor:
            return (values * weight).sum() / num_pairs

        # Scale by this policy's share of the batch. The side likelihoods are
        # each normalised by their OWN token count, while the main CE normalises
        # over every supervised position (praxis/losses/reduction.py), so an
        # unscaled term applied `rl_weight` of force to a population of a few
        # dozen tokens - measured at ~2.7x the main CE's per-token gradient on
        # the tokens it touched. Weighting by the scored share makes the
        # per-token pull a fixed multiple of rl_weight no matter how short the
        # responses are, which is what rl_weight reads as.
        scored_tokens = ((count_chosen + count_rejected) * weight).sum()
        num_supervised = supervised.sum().clamp(min=1)
        share = scored_tokens / num_supervised

        loss = self.rl_weight * share * mean_over_pairs(per_pair)

        p = self.prefix
        self._metrics = {
            f"{p}_margin": float(mean_over_pairs(margin).detach()),
            # The honest readout: how often the model already ranks the pair
            # correctly. A margin can drift for reasons that have nothing to do
            # with preference; this cannot.
            f"{p}_accuracy": float(
                mean_over_pairs((margin > 0).to(margin.dtype)).detach()
            ),
            f"{p}_pairs": float(num_pairs.detach()),
            f"{p}_share": float(share.detach()),
            f"{p}_chosen_logp": float(mean_over_pairs(logp_chosen).detach()),
            f"{p}_rejected_logp": float(mean_over_pairs(logp_rejected).detach()),
            f"{p}_chosen_tokens": float((count_chosen * weight).sum().detach()),
            f"{p}_rejected_tokens": float((count_rejected * weight).sum().detach()),
        }
        return loss, self._metrics

    def get_metrics(self) -> dict:
        return dict(self._metrics)
