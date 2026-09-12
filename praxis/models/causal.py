"""A pretrained causal LM trained under Praxis objectives."""

from typing import Dict, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from praxis.containers import LossContainer
from praxis.models.base import ForeignModel
from praxis.objectives import CausalObjectiveMixin


class ForeignCausalLM(ForeignModel, CausalObjectiveMixin):
    """``AutoModelForCausalLM`` output, wired to the Praxis objective half.

    The hosted model is asked for hidden states and logits and NOTHING ELSE -
    in particular never for its loss. Praxis pre-shifts labels
    (``input_ids[..., 1:]``) and trims the logits in ``_compute_loss``, while
    every HF model shifts internally, so handing ``labels`` down would shift
    twice. Keeping the criterion on this side is also the point: it is what
    makes ``--loss-func``, ``--regularizers``, ``--task-weights`` and the
    forward-path RL policies apply to a model Praxis did not build.
    """

    TASK = "causal_lm"

    # Praxis mechanisms that need a Praxis-built module to attach to. Each
    # is a hard error rather than a silent no-op; the reason is the message.
    UNSUPPORTED: Dict[str, str] = {
        "mtp_type": (
            "multi-token prediction drafts through the model's own classifier, "
            "and a foreign model exposes a plain lm_head with no draft path"
        ),
        "bidirectional": (
            "bidirectional training needs a second, Praxis-built classifier "
            "over the same trunk"
        ),
    }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rewards: Optional[torch.FloatTensor] = None,
        token_weights: Optional[torch.FloatTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        assistant_mask: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.LongTensor] = None,
        pair_ids: Optional[torch.LongTensor] = None,
        row_continues: Optional[torch.Tensor] = None,
        current_state: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Regularizers that collect state during the forward drop whatever the
        # last one left behind, before anything runs.
        self.criterion.reset()

        # `block_ids` and `row_continues` describe how the packer laid the batch
        # out, and only Praxis attention reads them. A foreign model attends
        # across document boundaries within a packed row - which is what it was
        # pretrained to do - so they are accepted and dropped here rather than
        # passed down into kwargs that transformers would silently ignore.
        base_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache if use_cache is not None else not self.training,
            # The last entry is the post-final-norm activation the model's own
            # lm_head consumes - verified identical to recomputing it - so
            # cut-CE and the representation regularizers see exactly the tensor
            # they see on a Praxis model. Costs nothing in training, where
            # autograd retains those activations anyway.
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = base_outputs.hidden_states[-1]
        logits = base_outputs.logits

        losses = LossContainer()
        loss = self.supervise(
            losses,
            logits,
            hidden_states,
            self.scorer,
            input_ids,
            labels,
            attention_mask=attention_mask,
            rewards=rewards,
            token_weights=token_weights,
            task_type_ids=task_type_ids,
            assistant_mask=assistant_mask,
            pair_ids=pair_ids,
        )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
            hidden_states=base_outputs.hidden_states if output_hidden_states else None,
            attentions=base_outputs.attentions,
        )
