import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, penalty_weight=0, *args, **kwargs):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        ce_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=-100
        )
        if self.penalty_weight == 0:
            return ce_loss.mean()
        token_output = torch.argmax(shift_logits, dim=1)
        duplicated_masks = (
            torch.eq(input_ids.view(-1), token_output.unsqueeze(-1)).any(dim=-1).float()
        )
        loss = ce_loss * (1 + duplicated_masks * self.penalty_weight)
        return loss.mean()
