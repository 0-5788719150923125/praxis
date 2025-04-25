from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContrastiveTokenLoss(nn.Module):
    """A Pytorch Module wrapper for the contrastive_token_loss function.
    https://arxiv.org/abs/2205.02517
    https://github.com/ShaojieJiang/CT-Loss/tree/main

    Args:
        ignore_index (int, optional): Default padding token id. Defaults to -100.
        pad_id (int, optional): Specified padding token id. Used to mask out irrelevant preceding tokens. Defaults to 0.
        ct_length (Union[int, float], optional): When it's a float value and in [0, 1], it's a portion to the original sequence length;
        when it's larger than 1, it specifies the absolute CT length. Defaults to 0.25.
        preced_m_negatives (Union[int, float], optional): When it's a float value and in [0, 1], it's a portion to the CT sequence length;
        when it's larger than 1, it specifies the absolute negative window size. Defaults to 0.5.

    Returns:
        Tensor: Calculated CT loss.
    """

    def __init__(
        self,
        ignore_index=-100,
        pad_id=0,
        ct_length=0.25,
        preced_m_negatives=0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.pad_id = pad_id
        self.ct_length = ct_length
        self.preced_m_negatives = preced_m_negatives

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *args,
        **kwargs,
    ):
        shift_logits_flat = logits.view(-1, logits.shape[-1])
        shift_labels_flat = labels.view(-1)
        ce_loss = F.cross_entropy(
            shift_logits_flat, shift_labels_flat, reduction="mean", ignore_index=-100
        )
        ct_loss = contrastive_token_loss(
            logits,
            labels,
            self.ignore_index,
            self.pad_id,
            self.ct_length,
            self.preced_m_negatives,
        )
        return ce_loss + ct_loss


def contrastive_token_loss(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -100,
    pad_id: int = 0,
    ct_length: Union[int, float] = 0.25,
    preced_m_negatives: Union[int, float] = 0.5,
    # negative_token_portion: float = 0.125,
    # infer_length: bool = True,
) -> Tensor:
    """Contrastive Token loss function

    Args:
        input (Tensor): Input logits
        target (Tensor): Target token indices
        ignore_index (int, optional): Default padding token id. Defaults to -100.
        pad_id (int, optional): Specified padding token id. Used to mask out irrelevant preceding tokens. Defaults to 0.
        ct_length (Union[int, float], optional): When it's a float value and in [0, 1], it's a portion to the original sequence length;
        when it's larger than 1, it specifies the absolute CT length. Defaults to 0.25.
        preced_m_negatives (Union[int, float], optional): When it's a float value and in [0, 1], it's a portion to the CT sequence length;
        when it's larger than 1, it specifies the absolute negative window size. Defaults to 0.5.

    Returns:
        Tensor: Calculated CT loss.
    """
    if ct_length <= 0:  # no need for calculating CT loss
        return 0.0

    if ct_length <= 1:  # portion of the total length (i.e., CE length)
        ct_length = round(input.size(1) * ct_length)
    else:  # exact value
        ct_length = round(ct_length)

    input = input[..., :ct_length, :]
    target = target[..., :ct_length]

    assert (
        preced_m_negatives > 0
    ), "preced_m_negatives must be greater than 0 when using CT loss."
    if preced_m_negatives <= 1:  # portion of ct_length
        preced_m_negatives = round(preced_m_negatives * ct_length)
    else:  # exact value
        preced_m_negatives = round(preced_m_negatives)

    if ignore_index != pad_id:
        target_with_pad = target.masked_fill(target.eq(ignore_index), pad_id)
    else:
        target_with_pad = target

    non_padding = target_with_pad != pad_id

    preced_tokens = preced_negatives(target_with_pad, preced_m_negatives, pad_id)
    # if preced_m_negatives:
    positive_scores = input.gather(2, target_with_pad.unsqueeze(-1))  # label scores
    negative_scores = input.gather(2, preced_tokens)
    neg_minus_pos = negative_scores - positive_scores
    exp = neg_minus_pos.exp()

    pad_mask = preced_tokens.ne(pad_id).int()
    sum_exp = (exp * pad_mask).sum(dim=-1)  # don't use pad tokens as negatives
    losses = (1 + sum_exp).log() * non_padding.int()

    denominator = non_padding.int().sum()
    ct_loss = losses.sum() / (denominator + 1e-10)  # Add small epsilon

    return ct_loss


def preced_negatives(
    labels=None,
    preced_m_negatives=0,
    pad_id=0,
):
    preced_tokens = None
    if preced_m_negatives:  # use previous k tokens as negatives
        preced_tokens = labels.unsqueeze(1).expand(
            labels.size(0), labels.size(1), labels.size(1)
        )
        mask = torch.ones_like(preced_tokens).bool()
        mask = torch.ones_like(preced_tokens).tril(-1).bool()
        if preced_m_negatives > 0:
            mask = mask.triu(-preced_m_negatives)
        preced_tokens = preced_tokens.masked_fill(~mask, pad_id)

    if preced_tokens is not None:
        preced_tokens = preced_tokens.masked_fill(
            preced_tokens == labels.unsqueeze(-1), pad_id
        )  # exclude same label tokens as negatives

    return preced_tokens
