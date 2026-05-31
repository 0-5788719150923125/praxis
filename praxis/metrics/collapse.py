"""Softmax collapse metric from "Grokking at the Edge of Stability".

Reference:
    https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/logger.py#L154
"""

import torch
from torch import Tensor


def compute_softmax_collapse(logits: Tensor) -> float:
    """Fraction of positions where softmax has fully collapsed to one-hot.

    A value of 1.0 means every position's softmax output is numerically
    one-hot (the partition function equals 1 after subtracting the max).
    Rising values during training signal loss of entropy in the output
    distribution.
    """
    # No token logits this step (e.g. CALM's codec-only pretraining phase
    # returns logits=None); the metric doesn't apply, report 0.
    if logits is None:
        return 0.0
    with torch.no_grad():
        output_off = logits - logits.amax(dim=1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_exp = torch.sum(exp_output, dim=-1, keepdim=True)
        return float((sum_exp == 1).float().mean().item())
