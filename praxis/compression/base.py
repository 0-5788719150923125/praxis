from typing import List, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class NoCompression(nn.Module):
    """
    A base compression class, which is a no-op by default.
    """

    def __init__(self, config: ConfigType):
        super().__init__()

    def reduce_sequence(self, sequence: Tensor, *args, **kwargs):
        return sequence

    def expand_sequence(self, sequence: Tensor, *args, **kwargs):
        return sequence

    def reduce_block_ids(self, block_ids: Tensor, *args, **kwargs):
        return block_ids
