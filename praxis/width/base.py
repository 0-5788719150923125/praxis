"""The identity width policy: every channel participates, always."""

from contextlib import contextmanager

import torch.nn as nn


class FullWidth(nn.Module):
    """No-op width policy. The full model runs at full rank every step - the
    default, selected by ``--width-type none``. An ``nn.Module`` (carrying no
    parameters) so it shows on the architecture blueprint like any other piece."""

    def __init__(self, **kwargs):
        super().__init__()

    def fraction(self, depth, max_depth):
        return 1.0

    def profile(self, max_depth):
        """No arch to plot."""
        return None

    @contextmanager
    def scope(self, experts, current_depth, max_depth):
        yield
