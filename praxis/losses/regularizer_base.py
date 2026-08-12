import torch.nn as nn
from torch import Tensor


class BaseRegularizer(nn.Module):
    """An additive representation-shaping loss.

    Regularizers read the trunk's hidden states and return a scalar that joins
    the main objective additively - they shape the representation geometry, they
    do not replace the LM loss. The model holds them in a list (``model.reg``)
    and folds each one's ``forward`` into the loss container under ``name``.

    Subclasses set ``name`` (the loss-container tag), may declare
    ``metric_descriptions`` (dashboard chart hints), and may override
    ``training_metrics`` to surface per-step diagnostics.

    ``**ctx`` carries optional extras the model can supply that not every
    regularizer wants (currently ``classifier``, the output readout). Accept and
    ignore what you don't use - the model passes the same context to all of
    them, so a regularizer that needs nothing extra keeps working unchanged.
    """

    name = "regularizer"
    metric_descriptions: dict = {}

    def forward(self, hidden_states: Tensor, input_ids: Tensor, **ctx) -> Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any state accumulated during a forward. Called at the START of
        every model forward, unconditionally.

        ``forward`` is only invoked when the model is training AND has labels,
        so a regularizer that collects state from other modules mid-forward
        cannot rely on its own call to clear that state - a labels-free
        training forward would leave it behind. Anything still holding an
        autograd graph then survives into a later iteration, where the
        parameters it references have already been modified in place by the
        optimizer, and backward dies with "one of the variables needed for
        gradient computation has been modified by an inplace operation".
        Override this to make that impossible."""

    def training_metrics(self) -> dict:
        return {}
