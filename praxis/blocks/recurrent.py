import torch.nn as nn

from praxis.modules.recurrent import minGRU
from praxis.modules.smear import PraxisSMEAR


class PraxisRecurrent(nn.Module):
    """
    A recurrent block type.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.experts = PraxisSMEAR(
            config,
            experts=nn.ModuleList(
                [
                    minGRU(config.hidden_size, proj_out=True)
                    for _ in range(config.num_experts)
                ]
            ),
        )

    def forward(self, inputs, current_state, *args, **kwargs):
        outputs, new_state, aux_loss = self.experts(self.norm(inputs), current_state)
        return outputs + inputs, new_state, aux_loss
