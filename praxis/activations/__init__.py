from torch.nn import PReLU
from transformers.activations import ACT2CLS, ClassInstantier

from praxis.activations.jagged_sine import JaggedSine
from praxis.activations.nmda import NMDA
from praxis.activations.periodic_relu import PeriodicReLU
from praxis.activations.serf import SERF
from praxis.activations.sin import Sine
from praxis.activations.sin_cos import SineCosine
from praxis.activations.sinlu import SinLU
from praxis.activations.snake import Snake
from praxis.activations.sugar import SUGAR

ACTIVATION_MAP = dict(
    jagged_sin=JaggedSine,
    nmda=NMDA,
    periodic_relu=PeriodicReLU,
    prelu=PReLU,
    serf=SERF,
    sin=Sine,
    sin_cos=SineCosine,
    sinlu=SinLU,
    snake=Snake,
    sugar=SUGAR,
)

for k, v in ACTIVATION_MAP.items():
    ACT2CLS.update({k: v})

ACT2FN = ClassInstantier(ACT2CLS)
ACTIVATION_REGISTRY = dict(sorted(ACT2FN.items()))
