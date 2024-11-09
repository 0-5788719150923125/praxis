from torch.nn import PReLU
from transformers.activations import ACT2CLS, ClassInstantier

from praxis.activations.nmda import NMDA
from praxis.activations.serf import SERF
from praxis.activations.sin import Sine
from praxis.activations.sinlu import SinLU

ACTIVATION_MAP = dict(prelu=PReLU, nmda=NMDA, serf=SERF, sin=Sine, sinlu=SinLU)

for k, v in ACTIVATION_MAP.items():
    ACT2CLS.update({k: v})

ACT2FN = ClassInstantier(ACT2CLS)
