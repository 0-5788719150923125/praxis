from transformers.activations import ACT2CLS, ClassInstantier

from praxis.activations.serf import SERF
from praxis.activations.sinlu import SinLU

ACT2CLS.update({"serf": SERF})
ACT2CLS.update({"sinlu": SinLU})

ACT2FN = ClassInstantier(ACT2CLS)
