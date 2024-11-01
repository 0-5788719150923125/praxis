from torch.nn import PReLU
from transformers.activations import ACT2CLS, ClassInstantier

from praxis.activations.nmda import NMDA
from praxis.activations.serf import SERF
from praxis.activations.sinlu import SinLU

ACT2CLS.update({"prelu": PReLU})
ACT2CLS.update({"nmda": NMDA})
ACT2CLS.update({"serf": SERF})
ACT2CLS.update({"sinlu": SinLU})

ACT2FN = ClassInstantier(ACT2CLS)
