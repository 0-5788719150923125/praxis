from transformers.activations import ACT2CLS, ClassInstantier

from praxis.activations.serf import SERF

ACT2CLS.update({"serf": SERF})

ACT2FN = ClassInstantier(ACT2CLS)
