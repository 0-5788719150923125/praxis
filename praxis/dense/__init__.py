from praxis.dense.glu import GatedLinearMLP
from praxis.dense.kan import KolmogorovArnoldNetwork
from praxis.dense.mlp import MultiLayerPerceptron
from praxis.dense.peer import ParameterEfficientExpertRetrieval
from praxis.dense.poly import PolynomialExpansionMLP
from praxis.dense.scatter import ScatterMLP

DENSE_REGISTRY = dict(
    mlp=MultiLayerPerceptron,
    glu=GatedLinearMLP,
    poly=PolynomialExpansionMLP,
    scatter=ScatterMLP,
    kan=KolmogorovArnoldNetwork,
    peer=ParameterEfficientExpertRetrieval,
)
