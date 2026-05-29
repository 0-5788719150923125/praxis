from praxis.encoding.alibi import ALiBi
from praxis.encoding.archope import ArcHoPE
from praxis.encoding.hope import HoPE
from praxis.encoding.nope import NoPE
from praxis.encoding.rope import RoPE

ENCODING_REGISTRY = dict(nope=NoPE, alibi=ALiBi, rope=RoPE, hope=HoPE, arc=ArcHoPE)
