from praxis.encoding.alibi import ALiBi
from praxis.encoding.nope import NoPE
from praxis.encoding.rope import RoPE

ENCODING_REGISTRY = {"alibi": ALiBi, "nope": NoPE, "rope": RoPE}
