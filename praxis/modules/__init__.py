from praxis.modules.common import MultiIdentity

DEFAULT_EXPERT_CONFIGS = {
    "peer": {
        "num_experts": 32**2,
        "num_heads": 4,
        "k": 8,
        "key_dims": 90,
        "offset_heads": False,
    },
    "smear": {"num_experts": 3},
    "glu": {},
    "mlp": {},
}
