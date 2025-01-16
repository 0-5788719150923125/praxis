def get_optimizer_profile(name="AdamW"):
    lowercase_profiles = {k.lower(): v for k, v in OPTIMIZER_PROFILES.items()}
    return {**lowercase_profiles.get(name.lower()), "wd_ban_list": WD_BAN_LIST}


# Most optimizer settings can be found here:
# https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
OPTIMIZER_PROFILES = {
    "AdamG": dict(
        optimizer_name="AdamG",
        lr=1.0,
        weight_decay=0.1,
        weight_decouple=True,
        p=0.5,
        q=0.24,
        betas=(0.95, 0.999, 0.95),
    ),
    "AdamW": dict(
        optimizer_name="AdamW",
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    ),
    "AdEMAMix": dict(
        optimizer_name="AdEMAMix",
        lr=0.001,
        weight_decay=0.1,
        weight_decouple=True,
        betas=(0.9, 0.95, 0.9999),
        alpha=5.0,
        cautious=False,
    ),
    "Lion": dict(
        optimizer_name="Lion",
        lr=0.000333,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        r=0.95,
        use_gc=True,
        adanorm=True,
        cautious=True,
    ),
    "Prodigy": dict(
        optimizer_name="Prodigy",
        lr=1.0,
        weight_decay=0.1,
        weight_decouple=True,
        bias_correction=True,
        safeguard_warmup=True,
    ),
    "SOAP": dict(
        optimizer_name="SOAP",
        lr=2e-4,
        weight_decay=0.1,
        precondition_frequency=10,
        max_precondition_dim=1024,
        normalize_gradient=False,
        correct_bias=True,
        precondition_1d=False,
        merge_dims=False,
    ),
}

WD_BAN_LIST = [
    "bias",
    "edge_embeddings",
    "spatial_embeddings",
    "Embedding",
    "BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "LayerNorm",
    "RMSNorm",
    "InstanceNorm",
    "InstanceNorm1d",
    "InstanceNorm3d",
    "InstanceNorm2d",
    "PReLU",
    "SinLU",
    "NMDA",
]
