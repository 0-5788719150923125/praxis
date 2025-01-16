def get_optimizer_profile(name="adamw"):
    return {**OPTIMIZER_PROFILES.get(name.lower()), "wd_ban_list": WD_BAN_LIST}


# Most optimizer settings can be found here:
# https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
OPTIMIZER_PROFILES = {
    "adamg": dict(
        optimizer_name="AdamG",
        lr=1.0,
        min_lr=1e-2,
        weight_decay=0.1,
        weight_decouple=True,
        p=0.5,
        q=0.24,
        betas=(0.95, 0.999, 0.95),
    ),
    "adamw": dict(
        optimizer_name="AdamW",
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    ),
    "ademamix": dict(
        optimizer_name="AdEMAMix",
        lr=0.001,
        min_lr=0.00001,
        weight_decay=0.1,
        weight_decouple=True,
        betas=(0.9, 0.95, 0.9999),
        alpha=5.0,
        cautious=False,
    ),
    "lion": dict(
        optimizer_name="Lion",
        lr=0.0003,
        min_lr=0.000003,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        r=0.95,
        use_gc=True,
        adanorm=True,
        cautious=True,
    ),
    "prodigy": dict(
        optimizer_name="Prodigy",
        lr=1.0,
        min_lr=1e-2,
        weight_decay=0.1,
        weight_decouple=True,
        bias_correction=True,
        safeguard_warmup=True,
    ),
    "soap": dict(
        optimizer_name="SOAP",
        lr=2e-4,
        min_lr=2e-5,
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
