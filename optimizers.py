from pytorch_optimizer import create_optimizer
from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad


def get_optimizer_profile(name="AdamW", shuffle=False, no_schedule=False):
    profiles = {k.lower(): v for k, v in OPTIMIZER_PROFILES.items()}
    profile = {**profiles.get(name.lower()), "wd_ban_list": WD_BAN_LIST}
    profile["weight_decay"] = 0 if shuffle else profile.get("weight_decay", None)
    no_schedule = profile.get("no_schedule", no_schedule)
    if "no_schedule" in profile:
        del profile["no_schedule"]
    return profile, no_schedule


def get_optimizer(model, trac=False, ortho=False, lookahead=False, *args, **kwargs):
    optimizer = create_optimizer(model, *args, **kwargs)
    if trac:
        optimizer = TRAC(optimizer, num_coefs=128)
    if ortho:
        optimizer = OrthoGrad(optimizer)
    if lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5, pullback_momentum="none")
    return optimizer


# Most optimizer settings can be found here:
# https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
OPTIMIZER_PROFILES = {
    "AdamG": dict(
        optimizer_name="AdamG",
        lr=1.0,
        weight_decay=0.1,
        weight_decouple=True,
        p=0.2,
        q=0.24,
        betas=(0.95, 0.999, 0.95),
        no_schedule=True,
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
        betas=(0.9, 0.95, 0.9999),
        alpha=5.0,
        cautious=True,
    ),
    "Grams": dict(
        optimizer_name="Grams",
        lr=0.001,
        betas=(0.9, 0.95),
        weight_decay=0.1,
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
        betas=(0.9, 0.95),
        beta3=0.9999,
        growth_rate=float("inf"),
        d_coef=0.1,
        bias_correction=True,
        safeguard_warmup=False,
        no_schedule=True,
    ),
    "SOAP": dict(
        optimizer_name="SOAP",
        lr=0.003,
        weight_decay=0.1,
        betas=(0.95, 0.95),
        precondition_frequency=10,
        max_precondition_dim=10000,
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
