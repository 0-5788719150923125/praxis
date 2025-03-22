from pytorch_optimizer import create_optimizer
from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad, ScheduleFreeWrapper


def get_optimizer_profile(name="AdamW", shuffle=False, disable_schedule=False):
    profiles = {k.lower(): v for k, v in OPTIMIZER_PROFILES.items()}
    profile = {**profiles.get(name.lower()), "wd_ban_list": WD_BAN_LIST}
    profile["weight_decay"] = 0 if shuffle else profile.get("weight_decay", 0)
    disable_schedule = profile.get("disable_schedule", disable_schedule)
    if "disable_schedule" in profile:
        del profile["disable_schedule"]
    return profile, disable_schedule


def get_optimizer(
    model,
    trac=False,
    ortho=False,
    lookahead=False,
    schedule_free=False,
    *args,
    **kwargs,
):
    optimizer = create_optimizer(model, *args, **kwargs)
    if trac:
        optimizer = TRAC(optimizer, num_coefs=128)
    if ortho:
        optimizer = OrthoGrad(optimizer)
    if lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5, pullback_momentum="none")
    if schedule_free:
        optimizer = ScheduleFreeWrapper(
            optimizer, momentum=0.98, r=0.5, weight_decay=0.1
        )
    if hasattr(optimizer, "train"):
        optimizer.train()
    return optimizer


# Most optimizer settings can be found here:
# https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
OPTIMIZER_PROFILES = {
    "AdamW": dict(
        optimizer_name="AdamW",
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    ),
    "Lion": dict(
        optimizer_name="Lion",
        lr=0.000333,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        r=0.98,
        use_gc=True,
        adanorm=True,
        cautious=True,
    ),
    "MARS": dict(
        optimizer_name="MARS",
        mars_type="soap",
        lr=0.000333,
        gamma=0.025,
        optimize_1d=True,
        betas=(0.95, 0.99),
        betas_1d=(0.9, 0.95),
        weight_decay=0.1,
        weight_decay_1d=0.1,
        cautious=True,
    ),
    "Prodigy": dict(
        optimizer_name="Prodigy",
        lr=1.0,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        beta3=0.98,
        growth_rate=float("inf"),
        d_coef=0.1,
        bias_correction=True,
        safeguard_warmup=False,
        disable_schedule=True,
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
