from pytorch_optimizer import create_optimizer

from praxis.optimizers.param_counter import (
    count_model_parameters,
    count_optimizer_parameters,
    get_parameter_stats,
)
from praxis.optimizers.wrappers import (
    WRAPPER_REGISTRY,
    SequentialWrapper,
    wrappers_disable_schedule,
)


def get_optimizer_profile(name="AdamW", disable_schedule=False):
    profiles = {k.lower(): v for k, v in OPTIMIZER_PROFILES.items()}
    profile = {**profiles.get(name.lower()), "wd_ban_list": WD_BAN_LIST}
    disable_schedule = profile.get("disable_schedule", disable_schedule)
    if "disable_schedule" in profile:
        del profile["disable_schedule"]
    return profile, disable_schedule


def _promote_tasker_lr(optimizer, model) -> None:
    """Move tasker.raw to its own param group with a boosted LR.

    Weighted-mean loss reduction cancels most of the gradient flowing to
    the per-task weights, so the default transformer LR leaves them
    visually frozen. This splits ``raw`` off after ``create_optimizer``
    has already placed it in whichever group (it normally lands in the
    no-weight-decay group because it's 1-D). Skips silently for fixed
    weighters or when the tasker is absent.
    """
    tm = getattr(model, "tasker", None)
    raw = getattr(tm, "raw", None)
    multiplier = float(getattr(tm, "lr_multiplier", 0.0) or 0.0)
    if raw is None or multiplier <= 0 or multiplier == 1.0:
        return

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is raw:
                group["params"].pop(i)
                new_group = {
                    **group,
                    "params": [raw],
                    "lr": float(group["lr"]) * multiplier,
                    "weight_decay": 0.0,
                }
                optimizer.add_param_group(new_group)
                print(
                    f"[Optimizer] tasker.raw promoted to dedicated group: "
                    f"lr={new_group['lr']:.4g} (x{multiplier:g})"
                )
                return


def get_optimizer(model, wrappers=(), *args, **kwargs):
    """Build the base optimizer and apply a sequence of registry wrappers.

    ``wrappers`` is an ordered list of WRAPPER_REGISTRY keys (e.g.
    ``["ortho", "schedule_free"]``), applied innermost-first. Schedule-free
    wrappers handle their own lr/weight-decay prep, so nothing here is
    special-cased.
    """
    optimizer = create_optimizer(model, *args, **kwargs)
    _promote_tasker_lr(optimizer, model)
    optimizer = SequentialWrapper(wrappers)(optimizer)
    if hasattr(optimizer, "train"):
        optimizer.train()
    return optimizer


def safe_parameter_stats(model, optimizer=None):
    """Compute parameter statistics, returning {} if anything goes wrong."""
    try:
        if optimizer is not None:
            return get_parameter_stats(model, optimizer)
        return get_parameter_stats(model)
    except Exception:
        return {}


def build_optimizer_and_scheduler(
    model,
    cfg,
    optimizer_config,
    disable_schedule,
    warmup_steps,
    services=None,
    param_stats=None,
):
    """Create the optimizer and scheduler, refreshing param stats.

    Returns ``(optimizer, scheduler, param_stats)``. On a stats failure the
    passed-in ``param_stats`` is kept so the API server isn't cleared.
    """
    from praxis.schedulers import get_scheduler_func

    scheduler_func = get_scheduler_func(
        optimizer_config=optimizer_config,
        disable_schedule=disable_schedule,
        warmup_steps=warmup_steps,
    )

    optimizer = get_optimizer(
        model,
        wrappers=cfg.optimizer_wrappers,
        **optimizer_config,
    )

    api_server = getattr(services, "api_server", None)
    try:
        param_stats = get_parameter_stats(model, optimizer)
        if api_server and hasattr(api_server, "update_param_stats"):
            api_server.update_param_stats(param_stats)
    except Exception:
        pass

    scheduler = scheduler_func(optimizer)
    return optimizer, scheduler, param_stats


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
        lr=0.0003,
        weight_decay=0.1,
        betas=(0.95, 0.98),
        use_gc=True,
        adanorm=True,
        cautious=True,
    ),
    "MARS": dict(
        optimizer_name="MARS",
        mars_type="shampoo",
        lr=0.0003,
        gamma=0.025,
        optimize_1d=True,
        betas=(0.95, 0.99),
        betas_1d=(0.9, 0.95),
        weight_decay=0.1,
        weight_decay_1d=0.1,
        cautious=True,
    ),
    "Muon": dict(
        optimizer_name="Muon",
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
        adamw_lr=0.0003,
        adamw_betas=(0.9, 0.95),
        adamw_wd=0.1,
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
    "Serpent",
]
