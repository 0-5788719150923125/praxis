from pytorch_optimizer import create_optimizer

from praxis.optimizers.composite import CompositeOptimizer
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


def _split_muon_params(model):
    """Partition params for Muon: only interior >=2D matrices get
    orthogonalized; embeddings, the LM head, norms, biases and scalars go to
    Muon's internal AdamW.

    Muon's library routing is purely ``ndim >= 2`` - it does NOT detect
    embeddings/heads despite the docstring, so without this split it would
    orthogonalize the embedding and output matrices (the classic instability).
    A ``vocab_size`` dimension in the shape flags embeddings, the head, and any
    tied weight at once; ``nn.Embedding`` membership is the belt-and-suspenders.
    On doubt we route to AdamW (safe) rather than Muon.
    """
    import torch.nn as nn

    vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    emb_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        is_vocab = vocab is not None and vocab in tuple(p.shape)
        if p.ndim < 2 or is_vocab or id(p) in emb_ids:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params


def _build_muon(muon_params, adamw_params, profile):
    """Construct Muon across its two incompatible APIs (Praxis tracks bleeding
    edge, so we adapt rather than pin).

    - Old (<=3.5): ``Muon(muon_params, adamw_params=..., betas=...)`` - routes
      by ndim internally, internal-AdamW betas live under ``betas``.
    - New (>=3.10): explicit param groups each flagged ``use_muon``; the
      internal-AdamW betas live under ``adamw_betas``.

    ``adamw_params=None`` builds a Muon-only optimizer (the composite primary).
    """
    import inspect

    from pytorch_optimizer import Muon

    old_api = "adamw_params" in inspect.signature(Muon.__init__).parameters
    if old_api:
        kwargs = dict(profile)
        if "adamw_betas" in kwargs:  # old API names this `betas`
            kwargs["betas"] = kwargs.pop("adamw_betas")
        if adamw_params:
            return Muon(muon_params, adamw_params=adamw_params, **kwargs)
        return Muon(muon_params, **kwargs)

    groups = [dict(params=muon_params, use_muon=True)]
    if adamw_params:
        groups.append(dict(params=adamw_params, use_muon=False))
    return Muon(groups, **profile)


def _create_muon(model, **profile):
    """Build Muon over interior >=2D matrices, routing embeddings/head/norms/
    biases elsewhere.

    With ``secondary_optimizer`` set (e.g. "Lion"), those params get their own
    optimizer via :class:`CompositeOptimizer`; otherwise they use Muon's
    internal AdamW. Weight decay (decoupled) applies only to the orthogonalized
    matrices - the vocab-facing group stays undecayed except for its 2D members
    (embeddings/head).
    """
    from pytorch_optimizer import Muon

    secondary_name = profile.pop("secondary_optimizer", None)
    profile = {
        k: v for k, v in profile.items() if k not in ("optimizer_name", "wd_ban_list")
    }
    muon_params, adamw_params = _split_muon_params(model)
    if not muon_params:  # nothing 2D to orthogonalize - Muon would be pointless
        raise ValueError("Muon: no >=2D interior params found to orthogonalize")

    if not secondary_name:
        print(
            f"[Optimizer] Muon: {len(muon_params)} matrices orthogonalized, "
            f"{len(adamw_params)} params (embeddings/head/norms/biases) on AdamW."
        )
        return _build_muon(muon_params, adamw_params, profile)

    # Composite: Muon orthogonalizes the body; a separate optimizer drives the
    # vocab-facing params (so Muon gets only the interior group).
    primary = _build_muon(muon_params, None, profile)
    secondary, sec_lr = _build_secondary(secondary_name, adamw_params)
    ratio = sec_lr / float(profile["lr"])
    print(
        f"[Optimizer] Muon+{secondary_name}: {len(muon_params)} matrices "
        f"orthogonalized; {len(adamw_params)} vocab-facing params on "
        f"{secondary_name} (lr ratio {ratio:.3g})."
    )
    return CompositeOptimizer(primary, secondary, secondary_lr_ratio=ratio)


def _build_secondary(name, params):
    """Build the composite's secondary optimizer over ``params``, with weight
    decay only on its >=2D members (embeddings/head), not norms/biases.
    Returns ``(optimizer, base_lr)``."""
    from pytorch_optimizer import load_optimizer

    profile, _ = get_optimizer_profile(name)
    profile = {
        k: v for k, v in profile.items() if k not in ("optimizer_name", "wd_ban_list")
    }
    wd = profile.pop("weight_decay", 0.0)
    base_lr = float(profile.get("lr", 1e-3))
    decay = [p for p in params if p.ndim >= 2]
    nodecay = [p for p in params if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return load_optimizer(name.lower())(groups, **profile), base_lr


def get_optimizer(model, wrappers=(), *args, **kwargs):
    """Build the base optimizer and apply a sequence of registry wrappers.

    ``wrappers`` is an ordered list of WRAPPER_REGISTRY keys (e.g.
    ``["ortho", "schedule_free"]``), applied innermost-first. Schedule-free
    wrappers handle their own lr/weight-decay prep, so nothing here is
    special-cased.
    """
    if str(kwargs.get("optimizer_name", "")).lower() == "muon":
        optimizer = _create_muon(model, **kwargs)
    else:
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

    # Let a multi-stage model re-warm the LR when a later stage activates cold
    # params (e.g. CALM's trunk/head at the codec freeze). Reads the boundary
    # live each step; -1 until/unless one is reported.
    stage_anchor = getattr(model, "stage_warmup_anchor", None)
    scheduler_func = get_scheduler_func(
        optimizer_config=optimizer_config,
        disable_schedule=disable_schedule,
        warmup_steps=warmup_steps,
        stage_anchor=stage_anchor,
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
        # Muon orthogonalizes only >=2D params; embeddings, the LM head, and
        # scalar/vector params auto-route to the internal AdamW (adamw_*) - the
        # classic instability (orthogonalizing an embedding matrix) can't happen.
        # use_adjusted_lr scales LR per matrix shape (Moonlight): automatic,
        # model-agnostic, and robust to CALM's variable latent shapes. lr is
        # conservative for small-model LM (warmup ramps in). NB: the internal
        # AdamW betas key is `betas`, not `adamw_betas` (the latter is ignored).
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.1,
        use_adjusted_lr=True,
        # Vocab-facing params (embeddings/head/norms/biases) get their own
        # optimizer via CompositeOptimizer. Lion's sign signal has clean
        # semantics on token-frequency geometry, paired with Muon's full-
        # spectrum signal in the interior. Set to None to use Muon's internal
        # AdamW instead (the adamw_* keys below apply only in that case).
        secondary_optimizer="Lion",
        adamw_lr=0.0003,
        adamw_betas=(0.9, 0.95),
        adamw_wd=0.0,
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
