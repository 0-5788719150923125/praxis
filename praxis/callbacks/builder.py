"""Assembles the trainer callback list for a run."""

import os
import random

_FIRST_AUTHOR = "Ryan J. Brooks"


def _resolve_authors(authors, seed):
    """Order the paper's authors: the original first, the rest appended - except
    a deterministic 10%% of runs (seeded by the run seed, so it is stable for a
    given run and leaves training RNG untouched) where the original slips to the
    bottom of the list, a quiet hint of discovery."""
    out = [_FIRST_AUTHOR] + [x for x in (authors or []) if x and x != _FIRST_AUTHOR]
    if len(out) > 1 and random.Random(seed).random() < 0.10:
        out = out[1:] + out[:1]  # first -> last
    return out


def build_training_callbacks(
    cfg,
    run,
    model,
    config,
    hparams,
    tokenizer,
    generator,
    dataintegration,
    services,
    progress_bar,
    model_info,
):
    """Build the ordered list of Lightning callbacks for training.

    Ordering matters: BrierLM must precede MetricsLogger so ``val_brierlm``
    is in ``callback_metrics`` when MetricsLogger drains them.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    from praxis.callbacks.lightning import (
        AccumulationSchedule,
        BrierLMCallback,
        DynamicsLoggerCallback,
        EngagementLiveRewardCallback,
        HarmonicWeightRLCallback,
        MemoryProfilerCallback,
        MetricsLoggerCallback,
        PaperBuildCallback,
        PeriodicEvaluation,
        TerminalInterface,
    )
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback
    from praxis.trainers.capabilities import get_trainer_capabilities
    from praxis.utils import get_memory_info

    api_server = services.api_server
    cache_dir = run.cache_dir

    callbacks = [SignalHandlerCallback()]

    if not cfg.no_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                every_n_train_steps=cfg.save_every,
                save_top_k=3,
                save_last="link",
                monitor="batch",
                mode="max",
                dirpath=os.path.join(cache_dir, "model"),
                filename="model-{batch}",
                enable_version_counter=False,
            )
        )

    if get_trainer_capabilities(cfg.trainer_type).supports_accumulation_schedule:
        callbacks.append(
            AccumulationSchedule(
                hparams["batch_size"] * cfg.num_nodes, hparams["target_batch_size"]
            )
        )

    callbacks.append(
        PeriodicEvaluation(
            eval_every=cfg.eval_every,
            eval_tasks=cfg.eval_tasks,
            model=model,
            device=cfg.device,
            vocab_size=cfg.vocab_size,
            debug=cfg.debug,
        )
    )

    # Sample-based proper scoring rule at validation; cheap on small batches.
    callbacks.append(BrierLMCallback(tokenizer=tokenizer))

    # Weight-editing RL controllers. rl_type is a list; each profile entry
    # selects a controller and bundles its behavior (edit_mode, selector), so the
    # experiment sets only rl_type, not a soup of rl_* flags. Driven here from the
    # training loop, not the forward pass. Ordered before MetricsLogger so the
    # rl_* scalars are in callback_metrics when MetricsLogger drains them.
    from praxis.policies import (
        RL_POLICIES_REGISTRY,
        get_rl_profile,
        normalize_rl_types,
    )

    for rl_name in normalize_rl_types(getattr(config, "rl_type", None)):
        _rl_profile = get_rl_profile(rl_name)
        if _rl_profile is None:
            continue  # forward-path policy; built inside the model, not here
        rl_policy = RL_POLICIES_REGISTRY[_rl_profile["policy"]](config)

        # Profile supplies the defaults; an explicit rl_* config key still wins.
        def _rl(key, cast, _p=_rl_profile):
            v = getattr(config, f"rl_{key}", None)
            return cast(v) if v is not None else cast(_p[key])

        callbacks.append(
            HarmonicWeightRLCallback(
                policy=rl_policy,
                period=_rl("period", int),
                horizon=_rl("horizon", int),
                warmup_steps=_rl("warmup_steps", int),
                reward_decay=_rl("reward_decay", float),
                edit_mode=_rl_profile["edit_mode"],
                selector=_rl_profile["selector"],
            )
        )

    # Online-learning seam: drain live web rewards into the matching forward-path
    # policy's energy baseline (Print answers -> engagement, joke approvals -> joke).
    _active_rl = normalize_rl_types(getattr(config, "rl_type", None))
    if "engagement" in _active_rl:
        callbacks.append(EngagementLiveRewardCallback())
    if "joke" in _active_rl:
        from praxis.policies.engagement_channel import LIVE_JOKES

        callbacks.append(
            EngagementLiveRewardCallback(
                channel=LIVE_JOKES,
                policy_class_name="JokePolicy",
                metric_prefix="joke",
            )
        )

    # Restrict charted task-loss weights to task types a live dataset
    # produces; a learnable weighter still drifts weights for absent tasks.
    if hasattr(dataintegration, "active_task_ids"):
        model.active_task_ids = dataintegration.active_task_ids() or None

    callbacks.append(MetricsLoggerCallback(run_dir=cache_dir))

    # Per-layer gradient dynamics are universal; expert dynamics need a router.
    routers_with_gradient_logging = ["prismatic", "smear"]
    num_experts = (
        getattr(config, "num_experts", 2)
        if config.router_type in routers_with_gradient_logging
        else 0
    )
    log_freq = 10
    print(
        f"[Setup] Adding DynamicsLoggerCallback (router_type={config.router_type}, "
        f"num_experts={num_experts}, log_freq={log_freq})"
    )
    callbacks.append(
        DynamicsLoggerCallback(
            run_dir=cache_dir, num_experts=num_experts, log_freq=log_freq
        )
    )

    # Living research paper: regenerate inputs + recompile research/main.pdf on
    # the checkpoint cadence, in a background thread. Off with --no-paper.
    if not getattr(cfg, "no_paper", False) and not getattr(
        cfg, "no_checkpoints", False
    ):
        callbacks.append(
            PaperBuildCallback(
                every=cfg.save_every,
                log_dir=cache_dir,
                authors=_resolve_authors(
                    getattr(cfg, "author", None), getattr(cfg, "seed", 0)
                ),
            )
        )

    if cfg.profile_memory:
        callbacks.append(
            MemoryProfilerCallback(
                run_dir=cache_dir,
                start_step=cfg.profile_memory_start,
                num_steps=cfg.profile_memory_steps,
                max_entries=cfg.profile_memory_max_entries,
            )
        )

    if progress_bar is not None and not cfg.headless:
        callbacks.append(progress_bar)

    # Remote-expert pool (orchestration): spins up the Node sidecar of tiny
    # experts and drives the pool each step. Added before TerminalInterface so
    # the pool status is fresh when the terminal callback paints remote_layers.
    # Profile-driven (--orchestration-type); None disables it.
    from praxis.orchestration import get_orchestration_profile

    _orch = get_orchestration_profile(getattr(config, "orchestration_type", "none"))
    if _orch:
        from praxis.callbacks.lightning import ExpertPoolCallback

        callbacks.append(
            ExpertPoolCallback(
                mixing=_orch.get("mixing", "vote"),
                sidecar=_orch.get("sidecar", True),
                init_experts=int(_orch.get("init_experts", 4)),
                vocab=int(getattr(config, "vocab_size", 16) or 16),
            )
        )

    # TerminalInterface routes dashboard/console output and manages the
    # dashboard internally when use_dashboard=True.
    callbacks.append(
        TerminalInterface(
            tokenizer=tokenizer,
            model_info=model_info,
            generator=generator,
            use_dashboard=cfg.use_dashboard,
            url=api_server.get_api_addr() if api_server else None,
            progress_bar=progress_bar,
            device=cfg.device,
            quiet=cfg.quiet,
            headless=cfg.headless,
            terminal_output_length=cfg.terminal_output_length,
            infer_every=cfg.infer_every,
            byte_level=cfg.byte_level,
            debug=cfg.debug,
            get_memory_info=get_memory_info,
            api_server=api_server,
            dashboard=services.dashboard if cfg.local_rank == 0 else None,
        )
    )

    return callbacks
