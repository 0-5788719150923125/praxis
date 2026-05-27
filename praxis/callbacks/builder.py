"""Assembles the trainer callback list for a run."""

import os


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
        MemoryProfilerCallback,
        MetricsLoggerCallback,
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
            byte_latent=cfg.byte_latent,
            debug=cfg.debug,
            get_memory_info=get_memory_info,
            api_server=api_server,
            dashboard=services.dashboard if cfg.local_rank == 0 else None,
        )
    )

    return callbacks
