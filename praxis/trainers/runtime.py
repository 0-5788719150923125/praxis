"""Trainer assembly, logging, and the fit/teardown loop."""

import os
import signal


def resolve_training_logger(cfg, run, ckpt_path):
    """Return an integration logger (e.g. wandb) or fall back to CSV."""
    from praxis.cli import integration_loader
    from praxis.trainers.factory import create_logger

    for provider in integration_loader.get_logger_providers():
        try:
            logger = provider(
                cache_dir=run.cache_dir,
                ckpt_path=ckpt_path,
                truncated_hash=run.truncated_hash,
                wandb_enabled=getattr(cfg.args, "wandb", False),
                args=cfg.args,
            )
            if logger:
                print(f"[TRAIN] Using integration logger: {type(logger).__name__}")
                return logger
        except Exception as e:
            print(f"[Warning] Logger provider failed: {e}")

    return create_logger(
        log_dir=os.path.join(run.cache_dir, "logs"), name="model", format="csv"
    )


def _build_trainer_params(cfg, bundle, callbacks, logger):
    """Build the Lightning trainer kwargs dict."""
    from praxis.environments import EnvironmentFeatures
    from praxis.trainers.capabilities import get_trainer_capabilities

    hparams = bundle.hparams
    caps = get_trainer_capabilities(cfg.trainer_type)

    return dict(
        accelerator="cpu" if cfg.device == "cpu" else "gpu",
        strategy=(
            "ddp_find_unused_parameters_true"
            if (cfg.num_nodes > 1 or cfg.device == "cuda")
            else "auto"
        ),
        num_nodes=cfg.num_nodes,
        devices=(
            [int(cfg.device.split(":")[1])]
            if cfg.device.startswith("cuda:")
            else "auto"
        ),
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=0,
        precision="32-true",
        gradient_clip_val=(1.0 if caps.supports_gradient_clipping else None),
        gradient_clip_algorithm=("norm" if caps.supports_gradient_clipping else None),
        benchmark=True,
        deterministic=False,
        enable_checkpointing=not cfg.no_checkpoints,
        enable_progress_bar=not cfg.use_dashboard and not cfg.headless,
        enable_model_summary=False,
        detect_anomaly=EnvironmentFeatures.is_enabled("detect_anomaly"),
        val_check_interval=cfg.val_every
        * hparams["target_batch_size"]
        // hparams["batch_size"],
        num_sanity_val_steps=0,
        limit_val_batches=16384 // hparams["batch_size"],
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
    )


def assemble_trainer(
    cfg,
    run,
    bundle,
    optimizer,
    scheduler,
    tokenizer,
    ckpt_path,
    callbacks,
    model_info,
    services,
    warmup_steps,
):
    """Create the trainer + training module, then route MF inference if needed.

    Keeps the create_trainer_with_module kwarg contract that the
    mono_forward trainers document; non-MF trainers ignore the Ray knobs.
    """
    from praxis.generation import bos_prompt, swap_inference_generator
    from praxis.trainers.factory import create_trainer_with_module

    api_server = services.api_server
    logger = resolve_training_logger(cfg, run, ckpt_path)
    train_params = _build_trainer_params(cfg, bundle, callbacks, logger)

    accumulate_grad_batches = (
        1
        if cfg.batch_size >= cfg.target_batch_size
        else -(-cfg.target_batch_size // cfg.batch_size)
    )

    trainer, train_model = create_trainer_with_module(
        trainer_type=cfg.trainer_type,
        model=bundle.model,
        optimizer=optimizer,
        scheduler=scheduler,
        hparams=bundle.hparams,
        tokenizer=tokenizer,
        cache_dir=run.cache_dir,
        ckpt_path=ckpt_path,
        trainer_params=train_params,
        encoder_type=cfg.encoder_type,
        byte_latent=cfg.byte_latent,
        pipeline_depth=cfg.pipeline_depth,
        device=cfg.device,
        # Ray Mono-Forward flags - ignored by non-mono_forward trainers.
        ray_address=cfg.ray_address,
        ray_num_replicas_per_layer=cfg.ray_num_replicas_per_layer,
        ray_head_sync_every=cfg.ray_head_sync_every,
        ray_pipeline_api=cfg.ray_pipeline_api,
        inference_prompt=bos_prompt(tokenizer),
        inference_every_seconds=cfg.infer_every,
        model_info=model_info,
        dashboard_url=api_server.get_api_addr() if api_server else None,
        accumulate_grad_batches=accumulate_grad_batches,
        optimizer_config=bundle.optimizer_config,
        optimizer_wrappers={
            "trac": cfg.trac,
            "ortho": cfg.ortho,
            "lookahead": cfg.lookahead,
            "schedule_free": cfg.schedule_free,
        },
        warmup_steps=warmup_steps,
        disable_schedule=bundle.disable_schedule,
        # val_every is in effective steps; the trainer converts to raw
        # batches via accumulate_grad_batches.
        val_every=cfg.val_every,
        dynamics_log_freq=10,
        save_every=cfg.save_every,
    )

    swap_inference_generator(trainer, tokenizer, api_server)
    return trainer, train_model


def print_training_banner(bundle, services):
    """Print the final params/optimizer line and the API URL if serving."""
    print(
        f"[TRAINING] Starting with {bundle.num_params} parameters, "
        f"{bundle.optimizer_config['optimizer_name']} optimizer"
    )
    api_server = services.api_server
    if api_server:
        addr = api_server.get_api_addr()
        url = (
            f"{addr}/"
            if addr.startswith(("http://", "https://"))
            else f"http://{addr}/"
        )
        print(f"[TRAINING] API available at {url}")


def run_training(
    trainer, train_model, dataintegration, ckpt_path, services, progress_bar
):
    """Run ``trainer.fit`` and tear down cleanly. Returns an exit code.

    Lightning usually drains its own SIGINT and returns into the success
    branch; the KeyboardInterrupt branch only runs when an interrupt escapes
    the trainer (e.g. during dataset setup).
    """
    import traceback

    from praxis.utils import graceful_shutdown

    api_server = services.api_server

    def cleanup_signal_handler(signum, frame):
        # During teardown a Ctrl+C should exit fast, not stall in another
        # graceful path.
        print("\n⚠️  Forcing exit...")
        os._exit(130)

    try:
        trainer.fit(
            train_model, dataintegration, ckpt_path=ckpt_path, weights_only=False
        )
        print("[TRAIN] Completed successfully")
        signal.signal(signal.SIGINT, cleanup_signal_handler)
        signal.signal(signal.SIGTERM, cleanup_signal_handler)
        graceful_shutdown(api_server, exit_code=0, reason="training complete")
        return 0  # unreachable; graceful_shutdown calls os._exit

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user")
        signal.signal(signal.SIGINT, cleanup_signal_handler)
        signal.signal(signal.SIGTERM, cleanup_signal_handler)
        graceful_shutdown(api_server, exit_code=130, reason="interrupted")
        return 130

    except Exception:
        if (
            progress_bar is not None
            and hasattr(progress_bar, "dashboard")
            and progress_bar.dashboard
        ):
            progress_bar.dashboard.crash_with_error(traceback.format_exc())
        else:
            traceback.print_exc()
        graceful_shutdown(api_server, exit_code=1, reason="fatal error")
        return 1
