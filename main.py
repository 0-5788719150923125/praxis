#!/usr/bin/env python3
"""Main entrypoint script for Praxis language models.

main() reads top-to-bottom as the training pipeline; each step lives in
its domain package (cli, data, trainers, callbacks, web, ...).
"""

from praxis.utils import configure_cuda_allocator, configure_multiprocessing

# Force 'spawn' before importing anything that may touch CUDA or spawn a
# DataLoader worker. The DataModule reads this to decide num_workers.
configure_multiprocessing()

# Default the CUDA allocator to expandable segments (curbs reserved-VRAM
# fragmentation from variable-shape workloads). Must precede CUDA init.
configure_cuda_allocator()

import sys

from praxis.callbacks import build_training_callbacks, create_printing_progress_bar
from praxis.cli import RunConfig, create_praxis_config, integration_loader, parse_cli
from praxis.data import get_datamodules
from praxis.data.runs import print_runs, setup_training_run
from praxis.generation import Generator
from praxis.optimizers import build_optimizer_and_scheduler, safe_parameter_stats
from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.train import run_train_tokenizer_cli
from praxis.trainers import (
    assemble_model,
    assemble_trainer,
    build_model_info,
    ensure_ray,
    print_training_banner,
    run_training,
    setup_environment,
)
from praxis.utils import (
    resolve_resume_checkpoint,
    show_launch_animation,
    update_license_timestamp,
)


def main():
    """Configure, build, and run one training session."""
    args, processed_args = parse_cli()
    setup_environment(no_docs=processed_args.get("no_docs", False))
    cfg = RunConfig.from_args(processed_args, args)

    # Shortcut modes that exit before building a model.
    if cfg.list_runs:
        return print_runs(cfg.cache_dir)
    if cfg.train_tokenizer:
        return run_train_tokenizer_cli(cfg)

    # Ray is an optional runtime install (like integrations): only fetched when
    # the selected trainer actually needs it, before anything imports `ray`.
    ensure_ray(cfg.trainer_type)

    cfg.apply_distributed_env()
    run = setup_training_run(cfg)

    tokenizer = create_tokenizer(
        vocab_size=cfg.vocab_size,
        encoder_type=cfg.encoder_type,
        tokenizer_type=cfg.tokenizer_type,
        cache_dir=run.cache_dir,
    )
    # The tokenizer owns vocab_size, so the config reconciles against it.
    config = create_praxis_config(args, tokenizer)
    cfg.vocab_size = config.vocab_size

    bundle = assemble_model(cfg, config)
    model_info = build_model_info(cfg, config, bundle, run)
    ckpt_path = resolve_resume_checkpoint(run.cache_dir, cfg.reset)

    generator = Generator(bundle.model, tokenizer, device=cfg.device)
    param_stats = safe_parameter_stats(bundle.model)

    # Web services first (so dataset loading can log to them), then init hooks.
    from praxis.web import start_services
    from praxis.web.spec_data import snapshot_run_spec

    services = start_services(
        cfg, run, generator, tokenizer, integration_loader, param_stats, ckpt_path
    )

    dataintegration = get_datamodules(
        cfg.seed,
        cfg.train_datasets,
        cfg.validation_datasets,
        tokenizer,
        bundle.hparams,
        cfg.data_path,
        cfg.rl_type,
        run_dir=run.cache_dir,
        data_metrics_log_interval=50,
        enable_chat_validation=True,  # Always enabled
        strict_chat_validation=False,  # Warning mode (skip invalid docs)
        weighting_mode=cfg.sampler_mode,
    )

    warmup_steps = config.warmup_steps or bundle.hparams["target_batch_size"] * 4
    optimizer, scheduler, param_stats = build_optimizer_and_scheduler(
        bundle.model,
        cfg,
        bundle.optimizer_config,
        bundle.disable_schedule,
        warmup_steps,
        services,
        param_stats,
    )

    # Snapshot the run spec so the Identity tab can inspect it after exit.
    snapshot_run_spec(cfg, run, generator, param_stats, services)

    progress_bar = create_printing_progress_bar(
        process_position=0, leave=True, use_dashboard=cfg.use_dashboard
    )
    callbacks = build_training_callbacks(
        cfg,
        run,
        bundle.model,
        config,
        bundle.hparams,
        tokenizer,
        generator,
        dataintegration,
        services,
        progress_bar,
        model_info,
    )

    trainer, train_model = assemble_trainer(
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
    )

    show_launch_animation(bundle.model, run.truncated_hash)
    print_training_banner(bundle, services)
    update_license_timestamp()

    return run_training(
        trainer, train_model, dataintegration, ckpt_path, services, progress_bar
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
