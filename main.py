#!/usr/bin/env python3
"""Main training script for Praxis language models."""

# CRITICAL: Set multiprocessing start method before ANY imports that might use CUDA
# This is required for MonoForward pipeline parallelism with CUDA
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Standard library imports
import contextlib
import importlib
import itertools
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import traceback
import uuid
import warnings
from collections import Counter
from datetime import datetime, timedelta
from queue import Queue
from typing import Any, Dict, List, Optional

# Third-party imports
import torch
import torch.nn as nn
from torcheval.metrics.functional import perplexity
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# Local application imports
from api import APIServer
from builders import get_datamodules
from cli import (
    create_praxis_config,
    get_cli_args,
    get_processed_args,
    integration_loader,
    log_command,
)
from interface import TerminalDashboard
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel
from praxis.callbacks import (
    AccumulationSchedule,
    PeriodicEvaluation,
    TerminalInterface,
    TimeBasedCheckpoint,
    create_printing_progress_bar,
)
from praxis.generation import Generator
from praxis.optimizers import get_optimizer, get_optimizer_profile, get_parameter_stats
from praxis.schedulers import get_scheduler_func
from praxis.tokenizers import create_tokenizer
from praxis.trainers import (
    BackpropagationTrainer,
    Trainer,
    TrainerConfig,
    create_checkpoint_callback,
    create_logger,
    create_progress_callback,
    create_trainer_with_module,
    disable_warnings,
    reset_seed,
    seed_everything,
)
from praxis.utils import (
    check_for_updates,
    find_latest_checkpoint,
    get_memory_info,
    initialize_lazy_modules,
    perform_reset,
    show_launch_animation,
    sigint_handler,
)

# Prevent Python from creating .pyc files
sys.dont_write_bytecode = True


def setup_environment():
    """Set up the environment and configurations."""
    # Set up the SIGINT handler
    signal.signal(signal.SIGINT, sigint_handler)

    # Check for updates at startup
    check_for_updates()

    # Configure environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Disable warnings
    disable_warnings()
    logging.getLogger("pytorch").setLevel(logging.ERROR)

    # Configure warning filters
    ignored_warnings = [
        ".*Checkpoint directory.*exists and is not empty*",
        ".*JAX is multithreaded, so this will likely lead to a deadlock*",
        ".*Total length of `list` across ranks is zero.*",
    ]
    for pattern in ignored_warnings:
        warnings.filterwarnings("ignore", pattern)

    # Register Praxis models with transformers
    AutoConfig.register("praxis", PraxisConfig)
    AutoModel.register(PraxisConfig, PraxisModel)
    AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["praxis"] = "PraxisForCausalLM"


def main():
    """Main training function."""
    # Set up environment
    setup_environment()

    # Get processed arguments from CLI
    args = get_cli_args()
    processed_args = get_processed_args(args)

    # Extract all processed arguments as local variables
    locals().update(processed_args)

    # Make necessary variables accessible
    seed = processed_args["seed"]
    tokenizer_type = processed_args.get("tokenizer_type")
    tokenizer_profile = processed_args.get("tokenizer_profile")
    tokenizer_path = processed_args.get("tokenizer_path")
    encoder_type = processed_args.get("encoder_type")
    vocab_size = processed_args["vocab_size"]
    cache_dir = processed_args["cache_dir"]
    optimizer = processed_args["optimizer"]
    fixed_schedule = processed_args.get("fixed_schedule", False)
    schedule_free = processed_args.get("schedule_free", False)
    trac = processed_args.get("trac", False)
    ortho = processed_args.get("ortho", False)
    lookahead = processed_args.get("lookahead", False)
    batch_size = processed_args["batch_size"]
    target_batch_size = processed_args.get("target_batch_size", batch_size)
    block_size = processed_args["block_size"]
    device = processed_args["device"]
    dev = processed_args.get("dev", False)
    max_steps = processed_args.get("max_steps")
    use_dashboard = processed_args.get("use_dashboard", False)
    reset = processed_args.get("reset", False)
    local_rank = processed_args.get("local_rank", 0)
    no_source = processed_args.get("no_source", False)
    pile = processed_args.get("pile", False)
    phi = processed_args.get("phi", False)
    data_path = processed_args.get("data_path")
    rl_type = processed_args.get("rl_type")
    no_compile = processed_args.get("no_compile", False)
    byte_latent = processed_args.get("byte_latent", False)
    host_name = processed_args.get("host_name", "localhost")
    port = processed_args.get("port", 2100)
    disable_schedule = processed_args.get("disable_schedule", False)
    strategy = processed_args.get("strategy")
    dropout = processed_args.get("dropout", 0.1)
    trainer_type = processed_args.get("trainer_type", "backpropagation")
    pipeline_depth = processed_args.get("pipeline_depth", 4)

    (_, args_hash, truncated_hash) = log_command()

    # Set seeds for reproducibility
    seed_everything(seed, workers=True)

    # Tokenizer initialization - single unified interface
    tokenizer = create_tokenizer(
        tokenizer_name=tokenizer_type,
        tokenizer_profile=tokenizer_profile,
        tokenizer_path=tokenizer_path,
        encoder_type=encoder_type,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
    )

    # Create Transformers config from CLI args
    config = create_praxis_config(args, tokenizer)

    # Add optimizer configuration
    optimizer_config, disable_schedule_from_optimizer = get_optimizer_profile(
        optimizer, any([fixed_schedule, schedule_free])
    )
    disable_schedule = disable_schedule or disable_schedule_from_optimizer

    config.optimizer_config = optimizer_config
    config.optimizer_wrappers = {
        "trac": trac,
        "ortho": ortho,
        "lookahead": lookahead,
        "schedule_free": schedule_free,
    }

    # Misc hyperparameters
    hparams = dict(
        batch_size=batch_size,
        target_batch_size=target_batch_size,
        block_size=block_size,
        oversample_chance=0.1,  # double the block_size
        supersample_chance=0.01,  # quadruple the block_size
        hypersample_chance=0.001,  # octuple the block_size
        device=device,
        dev=dev,
        trainer_type=trainer_type,
        **config.to_dict(),
    )

    # Training config
    train_params = dict(
        accelerator=f"cpu" if device == "cpu" else "gpu",
        strategy="ddp_find_unused_parameters_true" if device == "cuda" else "auto",
        devices=[int(device.split(":")[1])] if device.startswith("cuda:") else "auto",
        max_steps=max_steps if max_steps is not None else -1,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=0,
        precision="32-true",
        # Gradient clipping not supported with manual optimization (mono_forward)
        gradient_clip_val=1.0 if trainer_type != "mono_forward" else None,
        gradient_clip_algorithm="norm" if trainer_type != "mono_forward" else None,
        benchmark=True,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=not use_dashboard,
        enable_model_summary=False,
        detect_anomaly=True if dev else False,
        val_check_interval=1024 * hparams["target_batch_size"] // hparams["batch_size"],
        num_sanity_val_steps=0,
        limit_val_batches=16384 // hparams["batch_size"],
        log_every_n_steps=10,
        logger=None,  # Will be set below based on integrations
        callbacks=[],
    )

    # Configure the learning rate scheduler
    warmup_steps = 4096
    scheduler_func = get_scheduler_func(
        optimizer_config=optimizer_config,
        disable_schedule=disable_schedule,
        warmup_steps=warmup_steps,
    )

    # Define checkpointing behavior
    checkpoint_callback = TimeBasedCheckpoint(
        save_top_k=3,
        save_last="link",
        monitor="batch",
        mode="max",
        dirpath=os.path.join(cache_dir, "model"),
        filename="model-{batch}",
        enable_version_counter=False,
        save_interval=3600,
    )

    # Bootstrap the model and trainer
    model = AutoModelForCausalLM.from_config(config)

    initialize_lazy_modules(model, device)

    # Print the total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    reduced = str(int(total_params / 10**6)) + "M"
    hparams["num_params"] = reduced

    # File cleanup
    if reset:
        perform_reset(cache_dir, truncated_hash, integration_loader)

    ckpt_path = None
    if not reset and not dev:
        symlink = os.path.join(cache_dir, "model", "last.ckpt")
        true_link = find_latest_checkpoint(cache_dir)
        if os.path.exists(symlink):
            print(f"resuming from symbolic path: {symlink}")
            ckpt_path = symlink
        elif true_link is not None and os.path.exists(true_link):
            print(f"resuming from true path: {true_link}")
            ckpt_path = true_link

    # Initialize generator for tool calling during training and inference
    generator = Generator(model, tokenizer, device=device)
    param_stats = {}

    try:
        param_stats = get_parameter_stats(model)
    except Exception as e:
        param_stats = {}

    if local_rank == 0:
        # Force reload of api integration to pick up any recent changes
        import api

        importlib.reload(api)
        from api import APIServer

        api_server = APIServer(
            generator,
            host_name,
            port,
            tokenizer,
            integration_loader,
            param_stats,
            seed,
            truncated_hash=truncated_hash,
            full_hash=args_hash,
            dev_mode=dev,
        )
        api_server.start()

        # Call API server hooks with (host, port) as in original implementation
        for hook_func in integration_loader.get_api_server_hooks():
            hook_func(api_server.host, api_server.port)
    else:
        api_server = None

    # Run init hooks for integrations BEFORE loading datasets
    # This ensures integrations are properly initialized before their datasets are checked
    integration_loader.run_init_hooks(args, cache_dir, ckpt_path=ckpt_path, truncated_hash=truncated_hash)

    # Load datasets
    use_source_code = not no_source
    dataintegration = get_datamodules(
        seed, dev, pile, phi, use_source_code, tokenizer, hparams, data_path, rl_type
    )

    # create the optimizer
    optimizer = get_optimizer(
        model,
        trac=trac,
        ortho=ortho,
        lookahead=lookahead,
        schedule_free=schedule_free,
        **optimizer_config,
    )

    # Log initial optimizer state information
    try:
        # Calculate statistics for all optimizer states
        param_stats = get_parameter_stats(model, optimizer)

        # Update the API server if it exists
        if api_server and hasattr(api_server, "update_param_stats"):
            api_server.update_param_stats(param_stats)
    except Exception as e:
        pass

    # create the scheduler
    scheduler = scheduler_func(optimizer)

    # Print training configuration will be shown later

    # Create progress bar callback (returns None if using dashboard)
    progress_bar = create_printing_progress_bar(
        process_position=0, leave=True, use_dashboard=use_dashboard
    )

    # Get evaluation parameters from processed_args
    eval_every = processed_args.get("eval_every", None)
    eval_tasks = processed_args.get("eval_tasks", None)
    debug = processed_args.get("debug", False)
    
    # Configure callbacks list based on trainer type
    if trainer_type == "mono_forward":
        # Manual optimization doesn't support AccumulationSchedule
        train_params["callbacks"] = [
            checkpoint_callback,
            PeriodicEvaluation(
                eval_every=eval_every,
                eval_tasks=eval_tasks,
                model=model,
                device=device,
                vocab_size=vocab_size,
                debug=debug
            ),
        ]
    else:
        # Automatic optimization supports all callbacks
        train_params["callbacks"] = [
            checkpoint_callback,
            AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"]),
            PeriodicEvaluation(
                eval_every=eval_every,
                eval_tasks=eval_tasks,
                model=model,
                device=device,
                vocab_size=vocab_size,
                debug=debug
            ),
        ]

    # Add progress bar if not using dashboard
    if progress_bar is not None:
        train_params["callbacks"].append(progress_bar)

    # Add TerminalInterface which handles dashboard/console output routing
    # TerminalInterface creates and manages the dashboard internally when use_dashboard=True
    quiet = processed_args.get("quiet", False)
    terminal_output_length = processed_args.get(
        "terminal_output_length", block_size * 2
    )

    train_params["callbacks"].append(
        TerminalInterface(
            tokenizer=tokenizer,
            generator=generator,
            use_dashboard=use_dashboard,
            url=api_server.get_api_addr() if api_server else None,
            progress_bar=progress_bar,
            device=device,
            quiet=quiet,
            terminal_output_length=terminal_output_length,
            byte_latent=byte_latent,
            debug=debug,
            get_memory_info=get_memory_info,
            api_server=api_server,
            seed=seed,
            truncated_hash=truncated_hash,
            total_params=total_params,  # Pass the actual number, not the string
            # Additional parameters for info panel
            optimizer_config=optimizer_config,
            strategy=strategy,
            rl_type=rl_type,
            vocab_size=vocab_size,
            depth=config.depth,
            hidden_size=config.hidden_size,
            embed_size=config.embed_size,
            dropout=dropout,
            use_source_code=use_source_code,
            dev=dev,
            target_batch_size=target_batch_size,
        )
    )

    if hparams.get("decoder_type") == "mono_forward" and not no_compile:
        print(
            "[Compile] Skipping torch.compile (MonoForward not compatible with layer-wise updates)"
        )
    elif not no_compile:
        from praxis.trainers.compile import try_compile_model, try_compile_optimizer

        # Try to compile the model and optimizer
        model = try_compile_model(model, hparams)
        optimizer = try_compile_optimizer(optimizer, hparams)

    # Try to get logger from integrations (e.g., wandb)
    integration_logger = None
    for provider in integration_loader.get_logger_providers():
        try:
            logger = provider(
                cache_dir=cache_dir,
                ckpt_path=ckpt_path, 
                truncated_hash=truncated_hash,
                wandb_enabled=getattr(args, 'wandb', False),
                args=args
            )
            if logger:
                integration_logger = logger
                print(f"[Training] Using integration logger: {type(logger).__name__}")
                break
        except Exception as e:
            print(f"[Warning] Logger provider failed: {e}")
    
    # Use integration logger if available, otherwise fall back to CSV
    if integration_logger:
        train_params["logger"] = integration_logger
    else:
        # Fallback to CSV logger
        train_params["logger"] = create_logger(
            log_dir=os.path.join(cache_dir, "logs"), name="model", format="csv"
        )

    # Create trainer and fit model
    trainer_type = hparams.get("trainer_type", "backpropagation")

    trainer, train_model = create_trainer_with_module(
        trainer_type=trainer_type,
        model=model,  # Pass raw model, not BackpropagationTrainer
        optimizer=optimizer,
        scheduler=scheduler,
        hparams=hparams,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        ckpt_path=ckpt_path,
        trainer_params=train_params,
        encoder_type=encoder_type,
        byte_latent=byte_latent,
        pipeline_depth=pipeline_depth,
    )

    # Show launch animation just before training begins
    show_launch_animation(model, truncated_hash)

    # Print final training configuration
    print(
        f"[Training] Starting with {reduced} parameters, {optimizer_config['optimizer_name']} optimizer"
    )
    if api_server:
        print(f"[Training] API available at http://{api_server.get_api_addr()}/")

    try:
        trainer.fit(train_model, dataintegration, ckpt_path=ckpt_path)

        # Training completed successfully
        print("[Training] Completed successfully")

        # Run integration cleanup hooks
        integration_loader.run_cleanup_hooks()

        # Stop API server if running
        if api_server:
            api_server.stop()

        # Force exit to ensure all threads terminate
        os._exit(0)

    except Exception as e:
        # Run integration cleanup hooks
        integration_loader.run_cleanup_hooks()

        # If we have a dashboard running, force crash it to show the error
        if (
            "progress_bar" in locals()
            and hasattr(progress_bar, "dashboard")
            and progress_bar.dashboard
        ):
            error_text = traceback.format_exc()
            progress_bar.dashboard.crash_with_error(error_text)
        else:
            # No dashboard, just re-raise the exception normally
            raise

    except KeyboardInterrupt:
        # Run integration cleanup hooks
        integration_loader.run_cleanup_hooks()

        # Handle Ctrl+C gracefully
        if (
            "progress_bar" in locals()
            and hasattr(progress_bar, "dashboard")
            and progress_bar.dashboard
        ):
            progress_bar.dashboard.stop()
            # Dashboard already prints the interruption message
        else:
            print("\n🛑 Training interrupted by user", file=sys.stderr)

        # Stop API server if running
        if api_server:
            api_server.stop()

        # Force exit
        os._exit(0)


if __name__ == "__main__":
    main()
