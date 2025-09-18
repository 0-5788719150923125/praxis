#!/usr/bin/env python3
"""Main training script for Praxis language models."""

import warnings

# CRITICAL: Set multiprocessing start method before ANY imports that might use CUDA
# This is required for MonoForward pipeline parallelism with CUDA
import torch.multiprocessing as mp

# Suppress multiprocessing resource tracker warnings early
warnings.filterwarnings(
    "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
)

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# Standard library imports
import importlib
import logging
import os
import signal
import sys
import traceback
import warnings
from datetime import datetime

# Third-party imports
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# Local application imports
from praxis.api import APIServer
from praxis.callbacks import (
    AccumulationSchedule,
    PeriodicEvaluation,
    SignalHandlerCallback,
    TerminalInterface,
    TimeBasedCheckpoint,
    create_printing_progress_bar,
)
from praxis.cli import (
    create_praxis_config,
    get_cli_args,
    get_processed_args,
    integration_loader,
    log_command,
)
from praxis.data import get_datamodules
from praxis.data.runs import RunManager
from praxis.environments import EnvironmentFeatures
from praxis.generation import Generator
from praxis.interface import TerminalDashboard
from praxis.optimizers import get_optimizer, get_optimizer_profile, get_parameter_stats
from praxis.schedulers import get_scheduler_func
from praxis.tokenizers import create_tokenizer
from praxis.trainers import (
    create_logger,
    create_trainer_with_module,
    disable_warnings,
    seed_everything,
)
from praxis.trainers.capabilities import get_trainer_capabilities
from praxis.utils import (
    check_for_updates,
    find_latest_checkpoint,
    get_memory_info,
    initialize_lazy_modules,
    perform_reset,
    show_launch_animation,
)

# Prevent Python from creating .pyc files
sys.dont_write_bytecode = True

try:
    torch.set_float32_matmul_precision("medium")
    print("[INIT] Your system will train with low-precision kernels.")
except Exception as e:
    print(e)
    print("[INIT] Your system does not support low-precision kernels.")


def setup_environment():
    """Set up the environment and configurations."""

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
    encoder_type = processed_args.get("encoder_type")
    vocab_size = processed_args["vocab_size"]
    base_cache_dir = processed_args["cache_dir"]

    # Check for --list-runs first
    if processed_args.get("list_runs", False):
        run_manager = RunManager(base_cache_dir)
        runs = run_manager.list_runs()
        if not runs:
            print("No runs found.")
        else:
            print("\nAvailable runs:")
            for run in runs:
                status = "[CURRENT]" if run.get("is_current") else ""
                preserved = "[PRESERVED]" if run.get("preserve") else ""
                created = run.get("created", "Unknown")
                size = run.get("size_human", "Unknown")
                print(
                    f"  {run['truncated_hash']} - {size} - Created: {created} {preserved} {status}"
                )
        return 0

    # Check for --train-tokenizer (shortcut mode)
    if processed_args.get("train_tokenizer", False):
        from pathlib import Path

        from praxis.tokenizers.standard import StandardTokenizer

        # Get tokenizer training arguments
        tokenizer_type = processed_args.get("tokenizer_train_type", "unigram")
        num_examples = processed_args.get("tokenizer_num_examples", 5_000_000)
        vocab_size_for_training = processed_args.get(
            "tokenizer_train_vocab_size", 16384
        )

        # Hardcoded dataset configuration
        dataset_name = "HuggingFaceFW/fineweb"
        dataset_config = "sample-350BT"

        # Train the tokenizer
        print(
            f"Training {tokenizer_type} tokenizer with vocab_size={vocab_size_for_training}..."
        )
        print(f"Using {num_examples:,} examples from {dataset_name}")

        tokenizer = StandardTokenizer.train_from_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            num_examples=num_examples,
            vocab_size=vocab_size_for_training,
            tokenizer_type=tokenizer_type,
            dropout=0.1,
        )

        # Save the tokenizer to deterministic locations
        base_path = Path("build/tokenizers")

        # Main save path: build/tokenizers/praxis-{vocab_size}-{type}
        save_path = base_path / f"praxis-{vocab_size_for_training}-{tokenizer_type}"

        os.makedirs(save_path, exist_ok=True)

        tokenizer.save_pretrained(save_path)

        print(f"\n✓ Tokenizer saved to:")
        print(f"  - {save_path}")

        # Test chat template
        from praxis.tokenizers.train import test_chat_template, upload_to_hub

        test_chat_template(tokenizer)

        # Attempt to upload to HuggingFace Hub (gated by auth and user confirmation)
        upload_to_hub(tokenizer, vocab_size_for_training, tokenizer_type)

        return 0
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
    preserve = processed_args.get("preserve", False)

    (full_command, args_hash, truncated_hash) = log_command()

    # Initialize RunManager
    run_manager = RunManager(base_cache_dir)

    # Handle reset BEFORE setting up the run directory
    # When --reset is used with a specific configuration, we always force the reset
    # (preserve flag only protects from bulk/general resets, not explicit ones)
    if reset:
        run_manager.reset_run(truncated_hash, force=True)

    # Now set up the namespaced directory
    run_dir, is_existing_run = run_manager.setup_run(
        truncated_hash, full_command, args_hash, preserve
    )

    # Update cache_dir to point to the run-specific directory
    cache_dir = str(run_dir)

    if is_existing_run:
        print(f"[RUN] Resuming existing run: {truncated_hash}")
    else:
        print(f"[RUN] Starting new run: {truncated_hash}")

    # Set seeds for reproducibility
    seed_everything(seed, workers=True)

    # Tokenizer initialization
    tokenizer = create_tokenizer(
        vocab_size=vocab_size,
        encoder_type=encoder_type,
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
        # Gradient clipping based on trainer capabilities
        gradient_clip_val=(
            1.0
            if get_trainer_capabilities(trainer_type).supports_gradient_clipping
            else None
        ),
        gradient_clip_algorithm=(
            "norm"
            if get_trainer_capabilities(trainer_type).supports_gradient_clipping
            else None
        ),
        benchmark=True,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=not use_dashboard,
        enable_model_summary=False,
        detect_anomaly=EnvironmentFeatures.is_enabled("detect_anomaly"),
        val_check_interval=1024 * hparams["target_batch_size"] // hparams["batch_size"],
        num_sanity_val_steps=0,
        limit_val_batches=16384 // hparams["batch_size"],
        log_every_n_steps=10,
        logger=None,  # Will be set below based on integrations
        callbacks=[],
    )

    # Configure the learning rate scheduler
    warmup_steps = hparams["target_batch_size"] * 4
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

    ckpt_path = None
    # Check for force_reset feature or explicit reset flag
    if not reset and not EnvironmentFeatures.is_enabled("force_reset"):
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
        # Create dashboard first if needed (for API server logging)
        dashboard = None
        if use_dashboard:
            try:
                from praxis.interface import TerminalDashboard

                dashboard = TerminalDashboard(seed, truncated_hash)
                # Don't start it yet - TerminalInterface will handle that
            except Exception as e:
                print(f"Warning: Could not create dashboard for API logging: {e}")
                dashboard = None

        # Force reload of api integration to pick up any recent changes
        from praxis import api

        importlib.reload(api)
        from praxis.api import APIServer

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
            dev_mode=(EnvironmentFeatures.get_active_environment() == "dev"),
            dashboard=dashboard,
            launch_command=full_command,
        )
        api_server.start()

        # Call API server hooks with (host, port) as in original implementation
        for hook_func in integration_loader.get_api_server_hooks():
            hook_func(api_server.host, api_server.port)
    else:
        api_server = None

    # Run init hooks for integrations BEFORE loading datasets
    # This ensures integrations are properly initialized before their datasets are checked
    integration_loader.run_init_hooks(
        args, cache_dir, ckpt_path=ckpt_path, truncated_hash=truncated_hash
    )

    # Load datasets
    use_source_code = not no_source
    # Pass minimal_data feature flag instead of old dev variable
    dataintegration = get_datamodules(
        seed,
        EnvironmentFeatures.is_enabled("minimal_data"),
        pile,
        phi,
        use_source_code,
        tokenizer,
        hparams,
        data_path,
        rl_type,
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

    # Create progress bar callback (returns None if using dashboard)
    progress_bar = create_printing_progress_bar(
        process_position=0, leave=True, use_dashboard=use_dashboard
    )

    # Get evaluation parameters from processed_args
    eval_every = processed_args.get("eval_every", None)
    eval_tasks = processed_args.get("eval_tasks", None)
    debug = processed_args.get("debug", False)

    # Configure callbacks list - always include core callbacks
    train_params["callbacks"] = [
        SignalHandlerCallback(),  # Handle signals gracefully
        checkpoint_callback,
    ]

    # Add AccumulationSchedule only if trainer supports it
    trainer_caps = get_trainer_capabilities(trainer_type)
    if trainer_caps.supports_accumulation_schedule:
        train_params["callbacks"].append(
            AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"])
        )

    # Add evaluation callback
    train_params["callbacks"].append(
        PeriodicEvaluation(
            eval_every=eval_every,
            eval_tasks=eval_tasks,
            model=model,
            device=device,
            vocab_size=vocab_size,
            debug=debug,
        )
    )

    # Add progress bar if not using dashboard
    if progress_bar is not None:
        train_params["callbacks"].append(progress_bar)

    # Add TerminalInterface which handles dashboard/console output routing
    # TerminalInterface creates and manages the dashboard internally when use_dashboard=True
    quiet = processed_args.get("quiet", False)
    terminal_output_length = processed_args.get(
        "terminal_output_length", block_size * 2
    )

    # Create model_info dict for TerminalInterface
    model_info = {
        "optimizer_config": optimizer_config,
        "strategy": strategy,
        "rl_type": rl_type,
        "vocab_size": vocab_size,
        "depth": config.depth,
        "num_layers": config.num_layers,  # Number of layer components for controllers
        "hidden_size": config.hidden_size,
        "embed_size": config.embed_size,
        "dropout": dropout,
        "use_source_code": use_source_code,
        "dev": EnvironmentFeatures.get_active_environment() == "dev",
        "seed": seed,
        "truncated_hash": truncated_hash,
        "total_params": total_params,
        "target_batch_size": target_batch_size,
    }

    train_params["callbacks"].append(
        TerminalInterface(
            tokenizer=tokenizer,
            model_info=model_info,
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
            dashboard=dashboard if local_rank == 0 else None,  # Pass existing dashboard
        )
    )

    # Try to get logger from integrations (e.g., wandb)
    integration_logger = None
    for provider in integration_loader.get_logger_providers():
        try:
            logger = provider(
                cache_dir=cache_dir,
                ckpt_path=ckpt_path,
                truncated_hash=truncated_hash,
                wandb_enabled=getattr(args, "wandb", False),
                args=args,
            )
            if logger:
                integration_logger = logger
                print(f"[TRAIN] Using integration logger: {type(logger).__name__}")
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
        model=model,
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
        device=device,
    )

    # Show launch animation just before training begins
    show_launch_animation(model, truncated_hash)

    # Print final training configuration
    print(
        f"[TRAINING] Starting with {reduced} parameters, {optimizer_config['optimizer_name']} optimizer"
    )
    if api_server:
        print(f"[TRAINING] API available at http://{api_server.get_api_addr()}/")

    # Install a simple signal handler for post-training cleanup
    cleanup_interrupted = False
    interrupt_count = 0

    def cleanup_signal_handler(signum, frame):
        nonlocal cleanup_interrupted, interrupt_count
        interrupt_count += 1

        if interrupt_count == 1:
            cleanup_interrupted = True
            print("\n⚠️  Forcing exit...")
            # Don't wait for anything, just exit
            os._exit(130)
        else:
            # Should never get here but just in case
            os._exit(130)

    # Update LICENSE with year progress timestamp
    def update_license_timestamp():
        """Update the LICENSE file with current year progress (0-1)."""
        from datetime import datetime

        now = datetime.now()
        year_start = datetime(now.year, 1, 1)
        year_end = datetime(now.year + 1, 1, 1)

        elapsed = (now - year_start).total_seconds()
        total = (year_end - year_start).total_seconds()
        year_progress = elapsed / total

        # Read and update LICENSE file
        with open("LICENSE", "r") as f:
            lines = f.readlines()

        # Update line 3 with the new timestamp
        if len(lines) >= 3 and "Copyright (c)" in lines[2]:
            # Extract the year and replace the float
            parts = lines[2].split()
            if len(parts) >= 4:
                # Format: "Copyright (c) YEAR FLOAT\n"
                lines[2] = f"Copyright (c) {parts[2]} {year_progress}\n"

                with open("LICENSE", "w") as f:
                    f.writelines(lines)

    update_license_timestamp()

    try:
        trainer.fit(train_model, dataintegration, ckpt_path=ckpt_path)

        # Training completed successfully
        print("[TRAIN] Completed successfully")

        # Install aggressive signal handler for cleanup phase
        signal.signal(signal.SIGINT, cleanup_signal_handler)
        signal.signal(signal.SIGTERM, cleanup_signal_handler)

        # Set a flag to skip most cleanup since training is done
        cleanup_interrupted = False  # Reset flag

        return 0

    except Exception as e:
        # If we have a dashboard running, force crash it to show the error
        if (
            "progress_bar" in locals()
            and hasattr(progress_bar, "dashboard")
            and progress_bar.dashboard
        ):
            import traceback

            error_text = traceback.format_exc()
            progress_bar.dashboard.crash_with_error(error_text)
        else:
            # No dashboard, just re-raise the exception normally
            raise

    except KeyboardInterrupt:
        # Lightning handles Ctrl+C gracefully
        print("\n[TRAIN] Interrupted by user")
        # Install handler for cleanup
        signal.signal(signal.SIGINT, cleanup_signal_handler)
        signal.signal(signal.SIGTERM, cleanup_signal_handler)
        return 130  # Standard exit code for SIGINT


if __name__ == "__main__":
    sys.exit(main() or 0)
