#!/usr/bin/env python3
"""Main training script for Praxis language models."""

# CRITICAL: Set multiprocessing start method before ANY imports that
# might use CUDA. Fork-based workers deadlock when CUDA is initialized
# before the fork; "spawn" avoids this by re-importing from scratch in
# each worker. The DataModule reads this setting to decide num_workers:
# spawn -> 0 workers (main process), fork -> 1 worker (child process).
# Without this, the DataLoader times out after 60 seconds on any
# CUDA-capable host.
import torch.multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
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
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# Local application imports
# APIServer is imported locally where needed to avoid starting Flask/SocketIO at import time
from praxis.callbacks import (
    AccumulationSchedule,
    BrierLMCallback,
    DynamicsLoggerCallback,
    MetricsLoggerCallback,
    PeriodicEvaluation,
    SignalHandlerCallback,
    TerminalInterface,
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
    coerce_to_list,
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


def _bos_prompt(tokenizer):
    """Build the enable-prompt for the MF live-inference hook.

    The hook seeds its streaming buffer from ``random_char_seed`` (same
    as the backprop TerminalInterface), so this value is only the
    on/off gate: a non-``None`` return enables the hook, ``None``
    disables it. Returns a ``[1, 1]`` BOS id tensor, the plain
    ``bos_token`` string as a fallback, or ``None`` when the tokenizer
    has neither. Only used by ``--trainer-type mono_forward``.
    """
    if tokenizer is None:
        return None
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        return torch.tensor([[int(bos_id)]], dtype=torch.long)
    bos_token = getattr(tokenizer, "bos_token", None)
    return bos_token if isinstance(bos_token, str) and bos_token else None


def _graceful_shutdown(
    api_server: object,
    exit_code: int = 0,
    reason: str = "training complete",
) -> None:
    """Explicit teardown before Python finalization to avoid GIL races.

    Background daemon threads (Flask/Werkzeug API server, Lightning
    dataloader workers, integration clients like Discord / Wandb) can
    still be executing C-extension code when the main thread's
    ``return`` starts Python's shutdown. If one of them calls
    ``PyGILState_Release`` on an interpreter that's already finalizing,
    the process dies with a fatal error even after a successful run -
    the symptom the user hit after a clean ``Ctrl+C``-drained training.

    Mirroring the Ray MF trainer's ``ray.shutdown`` pattern, we stop
    each known background resource with a short per-component timeout
    (so a hung subsystem can't stall the whole process) and then
    bypass normal Python finalization via ``os._exit``. Metrics DBs
    and checkpoints are already flushed inside ``trainer.fit``'s own
    ``finally`` block by the time we get here, so skipping the
    interpreter's atexit / gc pass is safe.
    """
    import threading

    print(f"[SHUTDOWN] {reason}; stopping background services...")

    def _stop_with_timeout(name: str, fn, timeout: float) -> None:
        try:
            t = threading.Thread(target=fn, daemon=True, name=f"shutdown_{name}")
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                print(
                    f"[SHUTDOWN] {name} stop timed out after {timeout:.0f}s; "
                    "continuing teardown"
                )
        except Exception as exc:
            print(f"[SHUTDOWN] {name} stop raised: {exc!r}")

    if api_server is not None:
        _stop_with_timeout("api_server", api_server.stop, 5.0)

    try:
        _stop_with_timeout("integrations", integration_loader.run_cleanup_hooks, 5.0)
    except Exception as exc:
        print(f"[SHUTDOWN] integration cleanup failed to dispatch: {exc!r}")

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # ``os._exit`` skips atexit handlers, finalizers, stdio buffers
    # (already flushed), and the module-teardown pass where GIL races
    # happen. This is the same exit-path ``cleanup_signal_handler``
    # takes on Ctrl+C during the cleanup phase; using it here means
    # the successful-completion path gets the same guarantees.
    os._exit(exit_code)


def _maybe_swap_inference_generator(trainer, tokenizer, api_server):
    """Swap the API server's generator for :class:`MonoForwardGenerator`
    when running under ``--trainer-type mono_forward``.

    The backprop flow creates a standard :class:`Generator(model, tokenizer)`
    before the trainer is known. That Generator wraps ``model.generate()``
    running on the driver's CPU copy of the model - correct for backprop,
    wrong for Mono-Forward (the trained weights live on Ray actors). This
    helper runs right after ``create_trainer_with_module`` and swaps the
    live generator reference for an MF adapter so the ``/messages`` and
    ``/input`` routes Just Work without any route-level branching.

    Non-MF trainer types early-return and the backprop path stays
    untouched.
    """
    if api_server is None:
        return
    try:
        from praxis.trainers.mono_forward import MonoForwardTrainer
    except ImportError:
        # Ray isn't installed in this environment; the factory would
        # have raised earlier if the user asked for --trainer-type
        # mono_forward, so reaching this branch means they didn't.
        return
    if not isinstance(trainer, MonoForwardTrainer):
        return

    from praxis.generation import MonoForwardGenerator
    from praxis.web.app import app

    mf_generator = MonoForwardGenerator(trainer=trainer, tokenizer=tokenizer)
    api_server.generator = mf_generator
    # ``app.config["generator"]`` and ``app.config["api_server"]`` are
    # both read by generation routes; the api_server reference was
    # updated in-place above, but ``app.config["generator"]`` is a
    # separate slot and needs the explicit assignment.
    app.config["generator"] = mf_generator
    print("[MF] Routed API server through MonoForwardGenerator")


def _is_byte_latent_encoder(encoder_type: str) -> bool:
    """Check if an encoder type is a ByteLatentEncoder or subclass."""
    try:
        from praxis.encoders import ENCODER_REGISTRY
        from praxis.encoders.byte_latent import ByteLatentEncoder

        encoder_cls = ENCODER_REGISTRY.get(encoder_type)
        if encoder_cls is None:
            return False
        actual_cls = getattr(encoder_cls, "func", encoder_cls)
        return issubclass(actual_cls, ByteLatentEncoder)
    except ImportError:
        return "byte_latent" in encoder_type


def setup_environment(no_docs: bool = False):
    """Set up the environment and configurations."""

    # Check for updates at startup
    check_for_updates()

    if not no_docs:
        try:
            from praxis.docs import regenerate_docs

            regenerate_docs()
        except Exception as e:
            print(f"[DOCS] Skipped auto-regeneration: {e}")

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
    # Get processed arguments from CLI first (so --help works immediately)
    from praxis.cli import initialize_cli

    args = get_cli_args()
    if args is None:
        # Initialize CLI if not already done (e.g., when --help is used)
        _parser, args, _loader = initialize_cli()

    processed_args = get_processed_args(args)

    # Set up environment (after arg parsing, so --help doesn't trigger heavy operations)
    setup_environment(no_docs=processed_args.get("no_docs", False))

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
    val_every = processed_args.get("val_every", 1024)
    use_dashboard = processed_args.get("use_dashboard", False)
    headless = processed_args.get("headless", False)
    reset = processed_args.get("reset", False)
    local_rank = processed_args.get("local_rank", 0)
    train_datasets_raw = processed_args.get("train_datasets") or ["base"]
    validation_datasets_raw = processed_args.get("validation_datasets") or [
        "validation"
    ]

    train_datasets = coerce_to_list(train_datasets_raw)
    validation_datasets = coerce_to_list(validation_datasets_raw)
    weighting_mode = processed_args.get("sampler_mode", "novelty")
    data_path = coerce_to_list(processed_args.get("data_path"))
    rl_type = processed_args.get("rl_type")
    no_compile = processed_args.get("no_compile", False)
    # Automatically set byte_latent if using any ByteLatent encoder variant
    encoder_type = processed_args.get("encoder_type")
    byte_latent = processed_args.get("byte_latent", False) or (
        encoder_type and _is_byte_latent_encoder(encoder_type)
    )
    host_name = processed_args.get("host_name", "localhost")
    port = processed_args.get("port", 2100)
    disable_schedule = processed_args.get("disable_schedule", False)
    strategy = processed_args.get("strategy")
    dropout = processed_args.get("dropout", 0.0)
    trainer_type = processed_args.get("trainer_type", "backpropagation")
    pipeline_depth = processed_args.get("pipeline_depth", 4)
    # Ray Mono-Forward specific flags. These are registered by
    # praxis/cli/groups/training.py and read here so the factory's
    # mono_forward branch can forward them into the trainer's
    # __init__. Non-mono_forward trainer types ignore them
    # entirely. See PROJECT_PLAN.md Phase 3 step 4 for the rationale
    # behind threading these through main.py.
    ray_address = processed_args.get("ray_address")
    ray_num_replicas_per_layer = processed_args.get("ray_num_replicas_per_layer", 1)
    ray_head_sync_every = processed_args.get("ray_head_sync_every", 50)
    ray_pipeline_api = processed_args.get("ray_pipeline_api", "manual")
    preserve = processed_args.get("preserve", False)
    num_nodes = processed_args.get("num_nodes", 1)
    node_rank = processed_args.get("node_rank", 0)
    master_addr = processed_args.get("master_addr", "localhost")
    master_port = processed_args.get("master_port", 29500)
    no_checkpoints = processed_args.get("no_checkpoints", False)
    save_every = int(processed_args.get("save_every", 256))

    # Set distributed env vars before Lightning initializes the process group.
    # We only set them when num_nodes > 1 to avoid interfering with single-node runs.
    if num_nodes > 1:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NODE_RANK"] = str(node_rank)

    full_command, args_hash, truncated_hash = log_command()

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
        tokenizer_type=processed_args.get("tokenizer_type"),
        cache_dir=cache_dir,
    )

    # Create Transformers config from CLI args. The tokenizer owns
    # vocab_size, so create_praxis_config reconciles args with the
    # tokenizer's actual size.
    config = create_praxis_config(args, tokenizer)
    vocab_size = config.vocab_size
    processed_args["vocab_size"] = vocab_size

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
    # Spread config first, then explicit params override any duplicates
    hparams = {
        **config.to_dict(),
        "batch_size": batch_size,
        "target_batch_size": target_batch_size,
        "block_size": block_size,
        "oversample_chance": 0.1,  # double the block_size
        "supersample_chance": 0.01,  # quadruple the block_size
        "hypersample_chance": 0.001,  # octuple the block_size
        "device": device,
        "trainer_type": trainer_type,
    }

    # Training config
    train_params = dict(
        accelerator=f"cpu" if device == "cpu" else "gpu",
        strategy=(
            "ddp_find_unused_parameters_true"
            if (num_nodes > 1 or device == "cuda")
            else "auto"
        ),
        num_nodes=num_nodes,
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
        enable_checkpointing=not no_checkpoints,
        enable_progress_bar=not use_dashboard and not headless,
        enable_model_summary=False,
        detect_anomaly=EnvironmentFeatures.is_enabled("detect_anomaly"),
        val_check_interval=val_every
        * hparams["target_batch_size"]
        // hparams["batch_size"],
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
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=save_every,
        save_top_k=3,
        save_last="link",
        monitor="batch",
        mode="max",
        dirpath=os.path.join(cache_dir, "model"),
        filename="model-{batch}",
        enable_version_counter=False,
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

        # Mono-Forward saves to a different path than Lightning.
        # Checked unconditionally (not just when ckpt_path is None)
        # so MF checkpoints are found even when no Lightning model/
        # directory exists.
        if ckpt_path is None:
            mf_path = os.path.join(cache_dir, "mono_forward.pt")
            if os.path.exists(mf_path):
                print(f"resuming from mono-forward checkpoint: {mf_path}")
                ckpt_path = mf_path

    # Initialize generator for tool calling during training and inference
    generator = Generator(model, tokenizer, device=device)
    param_stats = {}

    try:
        param_stats = get_parameter_stats(model)
    except Exception as e:
        param_stats = {}

    dashboard = None
    if local_rank == 0:
        # Create dashboard first if needed (for API server logging)
        if use_dashboard:
            try:
                from praxis.interface import TerminalDashboard

                dashboard = TerminalDashboard(seed, truncated_hash)
                # Don't start it yet - TerminalInterface will handle that
            except Exception as e:
                print(f"Warning: Could not create dashboard for API logging: {e}")
                dashboard = None

        # Force reload of api integration to pick up any recent changes
        from praxis import web

        importlib.reload(web)
        from praxis.web import APIServer

        # Build web frontend before starting API server
        from praxis.web.src.build import build_dev

        print("[WEB] Building frontend...")
        build_dev()
        print("[WEB] ✓ Frontend build complete")

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
            config_file=getattr(args, "config_file", None),
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
    dataintegration = get_datamodules(
        seed,
        train_datasets,
        validation_datasets,
        tokenizer,
        hparams,
        data_path,
        rl_type,
        run_dir=cache_dir,
        data_metrics_log_interval=50,
        enable_chat_validation=True,  # Always enabled
        strict_chat_validation=False,  # Warning mode (skip invalid docs)
        weighting_mode=weighting_mode,
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

    # Snapshot this run's spec so the Identity tab can inspect it later
    # even after the process has exited.
    if local_rank == 0:
        try:
            from praxis.web.spec_data import build_spec_payload, save_run_spec

            snapshot = build_spec_payload(
                generator=generator,
                truncated_hash=truncated_hash,
                full_hash=args_hash,
                param_stats=param_stats,
                command=full_command,
                timestamp=(
                    api_server.launch_timestamp if api_server is not None else None
                ),
                seed=seed,
            )
            save_run_spec(cache_dir, snapshot)
        except Exception:
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
    ]
    if not no_checkpoints:
        train_params["callbacks"].append(checkpoint_callback)

    # Add AccumulationSchedule only if trainer supports it
    trainer_caps = get_trainer_capabilities(trainer_type)
    if trainer_caps.supports_accumulation_schedule:
        train_params["callbacks"].append(
            AccumulationSchedule(
                hparams["batch_size"] * num_nodes, hparams["target_batch_size"]
            )
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

    # BrierLM (sample-based proper scoring rule) at validation time.
    # Cheap on small val batches; safe for both CALM and discrete LMs.
    # Must be registered BEFORE MetricsLoggerCallback so val_brierlm is in
    # callback_metrics by the time MetricsLogger drains them.
    train_params["callbacks"].append(BrierLMCallback(tokenizer=tokenizer))

    # Add metrics logger callback for web visualization
    # Restrict charted task-loss weights to task types a live dataset
    # produces. A learnable weighter still drifts weights for absent tasks,
    # which is pure telemetry noise; get_metrics() reads this to skip them.
    if hasattr(dataintegration, "active_task_ids"):
        model.active_task_ids = dataintegration.active_task_ids() or None

    train_params["callbacks"].append(MetricsLoggerCallback(run_dir=cache_dir))

    # Add dynamics logger callback for gradient visualization
    # Always enabled: universal per-layer gradient dynamics work without routers.
    # Expert-specific dynamics are additionally logged when routers are present.
    routers_with_gradient_logging = ["prismatic", "smear"]
    num_experts = (
        getattr(config, "num_experts", 2)
        if config.router_type in routers_with_gradient_logging
        else 0
    )
    log_freq = 10  # Log gradients every 10 steps (reduce overhead)
    print(
        f"[Setup] Adding DynamicsLoggerCallback (router_type={config.router_type}, num_experts={num_experts}, log_freq={log_freq})"
    )
    train_params["callbacks"].append(
        DynamicsLoggerCallback(
            run_dir=cache_dir,
            num_experts=num_experts,
            log_freq=log_freq,
        )
    )

    # Add progress bar if not using dashboard or headless mode
    if progress_bar is not None and not headless:
        train_params["callbacks"].append(progress_bar)

    # Add TerminalInterface which handles dashboard/console output routing
    # TerminalInterface creates and manages the dashboard internally when use_dashboard=True
    quiet = processed_args.get("quiet", False)
    infer_every = processed_args.get("infer_every", 3)
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
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "embed_size": config.embed_size,
        "dropout": dropout,
        "dev": EnvironmentFeatures.get_active_environment() == "dev",
        "seed": seed,
        "truncated_hash": truncated_hash,
        "total_params": total_params,
        "batch_size": batch_size,
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
            headless=headless,
            terminal_output_length=terminal_output_length,
            infer_every=infer_every,
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
        # Ray Mono-Forward flags - ignored by non-mono_forward
        # trainer types thanks to the factory's ``**kwargs`` absorber.
        ray_address=ray_address,
        ray_num_replicas_per_layer=ray_num_replicas_per_layer,
        ray_head_sync_every=ray_head_sync_every,
        ray_pipeline_api=ray_pipeline_api,
        # Generic settings - read by any trainer that supports them.
        # The factory merges trainer_params + kwargs into one dict for
        # the MF path; the backprop path ignores unknown kwargs via
        # Lightning's own param handling.
        inference_prompt=_bos_prompt(tokenizer),
        inference_every_seconds=infer_every,
        model_info=model_info,
        dashboard_url=api_server.get_api_addr() if api_server else None,
        accumulate_grad_batches=(
            1
            if batch_size >= target_batch_size
            else -(-target_batch_size // batch_size)
        ),
        optimizer_config=optimizer_config,
        optimizer_wrappers={
            "trac": trac,
            "ortho": ortho,
            "lookahead": lookahead,
            "schedule_free": schedule_free,
        },
        warmup_steps=warmup_steps,
        disable_schedule=disable_schedule,
        # ``val_every`` is in effective steps (same unit as
        # ``--val-every``). The trainer converts to raw batches
        # internally using ``accumulate_grad_batches``.
        val_every=val_every,
        dynamics_log_freq=10,
        save_every=save_every,
    )

    # Phase 6: if the trainer is Mono-Forward, swap the API server's
    # backprop-shaped Generator for an MF adapter that routes through
    # ``trainer.generate()`` (the Ray actor chain). The backprop path
    # is untouched: ``_maybe_swap_inference_generator`` early-returns
    # for any non-MF trainer type. See praxis/generation/mono_forward_generator.py
    # for the rationale.
    _maybe_swap_inference_generator(trainer, tokenizer, api_server)

    # Show launch animation just before training begins
    show_launch_animation(model, truncated_hash)

    # Print final training configuration
    print(
        f"[TRAINING] Starting with {reduced} parameters, {optimizer_config['optimizer_name']} optimizer"
    )
    if api_server:
        addr = api_server.get_api_addr()
        url = (
            f"{addr}/"
            if addr.startswith(("http://", "https://"))
            else f"http://{addr}/"
        )
        print(f"[TRAINING] API available at {url}")

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
        current_year = now.year
        year_start = datetime(current_year, 1, 1)
        year_end = datetime(current_year + 1, 1, 1)

        elapsed = (now - year_start).total_seconds()
        total = (year_end - year_start).total_seconds()
        year_progress = elapsed / total

        # Read and update LICENSE file
        with open("LICENSE", "r") as f:
            lines = f.readlines()

        # Update line 3 with the current year and timestamp
        if len(lines) >= 3 and "Copyright (c)" in lines[2]:
            # Format: "Copyright (c) YEAR.FRACTION\n"
            fraction = str(year_progress).split(".", 1)[1]
            lines[2] = f"Copyright (c) {current_year}.{fraction}\n"

            with open("LICENSE", "w") as f:
                f.writelines(lines)

    update_license_timestamp()

    try:
        trainer.fit(
            train_model, dataintegration, ckpt_path=ckpt_path, weights_only=False
        )

        # Training completed successfully
        print("[TRAIN] Completed successfully")

        # Swap to the aggressive signal handler before we start the
        # cleanup phase - a Ctrl+C during teardown should just exit
        # fast, not stall in another graceful path.
        signal.signal(signal.SIGINT, cleanup_signal_handler)
        signal.signal(signal.SIGTERM, cleanup_signal_handler)
        cleanup_interrupted = False  # Reset flag

        # Explicit shutdown bypassing Python finalization; prevents
        # the ``PyGILState_Release`` race we see from Flask / dataloader
        # daemon threads during interpreter teardown. Does not return.
        _graceful_shutdown(api_server, exit_code=0, reason="training complete")
        return 0  # unreachable, kept for static analysis

    except KeyboardInterrupt:
        # Lightning's own SIGINT handler usually drains the trainer
        # and returns cleanly (that path lands in the success branch
        # above). This branch runs only when the interrupt escapes
        # the trainer, e.g. during dataset setup or between runs.
        print("\n[TRAIN] Interrupted by user")
        signal.signal(signal.SIGINT, cleanup_signal_handler)
        signal.signal(signal.SIGTERM, cleanup_signal_handler)
        _graceful_shutdown(api_server, exit_code=130, reason="interrupted")
        return 130  # unreachable

    except Exception:
        # If we have a dashboard running, force crash it to show the error
        if (
            "progress_bar" in locals()
            and hasattr(progress_bar, "dashboard")
            and progress_bar.dashboard
        ):
            import traceback

            error_text = traceback.format_exc()
            progress_bar.dashboard.crash_with_error(error_text)
            # Dashboard-crash path still wants fast teardown of
            # background services to stop the GIL race; bubble the
            # exit through _graceful_shutdown so the exit code
            # reflects the failure.
            _graceful_shutdown(api_server, exit_code=1, reason="fatal error")
            return 1  # unreachable
        else:
            # No dashboard - print traceback, then explicit shutdown
            # before Python tries to clean up on its own.
            import traceback

            traceback.print_exc()
            _graceful_shutdown(api_server, exit_code=1, reason="fatal error")
            return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main() or 0)
