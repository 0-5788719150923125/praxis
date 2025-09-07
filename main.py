import json
import os
import re
import signal
import subprocess
import sys

sys.dont_write_bytecode = True

# Import our new utilities
from praxis.utils import (
  check_for_updates,
  find_latest_checkpoint,
  get_memory_info,
  get_scheduler,
  initialize_lazy_modules,
  sigint_handler,
)

# Set up the SIGINT handler
signal.signal(signal.SIGINT, sigint_handler)

# Check for updates at startup
check_for_updates()

import contextlib
import itertools
import logging
import math
import random
import re
import shutil
import time
import traceback
import uuid
import warnings
from collections import Counter

# dataclass import removed - no longer needed
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from queue import Queue
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_optimizer import CosineAnnealingWarmupRestarts
from torcheval.metrics.functional import perplexity
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import (
  MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from praxis.callbacks import (
  AccumulationSchedule,
  PeriodicEvaluation,
  TerminalInterface,
  TimeBasedCheckpoint,
)
from praxis.generation import Generator
from praxis.optimizers import (
  get_optimizer,
  get_optimizer_profile,
  get_parameter_stats,
)

# Generic trainer imports
from praxis.trainers import (
  PraxisTrainer,
  Trainer,
  TrainerConfig,
  create_checkpoint_callback,
  create_logger,
  create_progress_callback,
  disable_warnings,
  reset_seed,
  seed_everything,
)

ignored_warnings = [
    ".*Checkpoint directory.*exists and is not empty*",
    ".*JAX is multithreaded, so this will likely lead to a deadlock*",
    ".*Total length of `list` across ranks is zero.*",
]
for pattern in ignored_warnings:
    warnings.filterwarnings("ignore", pattern)

from api import APIServer
from builders import get_datamodules
from cli import get_cli_args, integration_loader_with_conditions, log_command
from interface import TerminalDashboard
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# Register and configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable warnings
disable_warnings()
logging.getLogger("pytorch").setLevel(logging.ERROR)

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["praxis"] = "PraxisForCausalLM"

# Transform CLI args into global variables
globals().update(vars(get_cli_args()))

(_, args_hash, truncated_hash) = log_command()

# Set seeds for reproducibility
seed_everything(seed, workers=True)

# Set behavior flags
byte_latent = encoder_type == "byte_latent"

# Global configuration
block_size = block_size * 8 if byte_latent else block_size
terminal_output_length = block_size // 2 if byte_latent else block_size * 2
use_dashboard = False if no_dashboard else True

local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Tokenizer initialization - single unified interface
from praxis.tokenizers import create_tokenizer

tokenizer = create_tokenizer(
    tokenizer_name=tokenizer_type,
    tokenizer_profile=tokenizer_profile,
    tokenizer_path=tokenizer_path,
    encoder_type=encoder_type,
    vocab_size=vocab_size,
    cache_dir=cache_dir,
)

# Transformers config
config = PraxisConfig(
    depth=3 if dev else depth,
    num_experts=num_experts if num_experts else (3 if dev else depth),
    num_smear=num_smear,
    hidden_size=hidden_size,
    embed_size=embed_size,
    num_heads=int(num_heads.split(":")[0]),
    num_queries=int(num_heads.split(":")[1]),
    head_size=head_size,
    k_heads=k_heads,
    kv_rank=kv_rank,
    dropout=dropout,
    vocab_size=vocab_size,
    router_type=router_type,
    controller_type=controller_type,
    attention_type=attention_type,
    encoder_type=encoder_type,
    decoder_type=decoder_type,
    residual_type=residual_type,
    compression_type=compression_type,
    sorting_type=sorting_type,
    norm_type=norm_type,
    linear=linear,
    differential=differential,
    stickbreaking=stickbreaking,
    memory=memory,
    mla=mla,
    mta=mta,
    mega=mega,
    gated=gated,
    evolve=evolve,
    byte_latent=byte_latent,
    scaled=scaled,
    activation=activation,
    hivemind=hivemind,
    initial_peers=initial_peers,
    block=block_type,
    expert=expert_type,
    encoding=encoding_type,
    checkpoint_every=checkpoint_every,
    loss_func=loss_func,
    strategy=strategy,
    head_type=head_type,
    max_length=block_size * 8,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    sep_token_id=tokenizer.sep_token_id,
    device_map=device,
    cache_dir=cache_dir,
    debug=debug,
    seed=seed,
    meta=meta,
    bidirectional=bidirectional,
    tie_weights=tie_weights,
    rl_type=rl_type,
)

# Add optimizer configuration after config is created
optimizer_config, disable_schedule = get_optimizer_profile(
    optimizer, any([fixed_schedule, schedule_free])
)

# Store optimizer settings in config for decoders that need it (e.g., MonoForward)
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
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    deterministic=False,
    enable_checkpointing=True,
    enable_progress_bar=not use_dashboard,
    enable_model_summary=False,
    detect_anomaly=True if dev else False,
    val_check_interval=1024 * hparams["target_batch_size"] // hparams["batch_size"],
    num_sanity_val_steps=0,
    limit_val_batches=4096 // hparams["batch_size"],
    log_every_n_steps=10,
    logger=create_logger(
        log_dir=os.path.join(cache_dir, "logs"), name="praxis", format="csv"
    ),
    callbacks=[],
)

# Configure the learning rate scheduler
warmup_steps = 4096
if disable_schedule:

    def lr_lambda_with_warmup(current_step, warmup_steps=1024):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler_func = partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda step: lr_lambda_with_warmup(step, warmup_steps),
    )
else:

    class PatchedCosineAnnealingWarmupRestarts(CosineAnnealingWarmupRestarts):
        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            self._last_lr: List[float] = [
                group["lr"] for group in self.optimizer.param_groups
            ]

    scheduler_func = partial(
        PatchedCosineAnnealingWarmupRestarts,
        first_cycle_steps=1024 * 256,
        max_lr=optimizer_config["lr"],
        min_lr=optimizer_config["lr"] * 1e-2,
        gamma=1.0,
        warmup_steps=warmup_steps,
    )


# Define checkpointing behavior
checkpoint_callback = TimeBasedCheckpoint(
    save_top_k=3,
    save_last="link",
    monitor="batch",
    mode="max",
    dirpath=os.path.join(cache_dir, "praxis"),
    filename="model-{batch}",
    enable_version_counter=False,
    save_interval=3600,
)

# Bootstrap the model and trainer
model = AutoModelForCausalLM.from_config(config)

# Prepare for launch
plan = str(model.__repr__).splitlines()
launch_duration = random.uniform(6.7, 7.3)
acceleration_curve = random.uniform(
    3.5, 4.5
)  # Higher = more aggressive start (2=gentle, 4=moderate, 6=very aggressive)
start_time = time.time()
print(f"Staging: {truncated_hash}")
time.sleep(max(0, random.gauss(1.0, 3.0)))
for i, line in enumerate(plan):
    print(line)
    # Rolling down hill: bullet → boulder (normalized to launch_duration seconds total)
    progress = i / len(plan)  # 0 to 1
    # Normalize exponential curve to always sum to launch_duration seconds
    # Scale factor adjusts based on curve steepness: steeper curves need different normalization
    scale_factor = launch_duration * (acceleration_curve + 1) / len(plan)
    delay = scale_factor * (progress**acceleration_curve)
    time.sleep(delay)
elapsed_time = time.time() - start_time
print(f"Loaded: {truncated_hash} in {elapsed_time:.3f} seconds.")
time.sleep(2)


initialize_lazy_modules(model, device)

# Print the total parameter count
total_params = sum(p.numel() for p in model.parameters())
reduced = str(int(total_params / 10**6)) + "M"
hparams["num_params"] = reduced
print(f"parameters: {reduced}")

# Train info
print(f"optimizer: {optimizer_config['optimizer_name']}")

# File cleanup
if reset:
    directories = [
        "logs"
    ] + integration_loader_with_conditions.get_cleanup_directories()
    for directory in directories:
        shutil.rmtree(os.path.join(cache_dir, directory), ignore_errors=True)
    for checkpoint in glob(os.path.join(cache_dir, "praxis", "*.ckpt")):
        os.remove(checkpoint)


ckpt_path = None
if not reset and not dev:
    symlink = os.path.join(cache_dir, "praxis", "last.ckpt")
    true_link = find_latest_checkpoint(cache_dir)
    if os.path.exists(symlink):
        print(f"resuming from symbolic path: {symlink}")
        ckpt_path = symlink
    elif true_link is not None and os.path.exists(true_link):
        print(f"resuming from true path: {true_link}")
        ckpt_path = true_link


# Initialize loggers through integration system
class ArgsWrapper:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


args_obj = ArgsWrapper(wandb=wandb, wandb_run_name=globals().get("wandb_run_name"))

# Initialize loaded integrations with proper parameters
integration_loader_with_conditions.run_init_hooks(
    get_cli_args(), cache_dir, ckpt_path, truncated_hash
)

logger = integration_loader_with_conditions.create_logger(
    cache_dir, ckpt_path, truncated_hash, wandb_enabled=wandb, args=args_obj
)

if logger:
    train_params["logger"] = logger

generator = Generator(model, tokenizer, device=device)

# Calculate parameter statistics (simplified)
print(f"[DEBUG] Starting param_stats calculation")
try:
    param_stats = get_parameter_stats(model)
    print(f"[DEBUG] param_stats created: {param_stats}")
except Exception as e:
    print(f"[ERROR] calculating parameter stats: {e}")
    import traceback

    traceback.print_exc()
    param_stats = {}

if local_rank == 0:
    # Force reload of api integration to pick up any recent changes
    import importlib

    import api

    importlib.reload(api)
    from api import APIServer

    api_server = APIServer(
        generator,
        host_name,
        port,
        tokenizer,
        integration_loader_with_conditions,
        param_stats,
        seed,
    )
    print(f"[DEBUG] Created api_server with param_stats: {bool(param_stats)}")
    api_server.start()

    # Initialize any API server hooks from loaded modules
    # Use the ACTUAL port that the API server is using (after auto-increment)
    for hook_func in integration_loader_with_conditions.get_api_server_hooks():
        hook_func(api_server.host, api_server.port)
else:
    api_server = None


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

# Update optimizer parameter count after optimizer is created
if local_rank == 0 and param_stats:
    try:
        param_stats = get_parameter_stats(model, optimizer)
        print(f"[DEBUG] Updated param_stats with optimizer states: {param_stats}")

        # Update the API server's param_stats if it exists
        if "api_server" in locals() and hasattr(api_server, "update_param_stats"):
            if api_server:
                api_server.update_param_stats(param_stats)
                print(f"[DEBUG] Updated api_server.param_stats with optimizer info")
    except Exception as e:
        print(f"[ERROR] counting optimizer states: {e}")
        import traceback

        traceback.print_exc()

# create the scheduler
scheduler = get_scheduler(optimizer, optimizer_config, disable_schedule, warmup_steps)(
    optimizer
)

# Wrap the model in a training module
train_model = PraxisTrainer(
    model, optimizer, scheduler, hparams, tokenizer=tokenizer, byte_latent=byte_latent
)

# Load the callbacks
from praxis.callbacks import create_printing_progress_bar

progress_bar = create_printing_progress_bar(
    process_position=0, leave=True, use_dashboard=use_dashboard
)

if local_rank == 0:
    if progress_bar is not None:
        train_params["callbacks"].append(progress_bar)
    train_params["callbacks"].append(checkpoint_callback)
    # Create model info dict for dashboard display
    model_info = {
        "optimizer_config": optimizer_config,
        "strategy": strategy,
        "rl_type": rl_type,
        "vocab_size": vocab_size,
        "depth": depth,
        "hidden_size": hidden_size,
        "embed_size": embed_size,
        "dropout": dropout,
        "use_source_code": use_source_code,
        "dev": dev,
        "meta": meta,
        "seed": seed,
        "truncated_hash": truncated_hash,
        "total_params": total_params,
    }

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
            model_info=model_info,
        )
    )

train_params["callbacks"].append(
    AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"])
)

if eval_tasks:
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

# Create trainer using the specified trainer type
trainer_type = hparams.get("trainer_type", "praxis")

# Check if we should use MonoForward based on decoder type (backward compatibility)
if hparams.get("decoder_type") == "mono_forward" and trainer_type == "praxis":
    print("[INFO] Decoder type is mono_forward, automatically using mono-forward trainer")
    trainer_type = "mono-forward"

# Use the factory to create trainer with proper module wrapping
from praxis.trainers import create_trainer_with_module

trainer, train_model = create_trainer_with_module(
    trainer_type=trainer_type,
    model=train_model,
    optimizer=optimizer,
    scheduler=scheduler,
    hparams=hparams,
    tokenizer=tokenizer,
    cache_dir=cache_dir,
    ckpt_path=ckpt_path,
    trainer_params=train_params,
    encoder_type=encoder_type
)

# Wrap training in exception handler to catch crashes and display them immediately
try:
    trainer.fit(
        train_model,
        dataintegration,
        ckpt_path=ckpt_path,
    )
    
    # Training completed successfully
    print("Training completed successfully!")
    
    # Run integration cleanup hooks
    integration_loader_with_conditions.run_cleanup_hooks()
    
    # Stop API server if running
    if 'api_server' in globals() and api_server:
        print("Stopping API server...")
        api_server.stop()
    
    # Force exit to ensure all threads terminate
    import os
    os._exit(0)
    
except Exception as e:
    # Run integration cleanup hooks
    integration_loader_with_conditions.run_cleanup_hooks()

    # If we have a dashboard running, force crash it to show the error
    progress_bar = globals().get("progress_bar")
    if progress_bar and hasattr(progress_bar, "dashboard") and progress_bar.dashboard:
        import traceback

        error_text = traceback.format_exc()
        progress_bar.dashboard.crash_with_error(error_text)
    else:
        # No dashboard, just re-raise the exception normally
        raise
except KeyboardInterrupt:
    # Run integration cleanup hooks
    integration_loader_with_conditions.run_cleanup_hooks()

    # Handle Ctrl+C gracefully
    if progress_bar and hasattr(progress_bar, "dashboard") and progress_bar.dashboard:
        progress_bar.dashboard.stop()
        # Dashboard already prints the interruption message
    else:
        print("\n🛑 Training interrupted by user", file=sys.stderr)
    
    # Stop API server if running
    if 'api_server' in globals() and api_server:
        api_server.stop()
    
    # Force exit
    import os
    os._exit(0)
