import os
import signal
import subprocess
import sys

sys.dont_write_bytecode = True


# Ensures that orphaned threads and libp2p daemons are killed when this script dies
def sigint_handler(signum, frame):
    print("\nCtrl+C detected. Killing all spawned processes.")
    # Kill the entire process group
    os.killpg(os.getpgid(0), signal.SIGTERM)
    sys.exit(1)


# Create a new process group
# os.setpgrp()

# Set up the SIGINT handler
signal.signal(signal.SIGINT, sigint_handler)


def check_for_updates():
    try:
        # First, fetch the latest changes from remote
        subprocess.run(["git", "fetch"], check=True, capture_output=True)

        # Try to get the current branch name
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Get commit counts ahead and behind
        status = (
            subprocess.run(
                [
                    "git",
                    "rev-list",
                    "--left-right",
                    "--count",
                    f"HEAD...origin/{branch}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .split()
        )

        commits_behind = int(status[1])

        if commits_behind > 0:
            print(
                f"\nUpdate available: Your machine is {commits_behind} commit(s) behind the Praxis git repository."
            )
            print("\nTo update, use the command: 'git pull'")
            return time.sleep(10)
        else:
            return "Up to date: Your repository is synchronized with the Praxis git repository."

    except subprocess.CalledProcessError as e:
        return f"Error checking for updates: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


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
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from queue import Queue
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import (
    Callback,
    GradientAccumulationScheduler,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import disable_possible_user_warnings
from pytorch_optimizer import CosineAnnealingWarmupRestarts, create_optimizer
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists and is not empty*")
warnings.filterwarnings(
    "ignore", ".*JAX is multithreaded, so this will likely lead to a deadlock*"
)

from api import APIServer
from builders import get_datamodules
from cli import get_cli_args
from interface import TerminalDashboard
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# Register and configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
disable_possible_user_warnings()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)


args = get_cli_args()
globals().update(vars(args))


# Ensure errors are written to a log file
def exception_to_file(exc_type, exc_value, exc_traceback):
    # Write to file
    error_path = os.path.join(cache_dir, "error.log")
    with open(error_path, "w") as error_file:
        error_file.write("".join(traceback.format_tb(exc_traceback)))
    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    print(f"Error logged to: {error_path}")


sys.excepthook = exception_to_file

# Set seeds for reproducibility
seed_everything(seed, workers=True)

# An important warning
if gun and seed and not dev:
    print(
        "WARNING: GUN data is never deterministic, and cannot be reproduced when using a `seed`. You should omit the `--gun` argument for experiments."
    )
    time.sleep(5)

# Global configuration
vocab_size = 8192
use_dashboard = False if no_dashboard else True

# Tokenizer initialization
tokenizer_model = os.path.join(cache_dir, "praxis")
if byte_latent:
    from praxis.tokenizer_praxis import ByteLevelTokenizer

    # vocab_size = 1024
    tokenizer = ByteLevelTokenizer()
else:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, cache_dir=cache_dir)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(
            f"UNSAFE/praxis-{vocab_size}", cache_dir=cache_dir
        )

# Transformers config
config = PraxisConfig(
    depth=3 if dev else depth,
    num_experts=3 if dev else (num_experts if num_experts else depth),
    dropout=0.01,
    vocab_size=tokenizer.vocab_size,
    sparse=True if sparse else not dense,
    shuffle=shuffle,
    autopilot=autopilot,
    graph=graph,
    router=router,
    attention_type=attention_type,
    linear=linear,
    differential=differential,
    stickbreaking=stickbreaking,
    compression=compression,
    memory=memory,
    mega=mega,
    evolve=evolve,
    byte_latent=byte_latent,
    activation=activation,
    hivemind=hivemind,
    initial_peers=initial_peers,
    block=block_type,
    expert=expert_type,
    encoding=encoding_type,
    strategy=strategy,
    loss_func=loss_func,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map=device,
    cache_dir=cache_dir,
    debug=debug,
    seed=seed,
    meta=meta,
)

# Misc hyperparameters
hparams = dict(
    batch_size=batch_size if batch_size else 1,
    target_batch_size=target_batch_size,
    block_size=512,
    oversample_chance=0.1,  # double the block_size
    supersample_chance=0.01,  # quadruple the block_size
    training_data=dict(primary=[], validation=[]),
    **config.to_dict(),
)

# Training config
train_params = dict(
    accelerator=f"cpu" if device == "cpu" else "gpu",
    strategy="auto",
    devices=[int(device.split(":")[1])] if device.startswith("cuda:") else "auto",
    max_steps=-1,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=0,
    precision="32-true",
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    deterministic=False,
    enable_checkpointing=True,
    enable_progress_bar=not use_dashboard,
    enable_model_summary=False,
    detect_anomaly=True if dev else False,
    val_check_interval=1024 * hparams["target_batch_size"] // hparams["batch_size"],
    num_sanity_val_steps=0,
    limit_val_batches=1024,
    log_every_n_steps=1,
    logger=CSVLogger(os.path.join(cache_dir, "lightning"), name="praxis"),
    callbacks=[],
)

# Optimizer configuration
# https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
optimizer_defaults = dict(
    wd_ban_list=[
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
    ],
)
min_lr = 0
weight_decay = 1e-9
if optimizer.lower() == "adamg":
    optimizer_profile = dict(
        optimizer_name="AdamG",
        lr=1.0,
        min_lr=1e-2,
        weight_decay=weight_decay,
        weight_decouple=True,
        p=0.5,
        q=0.24,
    )
elif optimizer.lower() == "prodigy":
    optimizer_profile = dict(
        optimizer_name="Prodigy",
        lr=1.0,
        min_lr=1e-2,
        weight_decay=weight_decay,
        weight_decouple=True,
        bias_correction=True,
        safeguard_warmup=True,
    )
elif optimizer.lower() == "soap":
    optimizer_profile = dict(
        optimizer_name="SOAP",
        lr=2e-4,
        min_lr=2e-5,
        weight_decay=weight_decay,
        precondition_frequency=10,
        max_precondition_dim=1024,
        normalize_gradient=False,
        correct_bias=True,
        precondition_1d=False,
        merge_dims=False,
    )
else:
    optimizer_profile = dict(
        optimizer_name="AdamW",
        lr=5e-4,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    min_lr = 5e-6

if shuffle:
    optimizer_profile["weight_decay"] = 0

# Merge the optimizer profile with the default profile
hparams["optimizer"] = {**optimizer_defaults, **optimizer_profile}

# Configure the learning rate scheduler
if no_schedule:
    scheduler_func = partial(
        torch.optim.lr_scheduler.ConstantLR, factor=1.0, total_iters=0
    )
else:
    scheduler_func = partial(
        CosineAnnealingWarmupRestarts,
        first_cycle_steps=4096 * 16,
        max_lr=hparams["optimizer"]["lr"],
        min_lr=hparams["optimizer"].get("min_lr", min_lr),
        gamma=1.0,
        warmup_steps=hparams["optimizer"].get("warmup_steps", 512),
    )


class PraxisTrainer(LightningModule):
    """
    A training module for Praxis.
    """

    def __init__(self, model, optimizer, scheduler, hparams):
        super(PraxisTrainer, self).__init__()
        self.model, self.optimizer, self.scheduler = (model, optimizer, scheduler)
        self.automatic_optimization = True
        self.num_tokens = 0
        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler"])
        self.last_train_step_time = None
        self.train_step_ema = None

    def forward(self, inputs):
        return self.model(**inputs)

    def on_train_start(self):
        super().on_train_start()
        self.last_train_step_time = datetime.now()

    def training_step(self, batch, batch_idx):

        current_time = datetime.now()

        current_state = self.model.get_initial_state()

        outputs = self.model(input_ids=batch, current_state=current_state, labels=batch)
        loss = outputs.loss

        batch_size, num_tokens = batch.shape
        self.num_tokens += batch_size * num_tokens

        step_time = current_time - self.last_train_step_time
        self.train_step_ema = self._update_ema(self.train_step_ema, step_time)
        self.last_train_step_time = current_time

        self.log_dict(
            {
                "loss": loss,
                "batch": int(batch_idx),
                "learning_rate": self.scheduler.get_lr()[0],
                "num_tokens": self.num_tokens,
                "avg_step_time": self.train_step_ema,
            },
            on_step=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        current_state = self.model.get_initial_state()

        outputs = self.model(input_ids=batch, current_state=current_state, labels=batch)

        stats = {}

        loss = outputs.loss
        stats["val_loss"] = loss

        batch_size, seq_length = batch.shape
        if byte_latent:
            # Calculate number of bytes
            num_bytes = batch_size * seq_length
            # Convert mean loss back to sum loss
            sum_loss = loss * num_bytes
            # Calculate bits per byte using sum loss
            bits_per_byte = sum_loss / (torch.log(torch.tensor(2.0)) * num_bytes)
            stats["val_bits_per_byte"] = bits_per_byte
        else:
            perplexity = torch.exp(loss)
            stats["val_perplexity"] = perplexity

        self.log_dict(
            stats,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

    def on_validation_end(self):
        super().on_validation_end()
        self.last_train_step_time = datetime.now()

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["num_tokens"] = self.num_tokens

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.num_tokens = checkpoint.get("num_tokens", 0)

    def _update_ema(self, ema, new_value):
        if ema is None:
            return new_value.total_seconds()
        alpha = 0.1
        return alpha * new_value.total_seconds() + (1 - alpha) * ema


class TerminalInterface(Callback):
    """
    A single pane of glass containing charts and information.
    """

    def __init__(self, use_dashboard=False, url=None, progress_bar=None):
        super().__init__()
        self.alpha = 1e-2
        self.ema_loss = 0
        self.start_time = datetime.now()
        self.last_time = datetime.now()
        self.initial_text = tokenizer.bos_token
        self.text = self.initial_text
        self.interval = 3
        self.url = url
        self.dashboard = use_dashboard
        self.progress_bar = progress_bar

    def on_fit_start(self, trainer, lm):
        super().on_fit_start(trainer, lm)
        lm.model.get_addr()
        self.max_length = int(lm.model.config.context_length * 0.9)
        if self.dashboard:
            max_data_points = 1000
            self.dashboard = TerminalDashboard(seed, max_data_points)
            try:
                self.dashboard.start()
                self.dashboard.update_seed(seed)
                self.dashboard.update_url(self.url)
            except KeyboardInterrupt:
                self.dashboard.stop()
                api_server.stop()
            self.print = print
            total_params = sum(p.numel() for p in lm.model.parameters())
            self.dashboard.update_params(total_params)
            self.dashboard.set_start_time(self.start_time)
        elif self.progress_bar is not None:
            self.print = self.progress_bar.print

    def on_train_batch_start(self, trainer, lm, batch, batch_idx):
        super().on_train_batch_start(trainer, lm, batch, batch_idx)
        if self.dashboard:
            self.dashboard.set_mode("train")

    def on_validation_start(self, trainer, lm):
        super().on_validation_start(trainer, lm)
        if self.dashboard:
            self.dashboard.set_mode("validation")

    def on_validation_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, lm, outputs, batch, batch_idx)
        if not quiet:
            self._generate_text(lm, batch_idx, self.interval)

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        loss = trainer.callback_metrics.get("loss", 0)
        self.ema_loss = self._compute_ema_loss(float(loss), self.ema_loss, self.alpha)

        if not quiet:
            self._generate_text(lm, batch_idx, self.interval)

        batch_size, _ = batch.shape
        swarm_info = lm.model.get_metrics()
        local_experts = swarm_info["experts"].get("local", 0)
        remote_experts = swarm_info["experts"].get("remote", 0)

        data = {
            "step": int(batch_idx // trainer.accumulate_grad_batches),
            "local_experts": int(local_experts),
            "remote_experts": int(remote_experts),
        }

        if "fitness" in swarm_info:
            data.update({"fitness": swarm_info["fitness"]})

        if "churn" in swarm_info:
            data.update({"memory_churn": swarm_info["churn"]})

        if "predictions" in swarm_info:
            data.update(
                {
                    "acc0": swarm_info["predictions"]["mean"],
                    "acc1": 0,
                }
            )

        self.log_dict(
            data,
            on_step=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        if self.dashboard:
            batch = trainer.callback_metrics.get("batch", 0)
            step = trainer.callback_metrics.get("step", 0)
            rate = trainer.callback_metrics.get("avg_step_time", 0)
            tokens = trainer.callback_metrics.get("num_tokens", 0)
            self.dashboard.update_batch(batch.item())
            self.dashboard.update_step(step.item())
            self.dashboard.update_rate(rate.item())
            self.dashboard.update_tokens(tokens.item())
            self.dashboard.update_loss(self.ema_loss)
            self.dashboard.update_expert_count(local_experts, remote_experts)
            if "fitness" in data:
                self.dashboard.update_fitness(data["fitness"])
            if "memory_churn" in data:
                self.dashboard.update_memory(data["memory_churn"])
            if "acc0" in data:
                self.dashboard.update_accuracy(data["acc0"], data["acc1"])
            self.dashboard.fake_log(chance=0.000002)
            if random.random() < 0.25:
                self.dashboard.update_validator(
                    self._sign_wave(
                        amplitude=1.0,
                        frequency=0.00333,
                        phase_shift=0.23,
                        step=batch_idx,
                    )
                )

    def on_save_checkpoint(self, trainer, lm, checkpoint):
        super().on_save_checkpoint(trainer, lm, checkpoint)
        checkpoint["start_time"] = self.start_time

    def on_load_checkpoint(self, trainer, lm, checkpoint):
        super().on_load_checkpoint(trainer, lm, checkpoint)
        self.start_time = checkpoint.get("start_time", datetime.now())

    def _generate_text(self, lm, batch_idx=0, interval=10):

        if not self._is_trigger_passed(self.last_time, self.interval):
            return

        max_new_tokens = 1 if not byte_latent else self._biased_randint(1, 7)
        request_id = generator.request_generation(
            self.text,
            dict(
                max_new_tokens=max_new_tokens,
                temperature=0.5,
                # min_p=0.02,
                # eta_cutoff=0.002,
                # penalty_alpha=0.6,
                # top_k=4,
                repetition_penalty=1.1,
                skip_special_tokens=False,
                truncate_to=self.max_length,
            ),
        )
        while True:
            generator.fulfill_requests(max_requests=5)
            result = generator.get_result(request_id)
            if result is not None:
                self.text = result
                break
            time.sleep(0.1)

        while len(self.text) > self.max_length:
            self.text = self.text[1:]

        n_gram_size = 7
        frequency = 20
        ignored_n_grams = [
            tokenizer.bos_token,
            tokenizer.eos_token,
            "<|im_start|> user",
            "<|im_start|> assistant",
            "<|im_end|>",
        ]
        if (
            self._detect_repetition(n_gram_size, frequency)
            or self._detect_sequential_repetition(threshold=5, min_segment_length=8)
            or self._is_all_whitespace()
        ):
            self.text = self.initial_text
            if self.dashboard:
                self.dashboard.update_status("<|err|>")
        elif self.dashboard:
            self.dashboard.update_status(self.text)
        else:
            self.print(self.text)

        # allow for multiple tokens
        if random.random() < 0.1:
            return self._generate_text(lm)

        self.last_time = datetime.now()

    def _sign_wave(self, amplitude=1, frequency=1, phase_shift=0, step=1):
        distribution = random.gauss(0.25, 0.2)
        return distribution + (
            amplitude * math.sin(2 * math.pi * frequency * step + phase_shift)
        )

    def _biased_randint(self, low, high):
        # Take average of multiple random numbers to create center bias
        # Using 3 numbers gives a nice bell curve shape
        avg = sum(random.randint(low, high) for _ in range(3)) / 3
        # Round to nearest integer since we want whole numbers
        return round(avg)

    def _detect_repetition(self, top_n, threshold, excluded_ngrams=None):
        text = self.text
        if excluded_ngrams is None:
            excluded_ngrams = set()
        else:
            excluded_ngrams = set(excluded_ngrams)  # Convert to set for O(1) lookup

        # Step 1: Generate n-grams based on characters
        n_grams = [text[i : i + top_n] for i in range(len(text) - top_n + 1)]

        # Step 2: Filter out excluded n-grams and count frequencies
        filtered_ngrams = [ng for ng in n_grams if ng not in excluded_ngrams]
        n_gram_counts = Counter(filtered_ngrams)

        # Step 3: Check if any n-gram exceeds the threshold
        for count in n_gram_counts.values():
            if count > threshold:
                return True

        return False

    def _detect_sequential_repetition(self, threshold, min_segment_length=3):
        """
        Detect unbroken/sequential repetitions of any character sequence in text,
        only if the total repeated segment exceeds a minimum length.

        Args:
            threshold (int): Number of times a sequence must repeat consecutively
            min_segment_length (int): Minimum total length of the repeated sequence
                to be considered significant. For example, with threshold=3 and
                min_segment_length=6, "ab" repeated 3 times (total length 6) would
                be detected, but " " repeated 3 times (total length 3) would not.

        Returns:
            bool: True if significant sequential repetition is detected

        Example:
            text = "   code    here" (multiple spaces)
            With min_segment_length=3: Won't detect space repetition

            text = "abcabcabcabc"
            With min_segment_length=6: Will detect as it's a longer sequence
        """
        text = self.text

        # Early return for very short texts
        if len(text) < min_segment_length:
            return False

        # Try all possible pattern lengths, from 1 up to half the text length
        max_pattern_length = len(text) // 2

        for pattern_length in range(1, max_pattern_length + 1):
            # Skip if pattern_length * threshold would be too short
            if pattern_length * threshold < min_segment_length:
                continue

            # Check each possible starting position
            for start in range(len(text) - pattern_length * threshold + 1):
                pattern = text[start : start + pattern_length]

                # Count sequential repetitions
                repeat_count = 1
                current_pos = start + pattern_length

                while (
                    current_pos + pattern_length <= len(text)
                    and text[current_pos : current_pos + pattern_length] == pattern
                ):
                    repeat_count += 1
                    current_pos += pattern_length

                    # Only return True if the total repeated segment is long enough
                    if (
                        repeat_count >= threshold
                        and pattern_length * repeat_count >= min_segment_length
                    ):
                        return True

        return False

    def _is_all_whitespace(self):
        return self.text.isspace()

    def _is_trigger_passed(self, original_time, x_seconds):
        time_difference = datetime.now() - original_time
        return time_difference > timedelta(seconds=x_seconds)

    def _compute_ema_loss(self, current_loss, prev_avg_loss, alpha=0.01):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (alpha * current_loss) + (1 - alpha) * prev_avg_loss


@dataclass
class GenerationRequest:
    id: str
    prompt: str
    kwargs: Dict[str, Any]
    result: Optional[str] = None


class Generator:
    """
    Wraps a model in a simplified generation API with request queuing.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.request_queue = Queue()
        self.results = {}

    @contextlib.contextmanager
    def _eval_mode(self):
        training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(training)

    def request_generation(self, prompt: str, kwargs={}) -> str:
        """
        Submit a generation request and return a request ID.
        """
        request_id = str(uuid.uuid4())
        request = GenerationRequest(id=request_id, prompt=prompt, kwargs=kwargs)
        self.request_queue.put(request)
        return request_id

    def get_result(self, request_id: str) -> Optional[str]:
        """
        Check if a result is ready for a given request ID.
        Returns None if the result isn't ready yet.
        """
        result = self.results.get(request_id)
        if result is not None:
            del self.results[request_id]
        return result

    def _process_single_request(self, request: GenerationRequest):
        """
        Process a single generation request.
        """
        input_ids = self.tokenizer.encode(request.prompt, return_tensors="pt")

        if device.startswith("cuda"):
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_ids = input_ids.to(device)

        defaults = dict(
            do_sample=True,
            renormalize_logits=True,
            remove_invalid_values=True,
            # token_healing=True,
        )
        combined = {**defaults, **request.kwargs}
        if "prompt" in combined:
            del combined["prompt"]
        skip_special_tokens = True
        if "skip_special_tokens" in combined:
            if combined["skip_special_tokens"] == False:
                skip_special_tokens = False
            del combined["skip_special_tokens"]
        if "truncate_to" in combined:
            if input_ids.size(1) > combined["truncate_to"]:
                input_ids = input_ids[: combined["truncate_to"]]
            del combined["truncate_to"]

        generated_tokens = input_ids

        max_attempts = 10
        attempts = 0

        with self._eval_mode():
            while attempts < max_attempts:
                outputs = self.model.generate(
                    generated_tokens,
                    **combined,
                    tokenizer=self.tokenizer,
                    # token_healing=True,
                )

                # Update generated_tokens with the new token
                generated_tokens = outputs

                # Decode the tokens generated so far
                decoded_new = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=skip_special_tokens
                )

                # Check if the decoded text contains the replacement character
                if "�" not in decoded_new:
                    # Ensure that the new text is different from the prompt
                    if decoded_new != request.prompt:
                        return_text = decoded_new
                        break
                else:
                    # The decoded text contains '�', so we need to generate more tokens
                    attempts += 1
                    generated_tokens = input_ids
                    combined["max_new_tokens"] += 1
            else:
                # print(
                #     f"Warning: Request {request.id} reached maximum attempts without generating a valid token"
                # )
                # Return the original text
                return_text = request.prompt
                # return_text = decoded_new

            return return_text

    def fulfill_requests(self, max_requests: int = None) -> int:
        """
        Process pending generation requests. Should be called from inside the training loop.
        Returns the number of requests processed.
        """
        processed = 0
        while not self.request_queue.empty():
            if max_requests is not None and processed >= max_requests:
                break

            request = self.request_queue.get()
            result = self._process_single_request(request)
            self.results[request.id] = result
            processed += 1

        return processed


class TimeBasedCheckpoint(ModelCheckpoint):
    """
    Replaces the Pytorch Lightning checkpoint behavior with one that saves on
    a time-based interval (in seconds).
    """

    def __init__(self, save_interval: int, *args, **kwargs):
        # Disable other checkpointing triggers
        kwargs["every_n_train_steps"] = 0
        kwargs["every_n_epochs"] = 0

        super().__init__(*args, **kwargs)
        self.save_interval = save_interval
        self.last_checkpoint_time = time.monotonic()

    def on_train_batch_end(
        self,
        trainer,
        lm,
        outputs,
        batch,
        batch_idx,
    ):
        # Get current time
        current_time = time.monotonic()

        # Check if save_interval has elapsed
        if current_time - self.last_checkpoint_time >= self.save_interval:

            # Get current metrics
            monitor_candidates = self._monitor_candidates(trainer)

            # Save checkpoint
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

            # Update last checkpoint time
            self.last_checkpoint_time = current_time

            # Also save the model in Huggingface format
            lm.model.save_pretrained(self.dirpath, safe_serialization=False)

    def on_train_epoch_end(self, trainer, pl_module):
        # Disable saving checkpoints at the end of every epoch
        pass


class AccumulationSchedule(GradientAccumulationScheduler):
    """
    Change gradient accumulation factor according to scheduling.
    """

    def __init__(self, batch_size=1, target_batch_size=1):
        # NOTE: must be 1 for Hivemind training; will need adapting once we get there
        self.factor = self._fit_grad_accumulation(batch_size, target_batch_size)
        self.schedule = {1: self.factor}
        super().__init__(self.schedule)

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)
        # TODO: implement cosine oscillation of accumulation size
        trainer.accumulate_grad_batches = self.factor

    def _fit_grad_accumulation(self, batch_size, target_batch_size):
        return (
            1
            if batch_size >= target_batch_size
            else -(-target_batch_size // batch_size)
        )


# Define checkpointing behavior
checkpoint_callback = TimeBasedCheckpoint(
    save_top_k=3,
    save_last="link",
    monitor="loss",
    mode="min",
    dirpath=os.path.join(cache_dir, "praxis"),
    filename="model-{loss:.4f}",
    enable_version_counter=False,
    save_interval=60,
)

# Bootstrap the model and trainer
model = AutoModelForCausalLM.from_config(config)
print("model:", model)

# Print the total parameter count
total_params = sum(p.numel() for p in model.parameters())
reduced = str(int(total_params / 10**6)) + "M"
hparams["num_params"] = reduced
print(f"parameters: {reduced}")

# File cleanup
if reset:
    directories = ["lightning", "wandb"]
    for directory in directories:
        shutil.rmtree(os.path.join(cache_dir, directory), ignore_errors=True)
    for checkpoint in glob(os.path.join(cache_dir, "praxis", "*.ckpt")):
        os.remove(checkpoint)

ckpt_path = None
symlink = os.path.join(cache_dir, "praxis", "last.ckpt")
if os.path.exists(symlink) and not dev:
    print(f"resuming from: {symlink}")
    ckpt_path = symlink

if wandb:
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    class CustomWandbLogger(WandbLogger):
        def log_hyperparams(self, params):
            # Create new dict with all non-hparams entries
            cleaned_params = {k: v for k, v in params.items() if k != "hparams"}

            # Update with contents of hparams dict if it exists
            if "hparams" in params:
                cleaned_params.update(params["hparams"])

            super().log_hyperparams(cleaned_params)

    wandb.login()

    wandb_opts = dict(project="praxis", save_dir=cache_dir)
    if ckpt_path is not None:
        pattern = re.compile(r"run-([a-z0-9]+)\.wandb")
        for filename in os.listdir(os.path.join(cache_dir, "wandb", "latest-run")):
            match = pattern.match(filename)
            if match:
                # Capture the run ID from saved file name
                wandb_opts["id"] = match.group(1)
                break

        wandb_opts["resume"] = "must"

    wandb_logger = CustomWandbLogger(**wandb_opts)

    # log gradients and model topology
    wandb_logger.watch(model, log="all", log_freq=100, log_graph=False)
    train_params["logger"] = wandb_logger

generator = Generator(model, tokenizer)

api_server = APIServer(generator, host_name, port)
api_server.start()


# Load datasets
datamodule = get_datamodules(seed, dev, phi, gun, source, tokenizer, hparams, data_path)

# create the optimizer
optimizer = create_optimizer(model, **hparams["optimizer"])

# create the scheduler
scheduler = scheduler_func(optimizer)

# Wrap the model in a pytorch-lightning module
train_model = PraxisTrainer(model, optimizer, scheduler, hparams)

# Load the callbacks
progress_bar = None
if not use_dashboard:
    from typing import Any

    from lightning.pytorch.callbacks import TQDMProgressBar

    class PrintingProgressBar(TQDMProgressBar):
        def __init__(
            self, refresh_rate: int = 1, process_position: int = 0, leave: bool = False
        ):
            super().__init__(refresh_rate, process_position, leave)
            self._last_print_lines = 0
            self._is_jupyter = self._check_jupyter()
            self._output_div = None

        def _check_jupyter(self) -> bool:
            try:
                from IPython import get_ipython

                return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
            except Exception:
                return False

        def _get_active_progress_bar(self):
            active_progress_bar = None
            if (
                self._train_progress_bar is not None
                and not self.train_progress_bar.disable
            ):
                active_progress_bar = self.train_progress_bar
            elif (
                self._val_progress_bar is not None and not self.val_progress_bar.disable
            ):
                active_progress_bar = self.val_progress_bar
            elif (
                self._test_progress_bar is not None
                and not self.test_progress_bar.disable
            ):
                active_progress_bar = self.test_progress_bar
            elif (
                self._predict_progress_bar is not None
                and not self.predict_progress_bar.disable
            ):
                active_progress_bar = self.predict_progress_bar

            return active_progress_bar

        def _escape_html(self, text: str) -> str:
            """Escape HTML special characters while preserving whitespace"""
            import html

            # First escape special characters
            escaped = html.escape(str(text))
            # Replace newlines with <br> tags to preserve formatting
            escaped = escaped.replace("\n", "<br>")
            # Replace spaces with &nbsp; to preserve multiple spaces
            escaped = escaped.replace("  ", "&nbsp;&nbsp;")
            return escaped

        def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
            active_progress_bar = self._get_active_progress_bar()
            if active_progress_bar is None:
                return

            message = sep.join(map(str, args))

            if self._is_jupyter:
                from IPython.display import HTML, display

                # Escape the message
                safe_message = self._escape_html(message)

                # Create a dedicated output area if it doesn't exist
                if self._output_div is None:
                    self._output_div = display(
                        HTML(
                            '<div id="custom-output" style="white-space: pre-wrap;"></div>'
                        ),
                        display_id=True,
                    )

                # Update the output area with the escaped message
                self._output_div.update(
                    HTML(
                        f'<div id="custom-output" style="white-space: pre-wrap;">{safe_message}</div>'
                    )
                )
            else:
                # Terminal version - keep original ANSI approach
                if not message.endswith("\n"):
                    message += "\n"
                new_lines = message.count("\n")
                clear_sequence = (
                    "\033[F\033[K" * self._last_print_lines
                    if self._last_print_lines > 0
                    else ""
                )
                active_progress_bar.write(
                    clear_sequence + message.rstrip("\n"), end="\n"
                )
                self._last_print_lines = new_lines

    progress_bar = PrintingProgressBar(process_position=0, leave=True)

current_rank = int(os.environ.get("LOCAL_RANK", 0))
if current_rank == 0:
    if progress_bar is not None:
        train_params["callbacks"].append(progress_bar)
    train_params["callbacks"].append(checkpoint_callback)
    train_params["callbacks"].append(
        AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"])
    )
    train_params["callbacks"].append(
        TerminalInterface(use_dashboard, api_server.get_api_addr(), progress_bar)
    )

# fit the trainer and run forever
trainer = Trainer(**train_params)
trainer.fit(
    train_model,
    datamodule,
    ckpt_path=ckpt_path,
)

# import ipaddress
# from functools import partial
# from hivemind.utils.networking import log_visible_maddrs
# from lightning.fabric.utilities.seed import reset_seed, seed_everything
# from lightning_hivemind.strategy import HivemindStrategy


# # set some basic configuration values
# initial_peers = flatten_list(args.initial_peers)
# target_batch_size = 8192

# # define the hivemind strategy
# strategy = HivemindStrategy(
#     run_id=f"hiveminer",
#     batch_size=batch_size,
#     target_batch_size=target_batch_size,
#     initial_peers=initial_peers,
#     use_ipfs=False,
#     use_relay=True,
#     use_auto_relay=True,
#     verbose=False,
#     wait_timeout=60,
#     bootstrap_timeout=45,
#     matchmaking_time=90.0,
#     averaging_timeout=300.0,
#     delay_state_averaging=True,
#     delay_grad_averaging=True,
#     delay_optimizer_step=True,
#     offload_optimizer=True,
#     reuse_grad_buffers=False,
#     # grad_compression=Float16Compression(),
#     # state_averaging_compression=Float16Compression(),
#     # load_state_compression=NoCompression(),
#     # scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9999),
# )

# # print my peer id to console
# visible_addresses = [
#     str(a)
#     for a in strategy.dht.get_visible_maddrs()
#     if not ipaddress.ip_address(a.values()[0]).is_loopback
# ]

# log_visible_maddrs(strategy.dht.get_visible_maddrs(), only_p2p=False)
# # my_ids = []
# # pattern = r"(/p2p/.*)"
# # for peer in list(visible_addresses):
# #     match = re.search(pattern, peer)
# #     if match:
# #         my_ids.append(match.group(1))

# # for peer in list(set(my_ids)):
# #     print(f"PEER-ID: {peer}")
