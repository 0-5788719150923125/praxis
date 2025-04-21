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
from pytorch_optimizer import CosineAnnealingWarmupRestarts
from torcheval.metrics.functional import perplexity
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

# from evaluation import evaluate_model
from optimizers import get_optimizer, get_optimizer_profile

ignored_warnings = [
    ".*Checkpoint directory.*exists and is not empty*",
    ".*JAX is multithreaded, so this will likely lead to a deadlock*",
    ".*Total length of `list` across ranks is zero.*",
]
for pattern in ignored_warnings:
    warnings.filterwarnings("ignore", pattern)

from api import APIServer
from builders import get_datamodules
from cli import get_cli_args, log_command
from interface import TerminalDashboard
from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

# Register and configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
disable_possible_user_warnings()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

# Transform CLI args into global variables
globals().update(vars(get_cli_args()))
log_command()

# Set seeds for reproducibility
seed_everything(seed, workers=True)

# Global configuration
block_size = block_size * 8 if byte_latent else block_size
terminal_output_length = block_size // 2 if byte_latent else block_size * 2
use_dashboard = False if no_dashboard else True

local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Tokenizer initialization
if byte_latent:
    from praxis.tokenizer_praxis import ByteLevelTokenizer

    tokenizer = ByteLevelTokenizer()
else:
    possible_paths = [
        os.path.join(cache_dir, "praxis"),
        f"UNSAFE/praxis-{vocab_size}",
    ]
    for path in possible_paths:
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
        except Exception as e:
            logging.warn(f"No tokenizer found at: {str(path)}")


# Transformers config
config = PraxisConfig(
    depth=3 if dev else depth,
    num_experts=3 if dev else (num_experts if num_experts else depth),
    hidden_size=hidden_size,
    embed_size=embed_size,
    num_heads=int(num_heads.split(":")[0]),
    num_queries=int(num_heads.split(":")[1]),
    k_heads=k_heads,
    kv_rank=kv_rank,
    dropout=dropout,
    vocab_size=vocab_size,
    mod=mod,
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
    mta=mta,
    mega=mega,
    gated=gated,
    evolve=evolve,
    byte_latent=byte_latent,
    hyper=hyper,
    scaling=scaling,
    activation=activation,
    hivemind=hivemind,
    initial_peers=initial_peers,
    block=block_type,
    expert=expert_type,
    encoding=encoding_type,
    checkpoint_every=checkpoint_every,
    loss_func=loss_func,
    context_length=block_size * 8,
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
    batch_size=batch_size,
    target_batch_size=target_batch_size,
    block_size=block_size,
    oversample_chance=0.1,  # double the block_size
    supersample_chance=0.01,  # quadruple the block_size
    hypersample_chance=0.001,  # octuple the block_size
    **config.to_dict(),
)

# Training config
train_params = dict(
    accelerator=f"cpu" if device == "cpu" else "gpu",
    strategy="ddp_find_unused_parameters_true" if device == "cuda" else "auto",
    devices=[int(device.split(":")[1])] if device.startswith("cuda:") else "auto",
    max_steps=-1,
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
    log_every_n_steps=1,
    logger=CSVLogger(os.path.join(cache_dir, "lightning"), name="praxis"),
    callbacks=[],
)

# Optimizer configuration
optimizer_config, disable_schedule = get_optimizer_profile(
    optimizer, shuffle, any([no_schedule, schedule_free])
)

# Configure the learning rate scheduler
if disable_schedule:

    def lr_lambda_with_warmup(current_step, warmup_steps=1024):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler_func = partial(
        torch.optim.lr_scheduler.LambdaLR,
        lr_lambda=lambda step: lr_lambda_with_warmup(step),
    )
else:
    scheduler_func = partial(
        CosineAnnealingWarmupRestarts,
        first_cycle_steps=1024 * 256,
        max_lr=optimizer_config["lr"],
        min_lr=optimizer_config["lr"] * 1e-2,
        gamma=1.0,
        warmup_steps=1024,
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
        self.last_train_step_time = None
        self.train_step_ema = None
        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler"])

    def forward(self, inputs):
        return self.model(**inputs)

    def on_train_start(self):
        super().on_train_start()
        self.last_train_step_time = datetime.now()

    def training_step(self, batch, batch_idx):

        current_time = datetime.now()

        labels = batch
        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs.loss
        softmax_collapse = self._compute_softmax_collapse(outputs.logits)

        batch_size, num_tokens = batch.shape
        self.num_tokens += batch_size * num_tokens

        step_time = current_time - self.last_train_step_time
        self.train_step_ema = self._update_ema(self.train_step_ema, step_time)
        self.last_train_step_time = current_time

        self.log_dict(
            {
                "loss": loss,
                "batch": int(batch_idx),
                "learning_rate": self.scheduler.get_last_lr()[0],
                "num_tokens": self.num_tokens / 1_000_000_000,  # convert to billions
                "avg_step_time": self.train_step_ema,
                "softmax_collapse": softmax_collapse,
            },
            on_step=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        labels = batch
        outputs = self.model(input_ids=batch, labels=labels)

        stats = {}

        loss = outputs.loss
        stats["val_loss"] = loss

        if byte_latent:
            stats["val_bits_per_byte"] = self._compute_bits_per_byte(batch, loss)
        else:
            stats["val_perplexity"] = perplexity(
                outputs.logits[..., :-1, :], labels[..., 1:]
            )

        self.log_dict(
            stats,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch.size(0),
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

    def _compute_bits_per_byte(self, batch, loss):
        """
        From "Byte Latent Transformer: Patches Scale Better Than Tokens":
        https://arxiv.org/abs/2412.09871
        """
        batch_size, seq_length = batch.shape
        # Calculate number of bytes
        num_bytes = batch_size * seq_length
        # Convert mean loss back to sum loss
        sum_loss = loss * num_bytes
        # Calculate bits per byte using sum loss
        return sum_loss / (torch.log(torch.tensor(2.0)) * num_bytes)

    def _compute_softmax_collapse(self, output):
        """
        From "Grokking at the Edge of Stability".
        https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/0cc9e8dc62ce5ed66d29d80eebbaf14da2f71c67/logger.py#L154
        """
        output_off = output - output.amax(dim=1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_exp = torch.sum(exp_output, dim=-1, keepdim=True)
        log_softmax = output_off.amax(dim=1, keepdim=True) - torch.log(sum_exp)
        softmax_collapse = (sum_exp == 1).float().mean().item()
        return softmax_collapse


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
        self.initial_text = tokenizer.pad_token
        self.text = self.initial_text
        self.interval = 3
        self.url = url
        self.dashboard = use_dashboard
        self.progress_bar = progress_bar

    def on_fit_start(self, trainer, lm):
        super().on_fit_start(trainer, lm)
        lm.model.get_addr()
        # we limit the context length seen during training, to keep memory
        # usage consistent; very long sequences have a negative impact on training speed.
        self.max_length = terminal_output_length
        if self.dashboard:
            self.dashboard = TerminalDashboard(seed)
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
            val_loss = trainer.callback_metrics.get("val_loss", None)
            if val_loss is not None:
                self.dashboard.update_val(val_loss.item())
            if "fitness" in data:
                self.dashboard.update_fitness(data["fitness"])
            if "memory_churn" in data:
                self.dashboard.update_memory(data["memory_churn"])
            if "acc0" in data:
                self.dashboard.update_accuracy(data["acc0"], data["acc1"])

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

        # Chance to generate extra tokens
        while random.random() < 0.1:
            max_new_tokens += 1 if not byte_latent else self._biased_randint(1, 7)

        request_id = generator.request_generation(
            self.text,
            dict(
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                repetition_penalty=1.1,
                skip_special_tokens=False,
                truncate_to=self.max_length,
                use_cache=False,
            ),
        )
        while True:
            generator.fulfill_requests(max_requests=5)
            result = generator.get_result(request_id)
            if result is not None:
                self.text = result
                break
            time.sleep(0.1)

        n_gram_size = 13 if byte_latent else 7
        frequency = 50 if byte_latent else 20
        ignored_n_grams = [
            tokenizer.bos_token,
            tokenizer.eos_token,
            f"{tokenizer.bos_token}system",
            f"{tokenizer.bos_token}user",
            f"{tokenizer.bos_token}assistant",
        ]
        if (
            self._detect_repetition(n_gram_size, frequency)
            or self._detect_sequential_repetition(threshold=5, min_segment_length=8)
            or self._is_all_whitespace()
        ):
            self.text = self.initial_text
            if self.dashboard:
                self.dashboard.update_status(tokenizer.pad_token)
                self.dashboard.force_redraw()
        elif self.dashboard:
            self.dashboard.update_status(self.text)
        else:
            self.print(self.text)

        self.last_time = datetime.now()

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
        except Exception as e:
            import traceback

            print(traceback.format_exc())
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

        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)

        if device.startswith("cuda"):
            input_ids = input_ids.to(device)

        defaults = dict(
            do_sample=True,
            renormalize_logits=True,
            remove_invalid_values=True,
            use_cache=use_cache,
            # token_healing=True,
        )
        combined = {**defaults, **request.kwargs}

        # These values are largely an extension of the Huggingface `generate()` method, and
        # not supported by that API directly.
        if "prompt" in combined:
            del combined["prompt"]
        skip_special_tokens = True
        if "skip_special_tokens" in combined:
            if combined["skip_special_tokens"] == False:
                skip_special_tokens = False
            del combined["skip_special_tokens"]
        if "truncate_to" in combined:
            truncate_to = combined["truncate_to"]
            if input_ids.size(1) > truncate_to:
                input_ids = input_ids[:, -truncate_to:]
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
                    return_dict_in_generate=True,
                )

                # Update generated_tokens with the new token
                generated_tokens = outputs.sequences

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
                # Return the original text
                return_text = request.prompt

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

    def on_validation_end(self, trainer, pl_module):
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
    monitor="batch",
    mode="max",
    dirpath=os.path.join(cache_dir, "praxis"),
    filename="model-{batch}",
    enable_version_counter=False,
    save_interval=3600,
)

# Bootstrap the model and trainer
model = AutoModelForCausalLM.from_config(config)
print("model:", model)


def initialize_lazy_modules(model, device):
    model = model.to(device)

    # Create dummy batch for initialization
    batch_size = 2
    seq_length = 64
    dummy_input = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)

    # Run dummy batch through model to initialize lazy parameters
    with torch.no_grad():
        model(dummy_input)


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
    directories = ["lightning", "wandb"]
    for directory in directories:
        shutil.rmtree(os.path.join(cache_dir, directory), ignore_errors=True)
    for checkpoint in glob(os.path.join(cache_dir, "praxis", "*.ckpt")):
        os.remove(checkpoint)


def find_latest_checkpoint(cache_dir):
    # Construct the checkpoint directory path
    ckpt_dir = os.path.join(cache_dir, "praxis")

    # Get all checkpoint files
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    if not ckpt_files:
        return None

    # Extract batch numbers using regex
    # This will find numbers after "batch=" and before ".ckpt"
    batch_numbers = []
    for filename in ckpt_files:
        match = re.search(r"batch=(\d+)\.0\.ckpt", filename)
        if match:
            batch_numbers.append((int(match.group(1)), filename))

    if not batch_numbers:
        return None

    # Find the file with the largest batch number
    latest_batch = max(batch_numbers, key=lambda x: x[0])
    latest_checkpoint = os.path.join(ckpt_dir, latest_batch[1])

    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


ckpt_path = None
if not reset and not dev:
    symlink = os.path.join(cache_dir, "praxis", "last.ckpt")
    true_link = find_latest_checkpoint(cache_dir)
    if os.path.exists(symlink):
        print(f"resuming from symbolic path: {symlink}")
        ckpt_path = symlink
    elif os.path.exists(true_link):
        print(f"resuming from true path: {true_link}")
        ckpt_path = true_link

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

    def find_latest_wandb_run(cache_dir, cleanup_old_runs=True):
        """
        Find the latest wandb run directory, extract its ID, and clean up older run directories.
        """
        wandb_dir = os.path.join(cache_dir, "wandb")
        if not os.path.exists(wandb_dir):
            return None

        # Pattern for run directories: run-YYYYMMDD_HHMMSS-<random_id>
        run_pattern = re.compile(r"run-(\d{8}_\d{6})-([a-z0-9]+)")

        # Step 1: Find all run directories and identify the most recent one
        all_run_dirs = []
        latest_timestamp = None
        latest_run_id = None
        latest_dir_path = None

        for dirname in os.listdir(wandb_dir):
            dir_path = os.path.join(wandb_dir, dirname)
            if not os.path.isdir(dir_path):
                continue

            match = run_pattern.match(dirname)
            if match:
                timestamp_str = match.group(1)  # YYYYMMDD_HHMMSS
                run_id = match.group(2)  # random ID

                # Convert timestamp string to datetime object
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    # Store information about this run directory
                    all_run_dirs.append((dir_path, timestamp, run_id))

                    # Update if this is the most recent run
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_run_id = run_id
                        latest_dir_path = dir_path
                except ValueError:
                    continue

        # Step 2: Clean up older run directories if requested and if we have more than one directory
        if cleanup_old_runs and len(all_run_dirs) > 1 and latest_dir_path is not None:
            print(
                f"Cleaning up old wandb run directories. Keeping only: {os.path.basename(latest_dir_path)}"
            )
            for dir_path, _, _ in all_run_dirs:
                if dir_path != latest_dir_path:
                    try:
                        print(f"Removing old wandb run: {os.path.basename(dir_path)}")
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"Error removing directory {dir_path}: {e}")

        # Step 3: Now try to find the run ID using both methods
        # First check the symbolic link method
        latest_run_path = os.path.join(wandb_dir, "latest-run")
        if os.path.exists(latest_run_path):
            pattern = re.compile(r"run-([a-z0-9]+)\.wandb")
            for filename in os.listdir(latest_run_path):
                match = pattern.match(filename)
                if match:
                    print("resuming wandb from symbolic path")
                    return match.group(1)

        # If symbolic link method didn't work, use the latest run ID we found earlier
        if latest_run_id:
            print("resuming wandb from full path")
            return latest_run_id

        # If we didn't find any valid run IDs
        return None

    # Example usage:
    wandb_opts = dict(project="praxis", save_dir=cache_dir)
    if ckpt_path is not None:
        run_id = find_latest_wandb_run(cache_dir, cleanup_old_runs=True)
        if run_id is not None:
            wandb_opts["id"] = run_id
            wandb_opts["resume"] = "must"

    wandb_logger = CustomWandbLogger(**wandb_opts)

    # log gradients and model topology
    wandb_logger.watch(model, log="all", log_freq=100, log_graph=False)
    train_params["logger"] = wandb_logger

generator = Generator(model, tokenizer)

if local_rank == 0:
    api_server = APIServer(
        generator, host_name, port, tokenizer.bos_token, tokenizer.eos_token
    )
    api_server.start()


# Load datasets
datamodule = get_datamodules(
    seed, dev, pile, phi, gun, source, tokenizer, hparams, data_path
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
                print(message)

    progress_bar = PrintingProgressBar(process_position=0, leave=True)

if local_rank == 0:
    if progress_bar is not None:
        train_params["callbacks"].append(progress_bar)
    train_params["callbacks"].append(checkpoint_callback)
    train_params["callbacks"].append(
        TerminalInterface(use_dashboard, api_server.get_api_addr(), progress_bar)
    )

train_params["callbacks"].append(
    AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"])
)

# evaluate_model(model, max_samples=10)

# fit the trainer and run forever
trainer = Trainer(**train_params)
trainer.fit(
    train_model,
    datamodule,
    ckpt_path=ckpt_path,
)
