import os
import signal
import sys

sys.dont_write_bytecode = True


# Ensures that orphaned threads and libp2p daemons are killed when this script dies
def sigint_handler(signum, frame):
    print("\nCtrl+C detected. Killing all spawned processes.")
    # Kill the entire process group
    os.killpg(os.getpgid(0), signal.SIGTERM)
    sys.exit(1)


# Create a new process group
os.setpgrp()

# Set up the SIGINT handler
signal.signal(signal.SIGINT, sigint_handler)


import argparse
import itertools
import logging
import math
import random
import re
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import (
    Callback,
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import disable_possible_user_warnings
from pytorch_optimizer import CosineAnnealingWarmupRestarts, create_optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from api import APIServer
from interface import TerminalDashboard
from praxis import (
    PraxisConfig,
    PraxisForCausalLM,
    PraxisModel,
    PraxisTokenizer,
    PraxisTokenizerConfig,
)

# Register and configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
disable_possible_user_warnings()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)
AutoTokenizer.register(PraxisTokenizer, PraxisTokenizerConfig)

# User args, accepted via CLI
parser = argparse.ArgumentParser(description="User-supplied arguments to this script.")
parser.add_argument(
    "--seed",
    type=int,
    default=int(math.exp((1 - random.random())) * 65536),
    help="Global seed (default: random)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use (default: cpu)",
)
parser.add_argument(
    "--host_name",
    type=str,
    default="localhost",
    help="Serve the local API at this CNAME (default: 'localhost')",
)
parser.add_argument(
    "--port",
    type=int,
    default=2100,
    help="Serve the local API at this port (default: 5000)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Batch size to use for training (default: 1)",
)
parser.add_argument(
    "--depth",
    type=int,
    default=7,
    help="Number of layers to use (default: 3)",
)
parser.add_argument(
    "--data_path",
    type=str,
    nargs="+",
    default=None,
    help="Paths to directories of files to use as training data (default: None)",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    nargs="+",
    default="data",
    help="Paths to a directory where artifacts will be saved (default: ./data)",
)
parser.add_argument(
    "--no_dashboard",
    action="store_true",
    default=False,
    help="Use dashboard (default: True)",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Log experiment to Weights and Biases (Default: False)",
)
parser.add_argument(
    "--dense",
    action="store_true",
    default=False,
    help="Run as a dense model (default: False)",
)
parser.add_argument(
    "--sparse",
    action="store_true",
    default=True,
    help="Run as a sparse model (default: True)",
)
parser.add_argument(
    "--phi",
    action="store_true",
    default=False,
    help="Supplement with expert data (default: False)",
)
parser.add_argument(
    "--dev",
    action="store_true",
    default=False,
    help="Run with settings that make bootstrap faster (default: False)",
)
parser.add_argument(
    "--reset",
    action="store_true",
    default=False,
    help="Reset the checkpoint (default: False)",
)


args = parser.parse_args()

seed = args.seed
seed_everything(seed)

dev = args.dev
device = args.device if args.device else "cpu"
port = args.port
host_name = args.host_name
phi = args.phi

cache_dir = args.cache_dir
train_data_path = args.data_path

use_dashboard = False if args.no_dashboard else True


def exception_to_file(exc_type, exc_value, exc_traceback):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the error message
    error_msg = f"Error occurred at: {timestamp}\n"
    error_msg += f"Script: {os.path.abspath(sys.argv[0])}\n"
    error_msg += f"Exception Type: {exc_type.__name__}\n"
    error_msg += f"Exception Value: {exc_value}\n"
    error_msg += "Traceback:\n"
    error_msg += "".join(traceback.format_tb(exc_traceback))

    # Write to file
    with open(os.path.join(cache_dir, "error.log"), "w") as error_file:
        error_file.write(error_msg)

    # Print the full error to console
    print(error_msg)

    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = exception_to_file

# Global configuration
vocab_size = 4096

# All tokenizer initialization
# if args.use_tokenmonster:
#     tokenizer_model = "englishcode-8000-consistent-nocapcode-v1"
#     tokenizer_config = TokenMonsterConfig(
#         vocab_file=tokenizer_model, add_bos_token=False
#     )
#     tokenizer = TokenMonsterTokenizer(tokenizer_config)
# else:
tokenizer_model = os.path.join(cache_dir, "praxis")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, cache_dir=cache_dir)
except Exception as e:
    tokenizer = AutoTokenizer.from_pretrained(
        f"UNSAFE/praxis-{vocab_size}", cache_dir=cache_dir
    )

# Transformers config
config = PraxisConfig(
    n_emb=512,
    n_dim=384,
    n_layer=args.depth if not dev else 3,
    n_head=8,
    dropout=0.1,
    vocab_size=tokenizer.vocab_size,
    context_length=4096,
    sparse=False if args.dense else args.sparse,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    unk_token_id=tokenizer.unk_token_id,
    device_map=device,
    cache_dir=cache_dir,
)

# Misc hyperparameters
hparams = dict(
    seed=seed,
    batch_size=args.batch_size if args.batch_size else 1,
    target_batch_size=128,
    block_size=512,
    training_data=dict(primary=[], validation=[]),
    **config.to_dict(),
)

# Training config
train_params = dict(
    accelerator=f"cpu" if args.device == "cpu" else "gpu",
    strategy="auto",
    devices=[int(device.split(":")[1])] if args.device.startswith("cuda") else "auto",
    max_steps=-1,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=0,
    precision="32-true",
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    enable_checkpointing=True,
    enable_progress_bar=False if use_dashboard else True,
    enable_model_summary=False,
    detect_anomaly=True if dev else False,
    val_check_interval=hparams["target_batch_size"] * hparams["batch_size"] ** 2,
    limit_val_batches=1024,
    log_every_n_steps=1,
    logger=CSVLogger(os.path.join(cache_dir, "lightning"), name="praxis"),
    callbacks=[],
)

# Training data mixing
weights = [1, 0, 0, 0, 0, 0, 0] if dev else [0, 0, 0, 1, 0.666666, 0.333, 0.01]
population = [
    dict(path="open-phi/textbooks", keys=["markdown"]),
    dict(
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        keys=["prompt", "text"],
    ),
    dict(
        path="togethercomputer/RedPajama-Data-V2",
        name="sample-10B",
        snapshots=["2023-14"],
        keys=["raw_content"],
    ),
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", keys=["text"]),
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-100BT", keys=["text"]),
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-350BT", keys=["text"]),
    dict(path="HuggingFaceFW/fineweb", name="default", keys=["text"]),
]

hparams["training_data"]["primary"].append(random.choices(population, weights, k=1)[0])

if phi:
    hparams["training_data"]["primary"].append(population[0])
    hparams["training_data"]["primary"].append(population[1])

if not dev:
    hparams["training_data"]["validation"].append(population[2])

# Misc config
predict_interval = 3  # seconds
predict_tokens = 1

# Optimizer configuration
# from: https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
min_lr = 1e-6
hparams["optimizer"] = dict(
    optimizer_name="AdamW",
    lr=5e-4,
    weight_decay=1e-2,
    # num_embeds=config.n_emb,
    # num_heads=config.n_head,
    # num_query_groups=config.n_head,
    wd_ban_list=[
        "bias",
        "wte",
        "RMSNorm.weight",
        "RMSNorm.bias",
    ],
)


# class WarmupCosineLR(_LRScheduler):
#     """
#     An infinite learning rate scheduler with warmup steps and hard restarts.
#     """

#     def __init__(self, optimizer, warmup_steps, cosine_scheduler_func, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.cosine_scheduler = cosine_scheduler_func(optimizer)
#         super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < self.warmup_steps:
#             return [
#                 base_lr * (self.last_epoch / self.warmup_steps)
#                 for base_lr in self.base_lrs
#             ]
#         return self.cosine_scheduler.get_last_lr()

#     def step(self, epoch=None):
#         if self.last_epoch < self.warmup_steps:
#             super(WarmupCosineLR, self).step(epoch)
#         else:
#             if self.last_epoch == self.warmup_steps:
#                 self.cosine_scheduler.base_lrs = self.base_lrs
#             self.cosine_scheduler.step(epoch)
#         self._last_lr = self.get_lr()


# def create_warmup_cosine_scheduler(warmup_steps, T_0, T_mult, eta_min, eta_max):
#     cosine_scheduler_func = partial(
#         CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult, eta_min=eta_min
#     )

#     return partial(
#         WarmupCosineLR,
#         warmup_steps=warmup_steps,
#         cosine_scheduler_func=cosine_scheduler_func,
#     )


# Scheduler config
# scheduler_func = create_warmup_cosine_scheduler(
#     warmup_steps=512,  # Number of warmup steps
#     T_0=8192,  # Number of iterations for the first restart
#     T_mult=1,  # Multiplicative factor for T_i
#     eta_min=min_lr,  # Minimum learning rate
#     eta_max=hparams["optimizer"]["lr"],  # Maximum learning rate (after warmup)
# )
scheduler_func = partial(
    CosineAnnealingWarmupRestarts,
    first_cycle_steps=4096,
    max_lr=hparams["optimizer"]["lr"],
    min_lr=min_lr,
    warmup_steps=256,
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

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):

        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs[0]

        batch_size, num_tokens = batch.shape
        self.num_tokens += batch_size * num_tokens

        self.log_dict(
            {
                "loss": loss,
                "batch": int(batch_idx),
                "learning_rate": self.scheduler.get_lr()[0],
                "num_tokens": self.num_tokens,
            },
            on_step=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs[0]
        perplexity = torch.exp(loss)

        batch_size, _ = batch.shape

        self.log_dict(
            {
                "val_loss": loss,
                "val_perplexity": perplexity,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return {"val_loss": loss, "val_perplexity": perplexity}

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
        checkpoint["num_tokens"] = self.num_tokens

    def on_load_checkpoint(self, checkpoint):
        self.num_tokens = checkpoint.get("num_tokens", 0)


class TerminalInterface(Callback):
    """
    A single pane of glass containing charts and information.
    """

    def __init__(self, use_dashboard=False, url=None):
        super().__init__()
        self.alpha = 1e-2
        self.ema_loss = 0
        self.last_time = datetime.now()
        self.text = tokenizer.bos_token
        self.max_length = 4096
        self.interval = predict_interval
        self.num_tokens = predict_tokens
        self.dashboard = False
        if use_dashboard:
            max_data_points = 1000
            self.dashboard = TerminalDashboard(seed, max_data_points)
            try:
                self.dashboard.start()
                self.dashboard.update_seed(seed)
                self.dashboard.update_url(url)
            except KeyboardInterrupt:
                self.dashboard.stop()

    def on_train_batch_start(self, trainer, lm, batch, batch_idx):
        super().on_train_batch_start(trainer, lm, batch, batch_idx)
        if self.dashboard:
            self.dashboard.set_mode("train")

    def on_validation_start(self, trainer, lm):
        super().on_validation_start(trainer, lm)
        if self.dashboard:
            self.dashboard.set_mode("validation")

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        loss = trainer.callback_metrics.get("loss", 0)
        self.ema_loss = self._compute_ema_loss(float(loss), self.ema_loss, self.alpha)

        self._generate_sample_text(lm, batch_idx, self.interval)

        batch_size, _ = batch.shape

        self.log_dict(
            {
                "step": int(batch_idx // trainer.accumulate_grad_batches),
            },
            on_step=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        if self.dashboard:
            batch = trainer.callback_metrics.get("batch", 0)
            step = trainer.callback_metrics.get("step", 0)
            total_params = sum(p.numel() for p in lm.model.parameters())
            self.dashboard.update_params(total_params)
            self.dashboard.update_batch(batch.item())
            self.dashboard.update_step(step.item())
            self.dashboard.update_loss(self.ema_loss)
            if random.random() < 0.25:
                self.dashboard.update_validator(
                    self._sign_wave(
                        amplitude=1.0,
                        frequency=0.00333,
                        phase_shift=0.23,
                        step=batch_idx,
                    )
                )
            self.dashboard.fake_log(chance=0.000001)

    def _generate_sample_text(self, lm, batch_idx=0, interval=10):

        if not self._is_trigger_passed(self.last_time, self.interval):
            return

        lm.model.eval()

        self.text = generator.generate(self.text, {"max_new_tokens": self.num_tokens})

        while len(self.text) > self.max_length:
            self.text = self.text[1:]

        n_gram_size = 7
        frequency = 20
        if self._detect_repetition(n_gram_size, frequency) or self._is_all_whitespace():
            self.text = tokenizer.bos_token
            if self.dashboard:
                self.dashboard.update_status("[ERR]")
                self.dashboard.count()
        elif self.dashboard:
            self.dashboard.update_status(self.text)
        else:
            print(self.text)

        # allow for multiple tokens
        if random.random() < 0.1:
            return self._generate_sample_text(lm)

        self.last_time = datetime.now()

        lm.model.train()

    def _sign_wave(self, amplitude=1, frequency=1, phase_shift=0, step=1):
        distribution = random.gauss(0.25, 0.2)
        return distribution + (
            amplitude * math.sin(2 * math.pi * frequency * step + phase_shift)
        )

    def _detect_repetition(self, top_n, threshold):
        text = self.text

        # Step 1: Generate n-grams based on characters
        n_grams = [text[i : i + top_n] for i in range(len(text) - top_n + 1)]

        # Step 2: Count n-gram frequencies
        n_gram_counts = Counter(n_grams)

        # Step 3: Check if any n-gram exceeds the threshold
        for count in n_gram_counts.values():
            if count > threshold:
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


class HuggingfaceDataset(IterableDataset):
    """
    A wrapper that streams, tokenizes and batches data for training.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.keys = config.get("keys", ["text"])
        dataset_args = dict(
            path=config.get("path", "HuggingFaceFW/fineweb"),
            split="train",
            streaming=True,
            cache_dir=os.path.join(cache_dir, "datasets"),
            trust_remote_code=True,
        )

        if "name" in config:
            dataset_args["name"] = config["name"]

        self.dataset = load_dataset(**dataset_args)
        self.cached_text = ""

    def __iter__(self):

        buffer_size = 10_000
        text_cache_size = 10 * buffer_size

        shuffled = self.dataset.shuffle(
            seed=seed,
            buffer_size=buffer_size,
        )

        for document in shuffled:

            for i, key in enumerate(self.keys):

                content = document.get(key)

                if len(self.keys) > 1:
                    if i % 2 == 0:
                        content = "INPUT: " + content
                    else:
                        content = "OUTPUT: " + content + self.tokenizer.eos_token
                else:
                    content += self.tokenizer.eos_token

                self.cached_text += content

            if len(self.cached_text) < text_cache_size:
                continue

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=self.block_size,
                stride=random.randint(16, 64),
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )["input_ids"]

            self.cached_text = ""

            for batch in tokens:
                if len(batch) != self.block_size:
                    break
                yield batch


class MultiDirectoryDataset(IterableDataset):
    """
    A file-based iterable dataset that recursively reads files from multiple directories,
    tokenizes them, and returns batches for PyTorch Lightning.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, directories: List[str], block_size: int
    ):
        self.tokenizer = tokenizer
        self.directories = directories
        self.block_size = block_size
        self.cached_text = ""
        self.file_list = self._get_file_list()

    def _get_file_list(self) -> List[str]:
        """Recursively get all files in all directories."""
        file_list = []
        for directory in self.directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_list.append(os.path.join(root, file))
        return file_list

    def _read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def __iter__(self):
        buffer_size = 10_000
        text_cache_size = 10 * buffer_size
        block_size = self.block_size

        random.shuffle(self.file_list)

        for file_path in self.file_list:
            self.cached_text += self._read_file(file_path) + self.tokenizer.eos_token
            if len(self.cached_text) < text_cache_size:
                continue

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=block_size,
                stride=16,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )["input_ids"]

            self.cached_text = ""

            for batch in tokens:
                if len(batch) != block_size:
                    break
                yield batch


class Generator:
    """
    Wraps a model in a simplified generation API.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, kwargs={}):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if args.device.startswith("cuda"):
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_ids = input_ids.to(device)

        defaults = dict(
            do_sample=True,
            max_new_tokens=1,
            temperature=0.45,
            eta_cutoff=0.002,
            penalty_alpha=0.6,
            top_k=4,
            repetition_penalty=1.35,
            renormalize_logits=True,
            remove_invalid_values=True,
            suppress_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            ],  # else the model may degenerate to 100% [EOS] or [PAD] tokens
        )
        combined = {**defaults, **kwargs}
        if "prompt" in combined:
            del combined["prompt"]

        return_text = prompt
        max_attempts = 30  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            outputs = self.model.generate(input_ids, **combined)
            decoded_new = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded_new != prompt:
                return_text = decoded_new
                break
            else:
                input_ids = outputs
                attempts += 1

        if attempts == max_attempts:
            print("Warning: Reached maximum attempts without generating a valid token")

        return return_text


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
        pl_module,
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

        # return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

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
        trainer.accumulate_grad_batches = self.factor

    def _fit_grad_accumulation(self, batch_size, target_batch_size):
        return (
            1
            if batch_size >= target_batch_size
            else -(-target_batch_size // batch_size)
        )


class WeightedIterableDataset(IterableDataset):
    """
    Random sampling from multiple dataloaders with weighting.
    """

    def __init__(self, datasets, weights):
        assert len(datasets) == len(
            weights
        ), "Number of datasets and weights must match"
        assert sum(weights) == 1, "Weights must sum to 1"

        self.datasets = datasets
        self.weights = weights
        self.cumulative_weights = [sum(weights[: i + 1]) for i in range(len(weights))]

    def __iter__(self):
        iters = [iter(dataset) for dataset in self.datasets]
        while True:
            try:
                rand = random.random()
                for i, cum_weight in enumerate(self.cumulative_weights):
                    if rand < cum_weight:
                        yield next(iters[i])
                        break
            except StopIteration:
                break


class DataModule(LightningDataModule):
    def __init__(self, train_data, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.loaders = []
        for i, data in enumerate(train_data):
            self.loaders.append(
                DataLoader(
                    data,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    num_workers=1,
                )
            )

        self.weights = []
        if len(self.loaders) == 1:
            self.weights.append(1.0)
        if len(self.loaders) == 2:
            self.weights.append(0.9)  # global
            self.weights.append(0.1)  # expert
        if len(self.loaders) >= 3:
            self.weights.append(0.8)  # global
            self.weights.append(0.1)  # expert
            self.weights.append(0.1)  # expert

    def train_dataloader(self):
        return WeightedIterableDataset(self.loaders, self.weights)

    def val_dataloader(self):
        return WeightedIterableDataset(self.loaders, self.weights)


# Define checkpointing behavior
checkpoint_callback = TimeBasedCheckpoint(
    save_top_k=3,
    save_last="link",
    monitor="loss",
    mode="min",
    dirpath=os.path.join(cache_dir, "praxis"),
    filename="model-{loss:.4f}",
    enable_version_counter=False,
    save_interval=3600,
)

# Bootstrap the model and trainer
model = AutoModelForCausalLM.from_config(config)

print(model)

# Checkpoint management
if args.reset:
    for checkpoint in glob(os.path.join(cache_dir, "praxis", "*.ckpt")):
        os.remove(checkpoint)

ckpt_path = None
symlink = os.path.join(cache_dir, "praxis", "last.ckpt")
if os.path.exists(symlink):
    print(f"resuming from: {symlink}")
    ckpt_path = symlink

if args.wandb:
    import wandb
    from lightning.pytorch.loggers import WandbLogger

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

    wandb_logger = WandbLogger(**wandb_opts)

    # log gradients and model topology
    wandb_logger.watch(model, log="all", log_freq=100, log_graph=True)
    train_params["logger"] = wandb_logger

generator = Generator(model, tokenizer)

api_server = APIServer(generator, host_name, port)
api_server.start()

# Load training datasets
train_data = []
for dataset_config in hparams["training_data"]["primary"]:
    train_data.append(
        HuggingfaceDataset(tokenizer, dataset_config, hparams["block_size"])
    )

if train_data_path:
    train_data.append(
        MultiDirectoryDataset(tokenizer, train_data_path, hparams["block_size"])
    )

# Load validation data
validation_data = []
if len(hparams["training_data"]["validation"]) > 0:
    for dataset_config in hparams["training_data"]["validation"]:
        validation_data.append(
            HuggingfaceDataset(tokenizer, dataset_config, hparams["block_size"])
        )

# create the optimizer
optimizer = create_optimizer(model, **hparams["optimizer"])

# create the scheduler
scheduler = scheduler_func(optimizer)

# Put the data onto a dataloader
train_dataloader = DataModule(train_data, hparams["batch_size"])

validation_dataloader = None
if len(validation_data) > 0:
    validation_dataloader = DataModule(
        validation_data, hparams["batch_size"]
    ).val_dataloader()

# Wrap the model in a pytorch-lightning module
train_model = PraxisTrainer(model, optimizer, scheduler, hparams)

# Load the callbacks
train_params["callbacks"].append(checkpoint_callback)
train_params["callbacks"].append(
    AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"])
)
train_params["callbacks"].append(
    TerminalInterface(use_dashboard, api_server.get_api_addr())
)

# fit the trainer and run forever
trainer = Trainer(**train_params)
trainer.fit(
    train_model,
    train_dataloader.train_dataloader(),
    val_dataloaders=validation_dataloader,
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


# class MinerModelSaver(Callback):
#     """Periodically save the model during training."""

#     def __init__(
#         self,
#         save_every,
#         output_dir,
#     ):
#         super().__init__()
#         self.step = 0
#         self.last_step = 0
#         self.save_every = save_every
#         self.output_dir = output_dir

#     @property
#     def save_every_check(self):
#         return (
#             self.step > 0
#             and self.save_every > 0
#             and self.last_step != self.step
#             and self.step % self.save_every == 0
#         )

#     def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
#         super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

#         self.step = int(trainer.callback_metrics.get("global_step", 0))

#         if self.save_every_check:
#             self.save_pytorch_model(trainer, lm)

#         self.last_step = self.step

#     def save_pytorch_model(self, trainer, lm):
#         lm.model.save_pretrained(self.output_dir, safe_serialization=True)
