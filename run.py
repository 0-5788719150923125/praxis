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
import os
import random
from collections import Counter
from datetime import datetime, timedelta
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
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities import disable_possible_user_warnings
from pytorch_optimizer import create_optimizer
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
    TokenMonsterConfig,
    TokenMonsterTokenizer,
)

# Register and configure environment
disable_possible_user_warnings()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = CSVLogger("logs", name="praxis")

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)
AutoTokenizer.register(TokenMonsterConfig, TokenMonsterTokenizer)

# User args, accepted via CLI
parser = argparse.ArgumentParser(description="User-supplied arguments to this script.")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use (default: cpu)",
)
parser.add_argument(
    "--port",
    type=int,
    default=5000,
    help="Serve the local API at this port (default: 5000)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Batch size to use for training (default: 1)",
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
    "--use_tokenmonster",
    action="store_true",
    default=False,
    help="Use TokenMonster (default: False)",
)
parser.add_argument(
    "--dense",
    action="store_true",
    default=False,
    help="Run as a dense model (default: False)",
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


def sample_linear_decay(max_value=2**31 - 1):
    return int(math.exp((1 - random.random())) * max_value)


def sample_cosine_decay(max_value=2**31 - 1):
    seed = random.random()
    curve = 1 - seed  # invert distribution
    return int(curve * curve * max_value)


parser.add_argument(
    "--seed",
    type=int,
    default=int(sample_cosine_decay(65536)),
    help="Global seed (default: random)",
)

args = parser.parse_args()

seed = args.seed
seed_everything(seed)

dev = args.dev
device = args.device if args.device else "cpu"
port = args.port
phi = args.phi

cache_dir = args.cache_dir
train_data_path = args.data_path

use_dashboard = False if args.no_dashboard else True

if args.use_tokenmonster:
    tokenizer_model = "englishcode-8000-consistent-nocapcode-v1"
    tokenizer_config = TokenMonsterConfig(
        vocab_file=tokenizer_model, add_bos_token=False
    )
    tokenizer = TokenMonsterTokenizer(tokenizer_config)
else:
    tokenizer_model = "data/praxis"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, cache_dir=cache_dir)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(
            "UNSAFE/praxis-8192", cache_dir=cache_dir
        )

# System args
config = PraxisConfig(
    n_dim=384,
    n_emb=512,
    n_factors=3,
    n_layer=7 if not dev else 3,
    n_head=8,
    vocab_size=tokenizer.vocab_size,
    context_length=1024,
    foresight=-1e-9,
    sparse=False if args.dense else True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    unk_token_id=tokenizer.unk_token_id,
    device_map=device,
    cache_dir=cache_dir,
)

# Training and model
hparams = dict(
    batch_size=args.batch_size if args.batch_size else 1,
    target_batch_size=64,
    block_size=512,
)

# Training data mixing
population = [
    dict(path="open-phi/textbooks", key="markdown"),
    dict(path="HuggingFaceFW/fineweb-edu", key="text", name="sample-10BT"),
    dict(path="HuggingFaceFW/fineweb-edu", key="text", name="sample-100BT"),
    dict(path="HuggingFaceFW/fineweb-edu", key="text", name="sample-350BT"),
    dict(path="HuggingFaceFW/fineweb", key="text", name="default"),
]
weights = [1, 0, 0, 0, 0] if dev else [0, 1.0, 0.666666, 0.333, 0.1]
primary_dataset = random.choices(population, weights, k=1)[0]

if phi:
    secondary_dataset = population[0]

# Misc config
max_feed_chars = 2048
save_every = 1000
save_top_k = 3

# Predictions
prompt_text = tokenizer.bos_token
predict_interval = 3
predict_tokens = 1

# Optimizer configuration
optimizer_config = dict(
    optimizer_name="AdamW",
    lr=1e-3,
    weight_decay=1e-2,
    amsgrad=True,
    wd_ban_list=[
        "bias",
        "RMSNorm.weight",
        "RMSNorm.bias",
    ],
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
    enable_progress_bar=False if use_dashboard else True,
    enable_model_summary=False,
    detect_anomaly=True if dev else False,
    logger=logger,
    enable_checkpointing=True,
    callbacks=[],
)


class PraxisTrainer(LightningModule):
    """
    A training module for Praxis.
    """

    def __init__(self, model, optimizer, hparams):
        super(PraxisTrainer, self).__init__()

        self.model, self.optimizer = (model, optimizer)
        self.batch_size = hparams["batch_size"]

        self.automatic_optimization = True

        self.save_hyperparameters(ignore=["model", "optimizer"])

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs[0]

        batch_size, _ = batch.shape

        self.log_dict(
            {
                "loss": loss,
                "batch": int(batch_idx),
                "seed": int(seed),
            },
            on_step=True,
            logger=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return [self.optimizer]


class TerminalInterface(Callback):
    """
    A single pane of glass containing charts and information.
    """

    def __init__(self, use_dashboard=False, url=None):
        super().__init__()
        self.alpha = 1e-2
        self.ema_loss = 0
        self.last_time = datetime.now()
        self.text = prompt_text
        self.max_length = max_feed_chars
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

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        loss = trainer.callback_metrics.get("loss", 0)
        self.ema_loss = self._compute_ema_loss(float(loss), self.ema_loss, self.alpha)

        self._generate_sample_text(lm, batch_idx, interval=self.interval)

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
            self.dashboard.update_validator(
                self._sign_wave(amplitude=1.23, frequency=0.01, step=batch_idx)
            )

    def _generate_sample_text(self, lm, batch_idx, interval=10):

        if not self._is_trigger_passed(self.last_time, self.interval):
            return

        lm.model.eval()

        self.text = generator.generate(self.text, {"max_new_tokens": self.num_tokens})

        while len(self.text) > self.max_length:
            self.text = self.text[1:]

        self.last_time = datetime.now()

        n_grams = 5
        frequency = 10
        if self._detect_repetition(n_grams, frequency):
            self.text = tokenizer.bos_token
            self.dashboard.update_status("[ERR]")
        elif self.dashboard:
            self.dashboard.update_status(self.text)
        else:
            print(self.text)

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
        self.key = config.get("key", "text")
        dataset_args = dict(
            path=config.get("path", "HuggingFaceFW/fineweb"),
            split="train",
            streaming=True,
            cache_dir="./tmp/pile",
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
            self.cached_text += document.get(self.key) + self.tokenizer.eos_token
            if len(self.cached_text) < text_cache_size:
                continue

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=self.block_size,
                stride=16,
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
            input_ids = input_ids.to(device)

        defaults = dict(
            do_sample=True,
            max_new_tokens=1,
            temperature=0.3,
            # eta_cutoff=0.002,
            # penalty_alpha=0.6,
            # top_k=4,
            repetition_penalty=1.35,
            suppress_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            ],  # else the model will degenerate to 100% EOS tokens
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
            print(
                "Warning: Reached maximum attempts without generating a single non-empty token"
            )

        return return_text


# Define checkpointing behavior
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=save_every,
    save_top_k=save_top_k,
    save_last="link",
    monitor="step",
    mode="max",
    dirpath=f"{cache_dir}/praxis",
    filename="model-{step}",
    enable_version_counter=False,
)

# Bootstrap the model and trainer
model = AutoModelForCausalLM.from_config(config)

ckpt_path = None
symlink = f"{cache_dir}/praxis/last.ckpt"
if os.path.exists(symlink):
    print(f"resuming from: {symlink}")
    ckpt_path = symlink

print(model)

generator = Generator(model, tokenizer)

api_server = APIServer(generator, port)
api_server.start()
api_url = api_server.get_api_addr()


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


# create the optimizer
optimizer = create_optimizer(model, **optimizer_config)


# Load a dataset
train_data = []
if train_data_path:
    train_data.append(
        MultiDirectoryDataset(tokenizer, train_data_path, hparams["block_size"])
    )
else:
    train_data.append(
        HuggingfaceDataset(tokenizer, primary_dataset, hparams["block_size"])
    )


# Load expert dataset
if phi:
    train_data.append(
        HuggingfaceDataset(tokenizer, secondary_dataset, hparams["block_size"])
    )


class WeightedIterableDataset(IterableDataset):
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
        else:
            self.weights.append(0.9)  # global
            self.weights.append(0.1)  # expert

    def train_dataloader(self):
        return WeightedIterableDataset(self.loaders, self.weights)


# Put the data onto a dataloader
dataloader = DataModule(train_data, hparams["batch_size"])

# Wrap the model in a pytorch-lightning module
train_model = PraxisTrainer(model, optimizer, hparams)

# Load the callbacks
train_params["callbacks"].append(checkpoint_callback)
train_params["callbacks"].append(
    AccumulationSchedule(hparams["batch_size"], hparams["target_batch_size"])
)
train_params["callbacks"].append(TerminalInterface(use_dashboard, api_url))

# fit the trainer and run
trainer = Trainer(**train_params)
# if args.batch_size is None:
#     print("tuning batch size...")
#     tuner = Tuner(trainer)
#     auto_batch_size = tuner.scale_batch_size(
#         train_model,
#         dataloader,
#         mode="power",
#         max_trials=5,
#         steps_per_trial=3,
#         init_val=2,
#     )
#     print(f"stopped on batch size of: {auto_batch_size}")
#     train_params["accumulate_grad_batches"] = fit_grad_accumulation(
#         auto_batch_size, hparams["target_batch_size"]
#     )

trainer.fit(train_model, dataloader.train_dataloader(), ckpt_path=ckpt_path)

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
