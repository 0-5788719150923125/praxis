import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.callbacks import Callback
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from praxis import PraxisConfig, PraxisModel, PraxisForCausalLM
import numpy as np
import random
import math
from interface import TerminalDashboard
from api import APIServer

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)


config = PraxisConfig(
    n_positions=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    device_map="cpu",
)

dataset_config = {
    "repo": "HuggingFaceFW/fineweb",
    "key": "text",
}

hparams = dict(
    learning_rate=0.001,
    weight_decay=0.0001,
    batch_size=1,
    accumulate_grad_batches=64,
    block_size=256,
)

train_params = dict(
    accelerator="cpu",
    strategy="auto",
    devices="auto",
    max_steps=10000,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=1,
    precision="32-true",
    accumulate_grad_batches=hparams[
        "accumulate_grad_batches"
    ],  # must be 1 for Hivemind training
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    enable_progress_bar=False,
    enable_model_summary=False,
    callbacks=[],
)


max_data_points = 1000

tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-hf", cache_dir="./data"
)

model = AutoModelForCausalLM.from_config(config)
model.train()

print(model)

api_server = APIServer(model)
api_server.start()
api_url = api_server.get_url() + "/generate"
# api_url = "http://localhost:8585/generate"


class PraxisTrainer(LightningModule):
    """
    A training module for Praxis.
    """

    def __init__(self, model, optimizer, hparams):
        super(PraxisTrainer, self).__init__()

        self.model, self.optimizer = (model, optimizer)

        self.automatic_optimization = True
        self.batch_size = hparams["batch_size"]
        self.save_hyperparameters(ignore=["model", "optimizer"])

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs[0]

        self.log_dict(
            {
                "loss": loss,
                "step": math.floor(batch_idx / hparams["accumulate_grad_batches"]),
            },
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return [self.optimizer]


class TerminalInterface(Callback):
    """A single pane of glass containing charts and information."""

    def __init__(self):
        super().__init__()
        self.ema = 0
        self.text = ""
        self.max_length = 2048
        self.dashboard = TerminalDashboard(max_data_points)
        self.dashboard.start()
        self.dashboard.update_url(api_url)

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        loss = trainer.callback_metrics.get("loss", 0)
        self.ema = self._compute_ema(float(loss), self.ema)
        self.dashboard.update_losses(self.ema, random.random() * 0.1)

        step = trainer.callback_metrics.get("step", 0)
        self.dashboard.update_step(step.item())

        self._generate_sample_text(lm, batch_idx)

    def _generate_sample_text(self, lm, batch_idx, interval=10):
        if batch_idx % interval != 0:
            return

        lm.model.eval()
        input_ids = tokenizer.encode(self.text, return_tensors="pt")
        outputs = lm.model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=1,
            temperature=0.7,
            eta_cutoff=0.002,
            penalty_alpha=0.6,
            top_k=4,
            repetition_penalty=1.2,
        )
        self.text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        while len(self.text) > self.max_length:
            self.text = self.text[1:]
        self.dashboard.update_status(self.text)
        lm.model.train()

    def _compute_ema(self, current_loss, prev_avg_loss, alpha=0.01):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (alpha * current_loss) + (1 - alpha) * prev_avg_loss


class HuggingfaceDataset(IterableDataset):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            config.get("repo", "HuggingFaceFW/fineweb"),
            split="train",
            streaming=True,
            cache_dir="./tmp/pile",
            trust_remote_code=True,
        )
        self.key = config.get("key", "text")

        self.cached_text = ""

    def __iter__(self):

        buffer_size = 10_000
        text_cache_size = 10 * buffer_size

        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=buffer_size,
        )

        for document in shuffled:
            self.cached_text += document.get(self.key) + self.tokenizer.eos_token
            if len(self.cached_text) < text_cache_size:
                continue

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=hparams["block_size"],
                stride=16,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )["input_ids"]

            self.cached_text = ""

            for batch in tokens:
                if len(batch) != hparams["block_size"]:
                    break
                yield batch


# set weights as trainable
def set_trainable_parameters(model, hparams):
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "RMSNorm.weight",
        "RMSNorm.bias",
    ]
    grouped_parameters = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(nd in n for nd in no_decay):
            weight_decay = 0.0
        else:
            weight_decay = hparams["weight_decay"]

        grouped_parameters.append(
            {
                "params": [p],
                "weight_decay": weight_decay,
            }
        )

    return grouped_parameters


train_params["callbacks"].append(TerminalInterface())

# set model parameters as trainable
params = set_trainable_parameters(model, hparams)

# create the optimizer
optimizer = AdamW(
    params,
    lr=0.001,
)

# Load a dataset
dataset = HuggingfaceDataset(tokenizer, dataset_config)

# Wrap the model in a pytorch-lightning module
train_model = PraxisTrainer(model, optimizer, hparams)

# fit the trainer and run
model.train()
trainer = Trainer(**train_params)
trainer.fit(train_model, dataset)

# import argparse
# import ipaddress
# import logging
# import os
# import random
# import re
# import sys
# import time
# from functools import partial
# from math import isnan

# import numpy as np
# import requests
# import torch
# from datasets import load_dataset
# from hivemind.utils.networking import log_visible_maddrs
# from lightning.fabric.utilities.seed import reset_seed, seed_everything
# from lightning.pytorch import LightningModule
# from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.core.datamodule import LightningDataModule
# from lightning.pytorch.trainer import Trainer
# from lightning_hivemind.strategy import HivemindStrategy
# from torch.optim import AdamW
# from torch.utils.data import DataLoader, Dataset, IterableDataset
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
# logger = logging.getLogger("lightning.pytorch")

# args = Configurator.combine_configs()


# def flatten_list(nested_list):
#     """Flatten a nested list."""
#     if nested_list and isinstance(nested_list[0], list):
#         # Assumes only one level of nesting
#         return [item for sublist in nested_list for item in sublist]
#     return nested_list


# # set some basic configuration values
# initial_peers = flatten_list(args.initial_peers)
# batch_size = args.batch_size
# save_every = args.save_every
# block_size = 512
# num_steps = 100_000
# target_batch_size = 8192

# dataset_config = {
#     "dataset": "tiiuae/falcon-refinedweb",
#     "key": "content",
#     "split": "train",
#     "block_size": block_size,
# }


# # wrap the LightningModule in a custom class
# class MinerTrainer(LightningModule):
#     """
#     A training module for AIGen.
#     """

#     def __init__(self, model, optimizer, hparams):
#         super(MinerTrainer, self).__init__()

#         self.model, self.optimizer = (model, optimizer)
#         self.automatic_optimization = True
#         self.save_hyperparameters(hparams)

#     def forward(self, inputs):
#         return self.model(**inputs)

#     def training_step(self, batch, batch_idx):
#         outputs = self({"input_ids": batch, "labels": batch})
#         loss = outputs[0]
#         self.log(
#             "train_loss", float(loss), on_step=True, on_epoch=False, sync_dist=True
#         )
#         return loss

#     def on_train_batch_end(self, trainer, outputs, idx):
#         self.log(
#             "local_step",
#             int(self.global_step),
#             on_step=True,
#             on_epoch=False,
#             sync_dist=True,
#         )
#         self.log(
#             "global_step",
#             int(self.trainer.strategy.optimizers[0].local_epoch),
#             on_step=True,
#             on_epoch=False,
#             sync_dist=True,
#         )

#     def configure_optimizers(self):
#         "Create optimizer and scheduler"
#         return [self.optimizer]

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
