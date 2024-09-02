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
    callbacks=[],
)

tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-hf", cache_dir="./data"
)

model = AutoModelForCausalLM.from_config(config)
model.train()

print(model)


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
        self.max_length = 512
        self.dashboard = TerminalDashboard()
        self.dashboard.start()

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        step = trainer.callback_metrics.get("step", 0)
        # global_step = trainer.callback_metrics.get("global_step", 0)
        loss = trainer.callback_metrics.get("loss", 0)
        self.ema = self._compute_ema(float(loss), self.ema)

        self.dashboard.update_step(step.item())
        self.dashboard.update_losses(self.ema, random.random() * 0.1)
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            split="train",
            streaming=True,
            cache_dir="./tmp/pile",
            trust_remote_code=True,
        )
        self.key = "text"

        self.cached_text = ""

    def __iter__(self):

        buffer_size = 10_000
        text_cache_size = 10 * buffer_size

        block_size = hparams["block_size"]

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


dataset = HuggingfaceDataset(tokenizer)


# set weights as trainable
def set_trainable_parameters(model, hparams):
    no_decay = ["bias", "LayerNorm.weight"]
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

# # initialized and load the model
# config = AutoConfig.from_pretrained(
#     "gpt2",
#     n_embd=block_size,
#     n_ctx=block_size,
#     n_layer=2,
#     n_head=2,
#     n_positions=block_size,
#     n_inner=block_size * 4,
#     resid_pdrop=0.1,
#     embd_pdrop=0.1,
#     attn_pdrop=0.1,
#     summary_first_dropout=0.1,
#     layer_norm_epsilon=1e-5,
#     initializer_range=0.05,
#     summary_type="cls_index",
#     summary_proj_to_labels=True,
#     summary_use_proj=True,
#     torch_dtype=torch.bfloat16,
# )


# print(config)

# model = AutoModelForCausalLM.from_config(config)
# tokenizer = AutoTokenizer.from_pretrained(
#     "openai-community/gpt2",
#     cache_dir="/tmp/tokenizer",
#     padding="max_length",
#     padding_side="left",
#     use_fast=True,
#     return_overflowing_tokens=True,
#     truncation=True,
# )
# tokenizer.pad_token = tokenizer.eos_token


# # create a datamodule to wrap our remote datasets
# class StreamingDataModule(LightningDataModule):
#     def __init__(self, tokenizer, config):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.config = config
#         self.train_data = StreamingDataset(self.tokenizer, config)

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_data,
#             batch_size=batch_size,
#             pin_memory=True,
#             num_workers=2,
#         )


# # create an iterable dataset, which loops over the streaming data
# class StreamingDataset(IterableDataset):
#     def __init__(self, tokenizer, config):
#         self.tokenizer = tokenizer
#         self.config = config
#         self.dataset = load_dataset(
#             self.config.get("dataset", "tiiuae/falcon-refinedweb"),
#             split=self.config.get("split", "train"),
#             streaming=True,
#             cache_dir="/tmp/pile",
#         )

#     def __iter__(self):
#         shuffled = self.dataset.shuffle(
#             seed=random.randint(0, 2**31),
#             buffer_size=10000,
#         )

#         block_size = self.config.get("block_size", 512)

#         batch = []
#         for document in shuffled:
#             tokenized = self.tokenizer(
#                 text=document.get(self.config.get("key", "default")),
#                 max_length=block_size,
#                 stride=0,
#                 padding=True,
#                 truncation=True,
#                 return_overflowing_tokens=True,
#                 return_tensors="np",
#             )["input_ids"]
#             choice = random.choice(tokenized)
#             if len(choice) == 0:
#                 continue
#             elif len(batch) == 0:
#                 batch = choice
#             else:
#                 np.append(batch, self.tokenizer.eos_token_id)
#                 batch = np.concatenate([batch, choice])
#             if len(batch) >= block_size:
#                 yield batch[:block_size]
#                 batch = []
#             else:
#                 continue


# # prepare a dataset for use with training
# dataset = StreamingDataModule(tokenizer, dataset_config)


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


# # define the model hyperparameters
# hparams = dict(
#     learning_rate=0.001,
#     weight_decay=0.1,
#     eps=1e-8,
#     warmup_steps=10,
#     batch_size=batch_size,
#     num_steps=num_steps,
#     block_size=block_size,
# )

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

# # define training params
# train_params = dict(
#     accelerator="auto",
#     strategy=strategy,
#     devices="auto",
#     max_steps=num_steps * target_batch_size,
#     max_epochs=-1,
#     reload_dataloaders_every_n_epochs=1,
#     precision="32-true",
#     accumulate_grad_batches=1,  # must be 1 for Hivemind training
#     gradient_clip_val=1.0,
#     gradient_clip_algorithm="norm",
#     benchmark=True,
#     enable_progress_bar=False,
#     callbacks=[],
# )


# # set weights as trainable
# def set_trainable_parameters(model, hparams):
#     no_decay = ["bias", "LayerNorm.weight"]
#     grouped_parameters = []

#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue

#         if any(nd in n for nd in no_decay):
#             weight_decay = 0.0
#         else:
#             weight_decay = hparams["weight_decay"]

#         grouped_parameters.append(
#             {
#                 "params": [p],
#                 "weight_decay": weight_decay,
#             }
#         )

#     return grouped_parameters


# # set model parameters as trainable
# params = set_trainable_parameters(model, hparams)

# # create the optimizer
# optimizer = AdamW(
#     params,
#     lr=hparams.get("learning_rate", 0.001),
#     eps=hparams.get("eps", 1e-8),
# )


# # for logging progress
# class MinerConsoleLogging(Callback):
#     """A variant progress bar that works off of steps and prints periodically."""

#     def __init__(self, num_steps):
#         super().__init__()
#         self.num_steps = num_steps
#         self.num_peers = 0
#         self.previous_step = None
#         self.prev_avg_loss = None

#     def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
#         super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)
#         step = int(trainer.callback_metrics.get("global_step", -1))
#         if step == -1:
#             return

#         current_loss = float(trainer.callback_metrics["train_loss"])

#         avg_loss = 0
#         if not isnan(current_loss):
#             avg_loss = self.average_loss(current_loss, self.prev_avg_loss)
#             self.prev_avg_loss = avg_loss

#         output = f"Global Step: {str(step)}, Local Loss: {avg_loss:.3f}"

#         if hasattr(trainer.strategy, "num_peers"):
#             output += f", Peers: {trainer.strategy.num_peers}"

#         if step != self.previous_step or self.num_peers != trainer.strategy.num_peers:
#             print(output)
#             self.previous_step = step
#             self.num_peers = trainer.strategy.num_peers

#     def average_loss(self, current_loss, prev_avg_loss, smoothing=0.01):
#         if prev_avg_loss is None:
#             return current_loss
#         else:
#             return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss


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


# class ValidationCommunicator(Callback):
#     """Periodically save the model during training."""

#     def __init__(self, args, sync_interval=600, batch_to_send=100):
#         super().__init__()

#         BittensorNetwork.initialize(args)

#         # Now you can access wallet, subtensor, and metagraph like this:
#         self.wallet = BittensorNetwork.wallet
#         self.subtensor = BittensorNetwork.subtensor
#         self.metagraph = BittensorNetwork.metagraph
#         self.step = 0
#         self.sync_interval = sync_interval
#         self.last_sync_time = 0
#         self.validator_urls = []
#         self.batch_to_send = batch_to_send

#     def get_validator_uids_and_addresses(
#         self, metagraph: "bt.metagraph.Metagraph", vpermit_tao_limit: int = 1
#     ):
#         """
#         Check availability of all UIDs in a given subnet, returning their IP, port numbers, and hotkeys
#         if they are serving and have at least vpermit_tao_limit stake, along with a list of strings
#         formatted as 'ip:port' for each validator.

#         Args:
#             metagraph (bt.metagraph.Metagraph): Metagraph object.
#             vpermit_tao_limit (int): Validator permit tao limit.

#         Returns:
#             Tuple[List[dict], List[str]]: A tuple where the first element is a list of dicts with details
#                                             of available UIDs, including their IP, port, and hotkeys, and the
#                                             second element is a list of strings formatted as 'ip:port'.
#         """
#         available_uid_details = []
#         validator_addresses = []  # List to hold 'ip:port' strings
#         for uid in range(len(self.metagraph.S)):
#             if self.metagraph.S[uid] >= vpermit_tao_limit:
#                 ip = self.metagraph.axons[uid].ip
#                 port = self.metagraph.axons[uid].port
#                 details = {
#                     "uid": uid,
#                     "ip": ip,
#                     "port": port,
#                     "hotkey": self.metagraph.hotkeys[uid],
#                 }
#                 available_uid_details.append(details)
#                 validator_addresses.append(
#                     f"{ip}:{port}"
#                 )  # Format and add 'ip:port' to the list

#         return available_uid_details, validator_addresses

#     def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx, checksum=None):
#         super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

#         self.step = int(trainer.callback_metrics.get("local_step", 0)) + 1
#         if (self.step % self.batch_to_send) == 0:
#             if self.should_sync_metagraph():
#                 self.resync_metagraph()
#                 _, self.validator_urls = self.get_validator_uids_and_addresses(
#                     self.metagraph
#                 )
#             timestamp = str(int(time.time()))
#             message, signature, public_address = self.create_signed_message(timestamp)
#             self.last_sync_time = int(timestamp)

#             for url in self.validator_urls:
#                 try:
#                     response = requests.post(
#                         f"http://{url}/validate_metrics",
#                         json={
#                             "metrics": {"loss": outputs["loss"].item()},
#                             "message": message,
#                             "signature": signature,
#                             "public_address": public_address,
#                         },
#                         timeout=0.1,
#                     )
#                     if response.status_code == 200:
#                         logger.info(f"Metrics reported successfully to validator {url}")
#                     else:
#                         logger.warn(
#                             f"Error @ validator {url} --- Error: {response.json['error']}"
#                         )  # FIXME add gen
#                 except:
#                     logger.warn(
#                         f"Failed to send to validator at {url} -- Failure to communicate with validators will impact your incentive"
#                     )

#     def create_signed_message(self, message):
#         """Sign a message and return the signature."""
#         signature = self.wallet.hotkey.sign(
#             message
#         ).hex()  # Convert bytes to hex string for easy transmission
#         public_address = self.wallet.hotkey.ss58_address
#         return message, signature, public_address

#     def send_metrics(metrics, validator_urls):
#         timestamp = str(int(time.time()))
#         message, signature, public_address = create_signed_message(timestamp)
#         data = {
#             "message": message,
#             "signature": signature,
#             "public_address": public_address,
#             "metrics": metrics,
#         }
#         # Ensure metrics is a dictionary
#         if not isinstance(metrics, dict):
#             raise ValueError("Metrics must be provided as a dictionary.")
#         # Ensure validator_urls is a list
#         if not isinstance(validator_urls, list):
#             raise ValueError("validator_urls must be provided as a list.")

#     def resync_metagraph(self):
#         self.metagraph.sync(subtensor=self.subtensor)

#     def should_sync_metagraph(self):
#         """
#         Check if enough epoch blocks have elapsed since the last checkpoint to sync.
#         """
#         return (time.time() - self.last_sync_time) > self.sync_interval
#         # return (
#         #    self.block - self.metagraph.last_update[self.uid]
#         # ) > self.config.neuron.epoch_length

#     def check_registered(self):
#         # --- Check for registration.
#         if not self.subtensor.is_hotkey_registered(
#             netuid=self.config.netuid,
#             hotkey_ss58=self.wallet.hotkey.ss58_address,
#         ):
#             bt.logging.error(
#                 f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
#                 f" Please register the hotkey using `btcli subnets register` before trying again"
#             )
#             exit()


# train_params["callbacks"].append(MinerConsoleLogging(hparams.get("num_steps")))
# train_params["callbacks"].append(MinerModelSaver(save_every, "/data"))
# train_params["callbacks"].append(ValidationCommunicator(args, 60))

# # Wrap the model in a pytorch-lightning module
# train_model = MinerTrainer(model, optimizer, hparams)

# # fit the trainer and run
# model.train()
# trainer = Trainer(**train_params)
# trainer.fit(train_model, dataset)
