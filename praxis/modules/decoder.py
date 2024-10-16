from typing import Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import hivemind
from hivemind import DHT
from hivemind.moe import ModuleBackend, RemoteExpert, Server, get_experts
from hivemind.moe.server.layers import name_to_block
from hivemind.utils import BatchTensorDescriptor

from praxis import PraxisConfig
from praxis.modules.experts import PraxisBlock
from praxis.modules.router import PraxisMixtureOfDepths
import asyncio
import os
from pathlib import Path
import time


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire decoder block (and all intermediate layers)
    in a single class.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.shuffle = config.shuffle
        self.checkpoint_layers = self._checkpoint_strategy(
            config.memory_profile, config.num_layers
        )
        self.experts = nn.ModuleList()
        if config.hivemind:
            self.dht = DHT(
                initial_peers=config.initial_peers,
                start=True,
                use_auto_relay=True,
                use_relay=True,
                use_ipfs=True,
                ensure_bootstrap_success=True,
                daemon=True,
                identity_path=os.path.join(os.getcwd(), "id.key"),
            )
            schema = BatchTensorDescriptor(
                config.num_dims,
            )
            self.backends = {}
            for i in range(config.num_layers):
                expert = ModuleBackend(
                    name=f"expert.{i}",
                    module=name_to_block["praxis_block"](config),
                    args_schema=(schema,),
                    outputs_schema=schema,
                    max_batch_size=64,  # should match the `target_batch_size`
                    start=True,
                )
                self.backends[f"expert.{i}"] = expert
                self.experts.append(expert.module)
            # directory = Path(os.path.join("data/praxis", "experts"))
            # os.makedirs(directory, exist_ok=True)
            server = Server(
                self.dht,
                self.backends,
                num_connection_handlers=4 * config.num_layers,
                device=config.device_map,
                # checkpoint_dir=directory,
            )

            while not server.runtime.ready:
                server.run_in_background(timeout=5.0)
                server.runtime.clear()

        else:
            [self.experts.append(PraxisBlock(config)) for _ in range(config.num_layers)]

        self.routers = (
            nn.ModuleList(
                PraxisMixtureOfDepths(config) for _ in range(config.num_layers // 2)
            )
            if config.sparse
            else None
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        if self.shuffle:
            random.shuffle(self.experts)

        hidden_states = inputs
        aux_losses = []

        for i, expert in enumerate(self.experts):
            router = (
                self.routers[(i - 1) // 2] if self.routers and i % 2 != 0 else None
            )  # select odd layers
            gradient_checkpointing = True if i in self.checkpoint_layers else False
            hidden_states, aux_loss = self._create_forward(
                expert, router, hidden_states, attention_mask, gradient_checkpointing
            )
            aux_losses.append(aux_loss)

        return hidden_states, sum(aux_losses)

    def _checkpoint_strategy(self, strategy="speed", num_layers=0):
        if strategy == "aggressive":
            # every layer
            return [i for i in range(num_layers)]
        elif strategy == "balanced":
            # every fourth layer
            return [i for i in range(num_layers) if i % 4 == 0]
        else:
            # no gradient checkpointing
            return []

    def _create_forward(
        self,
        expert: nn.Module,
        router: Optional[nn.Module],
        hidden_states: Tensor,
        attention_mask: Tensor,
        gradient_checkpointing=False,
    ):
        def custom_forward(*inputs):
            return router(expert, *inputs) if router else expert(*inputs)

        if gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, use_reentrant=False
            )
        else:
            return custom_forward(hidden_states, attention_mask)
