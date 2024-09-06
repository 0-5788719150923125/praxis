import random

import hivemind
import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.moe import Server
from hivemind.moe.client.expert import RemoteExpert
from hivemind.moe.server import (
    ModuleBackend,
    Server,
    background_server,
    declare_experts,
    get_experts,
)
from hivemind.moe.server.layers import name_to_block
from hivemind.utils import BatchTensorDescriptor

from .attention import PraxisAttention
from .mlp import PraxisMLP


class PraxisBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = PraxisAttention(config)

        self.n_experts = 3
        self.k_best = 2

        experts = {}
        for i in range(self.n_experts):
            expert = name_to_block["praxis_mlp"](config)
            experts[f"expert.{i}"] = ModuleBackend(
                name=f"expert.{i}",
                module=expert,
                args_schema=(
                    BatchTensorDescriptor(
                        config.n_embd,
                    ),
                ),
                outputs_schema=BatchTensorDescriptor(
                    config.n_embd,
                ),
                max_batch_size=16,
            )

        relay = DHTSingleton.get_instance()

        server = Server(
            relay.get_dht(),
            experts,
            num_connection_handlers=1,
        )
        server.start()
        server.ready.wait()

        self.dht = DHT(
            initial_peers=relay.get_visible_maddrs(),
            start=True,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
        )
        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.experts = get_experts(
            self.dht, [f"expert.{i}" for i in range(self.n_experts)]
        )

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp_norm(x)

        # expert handling
        x_cpu = x.to("cpu")

        # Collect outputs from selected experts
        outputs = []
        selections = random.sample(self.experts, min(self.k_best, len(self.experts)))
        for expert in selections:
            outputs.append(expert(x_cpu).to(x.device))

        # Stack outputs along a new dimension
        x = torch.stack(outputs)

        # Compute the mean along the expert dimension
        x = torch.mean(x, dim=0)

        x = residual + x
        balancing_loss = 0  # dummy loss
        return x, balancing_loss


# PUBLIC_INITIAL_PEERS = [
#     # IPv4 DNS addresses
#     "/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
#     "/dns/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
#     # IPv6 DNS addresses
#     "/dns6/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
#     "/dns6/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
#     # Reserved IPs
#     "/ip4/159.89.214.152/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
#     "/ip4/159.203.156.48/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
# ]


class DHTSingleton:
    """
    Ensures that we only initialize the global DHT once.
    """

    _instance = None

    @staticmethod
    def get_instance():
        if DHTSingleton._instance is None:
            DHTSingleton()
        return DHTSingleton._instance

    def __init__(self):
        if DHTSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DHTSingleton._instance = self
            self.dht = self._initialize_dht()

    def _initialize_dht(self):
        dht_kwargs = dict(
            # initial_peers=PUBLIC_INITIAL_PEERS,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
            ensure_bootstrap_success=True,
            parallel_rpc=4,
            # client_mode=False,
            # identity_path="./data/id.key",
        )

        print("Waiting for the DHT to initialize")
        # dht = DHT(start=True, daemon=True, await_ready=True, **dht_kwargs)
        dht = DHT(
            start=True,
            initial_peers=None,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
        )

        return dht

    def get_visible_maddrs(self):
        return self.dht.get_visible_maddrs()

    def get_dht(self):
        return self.dht
