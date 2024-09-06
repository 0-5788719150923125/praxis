import random
import time

import hivemind
import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.moe import Server
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

# from hivemind.moe.client.switch_moe import (
#     RemoteMixtureOfExperts,
#     RemoteSwitchMixtureOfExperts,
# )
from .switch_moe import RemoteSwitchMixtureOfExperts


class PraxisBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = PraxisAttention(config)

        n_experts = 3
        experts = {}
        for i in range(n_experts):
            expert = name_to_block["praxis_mlp"](config)
            experts[f"expert.{i}"] = ModuleBackend(
                name=f"expert.{i}",
                module=expert,
                # optimizer=torch.optim.Adam(expert.parameters()),
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

        global_dht = DHTSingleton.get_instance()

        server = Server(
            global_dht.get_dht(), experts, num_connection_handlers=4, start=True
        )

        self.dht = DHT(
            start=True,
            daemon=True,
            await_ready=True,
            initial_peers=global_dht.get_visible_maddrs(),
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
            ensure_bootstrap_success=True,
            parallel_rpc=4,
        )

        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.moe = RemoteSwitchMixtureOfExperts(
            in_features=config.n_embd,
            grid_size=(3,),
            dht=self.dht,
            uid_prefix=f"expert.",
            jitter_eps=0.01,
            forward_timeout=30.0,
            backward_timeout=30.0,
            allow_zero_outputs=False,
            k_best=2,
        )

        # TODO: For some reason, this part is required, else Hivemind will fail in the forward passes
        out = self.moe(torch.randn(16, config.n_embd))

    def forward(self, x, attention_mask=None):
        # Self-Attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x

        # Mixture of feedforward experts
        residual = x
        x = self.mlp_norm(x)
        batch_size, time_steps, features = x.shape

        # Reshape to (batch_size * time_steps, features)
        x_reshaped = x.view(-1, features)

        # Apply MoE
        x, balancing_loss = self.moe(x_reshaped.to(x.device))

        # Reshape back to (batch_size, time_steps, features)
        x = x.view(batch_size, time_steps, features)
        x = residual + x
        # balancing_loss = 0
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
        dht = DHT(start=True, daemon=True, await_ready=True, **dht_kwargs)

        return dht

    def get_visible_maddrs(self):
        return self.dht.get_visible_maddrs()

    def get_dht(self):
        return self.dht
