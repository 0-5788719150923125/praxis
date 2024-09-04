import hivemind
import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.moe import Server
from hivemind.moe.client.switch_moe import RemoteSwitchMixtureOfExperts
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

dht_kwargs = dict(
    # cache_locally=False,
    # initial_peers=None,
    use_auto_relay=True,
    use_relay=True,
    use_ipfs=True,
    ensure_bootstrap_success=False,
    # identity_path="./data/identity.key",
)

dht = DHT(start=True, **dht_kwargs)

dht_kwargs["initial_peers"] = dht.get_visible_maddrs()

import time

print("waiting for the DHT to propagate the network")

time.sleep(5)


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
                args_schema=(BatchTensorDescriptor(config.n_embd),),
                outputs_schema=BatchTensorDescriptor(config.n_embd),
                max_batch_size=16,
            )

        server = Server(dht, experts, num_connection_handlers=1, start=True)
        # # server.start()
        # # server.ready.wait()
        self.dht = DHT(
            start=True,
            **dht_kwargs,
            # initial_peers=dht.get_visible_maddrs(),
        )
        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        # self.mlp = get_experts(self.dht, ["expert.0"])[0]
        # features = 256
        # x = torch.randn(4, 128, 256)
        # batch_size, time_steps, features = x.shape
        self.moe = RemoteSwitchMixtureOfExperts(
            in_features=config.n_embd,
            grid_size=(3,),
            dht=self.dht,
            uid_prefix="expert.",
            jitter_eps=0.1,
            forward_timeout=0.1,
            backward_timeout=0.1,
            allow_zero_outputs=True,
            k_best=2,
        )

        # For some reason, this part is required, else Hivemind will fail in the forward passes
        out, balancing_loss = self.moe(torch.randn(16, config.n_embd))

    def forward(self, x, attention_mask=None):
        # Self-Attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        # MoE of Feedforward layers
        residual = x
        x = self.mlp_norm(x)
        batch_size, time_steps, features = x.shape
        # Reshape to (batch_size * time_steps, features)
        x_reshaped = x.view(-1, features)

        # Apply MoE
        x, balancing_loss = self.moe(x_reshaped)

        # Reshape back to (batch_size, time_steps, features)
        x = x.view(batch_size, time_steps, features)
        x = residual + x

        return x, balancing_loss
