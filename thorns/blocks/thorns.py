import torch.nn as nn
import hivemind
from hivemind import DHT
from hivemind.moe import Server
from hivemind.utils import BatchTensorDescriptor
from hivemind.moe.server.layers import name_to_block
from hivemind.moe.server import (
    ModuleBackend,
    Server,
    background_server,
    declare_experts,
    get_experts,
)
from ..layers.attention import ThornsAttention
from ..layers.mlp import ThornsMLP


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

dht = DHT(
    start=True, initial_peers=None, use_auto_relay=True, use_relay=True, use_ipfs=True
)


class ThornsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = ThornsAttention(config)

        experts = {}
        for i in range(1):
            expert = name_to_block["thorn"](config)
            experts[f"expert.{i}"] = ModuleBackend(
                name=f"expert.{i}",
                module=expert,
                # optimizer=torch.optim.Adam(expert.parameters()),
                args_schema=(BatchTensorDescriptor(config.n_embd),),
                outputs_schema=BatchTensorDescriptor(config.n_embd),
                max_batch_size=16,
            )

        server = Server(dht, experts, num_connection_handlers=1)
        server.start()
        server.ready.wait()
        self.dht = DHT(
            initial_peers=dht.get_visible_maddrs(),
            start=True,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
        )
        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.mlp = get_experts(self.dht, ["expert.0"])[0]

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x
