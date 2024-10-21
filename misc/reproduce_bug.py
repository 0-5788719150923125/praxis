import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import DHT
from hivemind.moe import ModuleBackend, Server, register_expert_class
from hivemind.moe.server.layers import name_to_block
from hivemind.utils import BatchTensorDescriptor

sample_input = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("perceptron", sample_input)
class MultilayerPerceptron(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.layer2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.layer3 = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


PUBLIC_INITIAL_PEERS = [
    # IPv4 DNS addresses
    "/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    "/dns/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
    # IPv6 DNS addresses
    "/dns6/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    "/dns6/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
    # Reserved IPs
    "/ip4/159.89.214.152/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    "/ip4/159.203.156.48/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
]

dht = DHT(
    initial_peers=PUBLIC_INITIAL_PEERS,
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    start=True,
)
hidden_schema = BatchTensorDescriptor(
    64,
)
backends = {}
num_layers = 3
for i in range(num_layers):
    expert_name = f"expert.{i}"
    expert = ModuleBackend(
        name=expert_name,
        module=name_to_block["perceptron"](64),
        args_schema=(hidden_schema,),
        outputs_schema=(hidden_schema),
        max_batch_size=64,
        start=True,
    )
    backends[expert_name] = expert
server = Server(
    dht,
    backends,
)
server.run_in_background(timeout=5)

while True:
    print("waiting...")
    time.sleep(3)
