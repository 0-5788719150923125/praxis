import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import DHT
from hivemind.moe import (
    ModuleBackend,
    RemoteExpert,
    Server,
    get_experts,
    register_expert_class,
)
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

    def forward(self, x, attention_mask):
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
dht.get_visible_maddrs()
hidden_schema = BatchTensorDescriptor(
    64,
)
attention_schema = BatchTensorDescriptor(
    64,
)

expert_uids = ["expert.0", "expert.1"]
my_name = None
their_name = None


new_name = expert_uids[0]
new_expert = get_experts(dht, [new_name])[0]
if new_expert is None:
    my_name = expert_uids[0]
    their_name = expert_uids[1]
else:
    my_name = expert_uids[1]
    their_name = expert_uids[0]

backends = {
    my_name: ModuleBackend(
        name=my_name,
        module=name_to_block["perceptron"](64),
        args_schema=(
            hidden_schema,
            attention_schema,
        ),
        outputs_schema=(hidden_schema),
        max_batch_size=64,
        start=True,
    )
}
server = Server(
    dht,
    backends,
)
server.run_in_background(timeout=5)


my_expert = backends[my_name].module

print(my_name)
print(their_name)

while True:
    print("waiting...")
    their_expert = get_experts(dht, [their_name])[0]
    time.sleep(3)
    if their_expert is not None:
        print("sending inputs through their expert")
        inputs = torch.randn(4, 32, 64)
        attention_mask = torch.randn(4, 32)
        outputs = their_expert(inputs, attention_mask)
        print("received:", outputs.shape)
